"""Immutable snapshots of the RobotSNAP simulator state.

This module only reorganises decoded messages into dataclasses. The simulator
values are kept as received: the mapping between Unity's world frame and the
Python world frame is **not yet validated**, so no conversion happens here. That
mapping lives in the single, isolated :func:`to_world_frame` entry point below
and nowhere else.

Two families of data meet here. The typed streams carry the robot (``/odom``)
and its lidar (``/scan``); the JSON bodies carry the rest, and they are not in
the same frame: ``/simulation/agents`` holds every agent in the **robot** frame,
while the humans of ``/simulation/state`` are in the **world** frame, next to
the run, scenario and camera fields of the same snapshot.

A body that cannot be read leaves its part of the snapshot empty instead of
raising: :func:`build_state` runs on every incoming message, and a snapshot that
may explode would take the bridge's callback down with it. Use
:mod:`robotsnap.bridge.codec` when a single body has to be parsed strictly.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from robotsnap import topics
from robotsnap.bridge.codec import (
    CodecError,
    as_bool,
    decode_agents_json,
    json_object,
)

__all__ = [
    "AgentState",
    "HumanState",
    "PointState",
    "PoseState",
    "RobotState",
    "LaserScanState",
    "SimulationState",
    "RobotSNAPState",
    "build_state",
    "to_world_frame",
]

#: Base topic names, the keys ``build_state`` expects in its ``messages`` map.
_ODOM = topics.base(topics.ODOM)
_SCAN = topics.base(topics.SCAN)
_MAP = topics.base(topics.MAP)
_AGENTS = topics.base(topics.SIMULATION_AGENTS)
_SIMULATION_STATE = topics.base(topics.SIMULATION_STATE)
_PRIMARY = topics.PRIMARY_ROBOT_ID


@dataclass(frozen=True)
class AgentState:
    """One detected agent, in the robot frame as published by Unity."""

    agent_id: str
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    visible: bool


@dataclass(frozen=True)
class PointState:
    """A position, as the JSON bodies spell it: x forward, y left, z up."""

    x: float
    y: float
    z: float


@dataclass(frozen=True)
class PoseState:
    """An object pose: a position plus its heading around z, in radians."""

    x: float
    y: float
    z: float
    yaw: float


@dataclass(frozen=True)
class HumanState:
    """One simulated human, from the ``humans`` list of ``/simulation/state``.

    Position, velocity and goal are in the world frame, unlike the robot-frame
    :class:`AgentState` of ``/simulation/agents``. ``speed`` is the scalar the
    simulator reports, ``goal`` is ``None`` while the human has no destination,
    and ``controller`` is ``"sfm"`` or ``"external"``.
    """

    human_id: int
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    speed: float
    goal: PointState | None
    group: str | None
    controller: str | None
    end_behavior: str | None


@dataclass(frozen=True)
class RobotState:
    """Robot pose and body velocity from ``/odom``."""

    x: float
    y: float
    z: float
    yaw: float
    linear_x: float
    angular_z: float
    stamp: float | None


@dataclass(frozen=True)
class LaserScanState:
    """Lidar geometry from ``/scan``, with raw ranges preserved."""

    angle_min: float
    angle_max: float
    angle_increment: float
    range_min: float
    range_max: float
    ranges: tuple[float, ...]
    frame_id: str
    stamp: float | None

    def points_2d(self) -> tuple[tuple[float, float], ...]:
        """Return (x, y) points in the scan frame.

        Beams that are non-finite or outside ``[range_min, range_max]`` are
        skipped.
        """
        points: list[tuple[float, float]] = []
        for index, raw_range in enumerate(self.ranges):
            distance = float(raw_range)
            if not math.isfinite(distance):
                continue
            if distance < self.range_min or distance > self.range_max:
                continue
            angle = self.angle_min + index * self.angle_increment
            points.append((distance * math.cos(angle), distance * math.sin(angle)))
        return tuple(points)


@dataclass(frozen=True)
class SimulationState:
    """The run, scenario, map and camera fields of ``/simulation/state``.

    ``simulation_state`` is the simulator's own word for the session, one of
    ``"idle"``, ``"ready"``, ``"running"`` or ``"paused"``; ``playing``,
    ``paused`` and ``stopped`` are its reading of it. The two clock values are
    ``None`` while the scene carries no clock, and the scenario identity is
    ``None`` while no scenario is loaded. The humans of the same snapshot are
    modelled apart, in :class:`HumanState`, because they are a list rather than
    a field of the run.

    ``refreshed_at`` is kept exactly as it arrived: the contract does not fix
    whether the simulator stamps that key with seconds or with a text
    timestamp.
    """

    simulation_state: str
    playing: bool
    paused: bool
    stopped: bool
    scenario_applied: bool
    sim_time_seconds: float | None
    time_scale: float | None
    scenario_id: str | None
    scenario_name: str | None
    environment: str | None
    map_name: str | None
    map_width: int
    map_height: int
    map_resolution: float
    map_origin_x: float
    map_origin_y: float
    human_count: int
    robot_has_goal: bool
    robot_goal: PointState | None
    camera_focus: str | None
    camera_focus_is_robot: bool
    camera_focus_is_human: bool
    camera_focus_agent_id: int | None
    camera_tool: str | None
    camera_mode: str | None
    camera_pose: PoseState | None
    refreshed_at: Any


@dataclass(frozen=True)
class RobotSNAPState:
    """Aggregated state of the robot, its world and its neighbours."""

    robot: RobotState | None
    agents: tuple[AgentState, ...]
    laser: LaserScanState | None
    humans: tuple[HumanState, ...]
    simulation: SimulationState | None
    map_received: bool
    agents_received_at: float | None


def to_world_frame(x: float, y: float, z: float = 0.0) -> tuple[float, float, float]:
    """Reserved home for the simulator-to-world frame mapping.

    Unity is left-handed (x right, y up, z forward) and the publishers convert
    to ROS FLU, but that conversion has not been validated against Python yet.
    Until it is, this function is the only place allowed to define the mapping.
    """
    raise NotImplementedError("simulator-to-world frame mapping is not defined yet")


def build_state(messages: dict[str, Any]) -> RobotSNAPState:
    """Build a :class:`RobotSNAPState` from decoded messages. Never raises.

    ``messages`` is keyed by base topic name (:func:`robotsnap.topics.base`:
    ``odom``, ``scan``, ``map``, ``simulation/agents``, ``simulation/state``,
    ...). The typed streams hold decoded messages; the JSON ones may hold either
    a decoded ``std_msgs/String`` or the raw ``data`` string, and a body that is
    not the expected JSON object leaves its part of the snapshot empty.
    """
    agents, agents_received_at = _build_agents(messages)
    humans, simulation = _build_simulation(_match(messages, _SIMULATION_STATE))

    return RobotSNAPState(
        robot=_build_robot(_match(messages, _ODOM)),
        agents=agents,
        laser=_build_laser(_match(messages, _SCAN)),
        humans=humans,
        simulation=simulation,
        map_received=_match(messages, _MAP) is not None,
        agents_received_at=agents_received_at,
    )


def _build_robot(odom) -> RobotState | None:
    if odom is None:
        return None
    pose = odom.pose.pose
    twist = odom.twist.twist
    return RobotState(
        x=float(pose.position.x),
        y=float(pose.position.y),
        z=float(pose.position.z),
        yaw=_yaw_from_quaternion(pose.orientation),
        linear_x=float(twist.linear.x),
        angular_z=float(twist.angular.z),
        stamp=_stamp_from_header(odom.header),
    )


def _build_laser(scan) -> LaserScanState | None:
    if scan is None:
        return None
    return LaserScanState(
        angle_min=float(scan.angle_min),
        angle_max=float(scan.angle_max),
        angle_increment=float(scan.angle_increment),
        range_min=float(scan.range_min),
        range_max=float(scan.range_max),
        ranges=tuple(float(value) for value in scan.ranges),
        frame_id=str(scan.header.frame_id),
        stamp=_stamp_from_header(scan.header),
    )


def _build_agents(messages: dict[str, Any]):
    """Every agent of ``/simulation/agents``, in the robot frame."""
    source = _match(messages, _AGENTS)
    if source is None:
        return (), None

    try:
        body = decode_agents_json(source)
    except CodecError:
        return (), None

    agents = tuple(
        AgentState(
            agent_id=entry["id"],
            x=entry["x"],
            y=entry["y"],
            z=entry["z"],
            vx=entry["vx"],
            vy=entry["vy"],
            vz=entry["vz"],
            visible=entry["visible"],
        )
        for entry in body["agents"]
    )
    # std_msgs/String has no header: only the arrival time is available.
    return agents, time.monotonic()


def _build_simulation(source):
    """World-frame humans and run state from one ``/simulation/state`` body."""
    body = json_object(source)
    if body is None:
        return (), None

    humans = tuple(
        human
        for human in (_build_human(entry) for entry in _as_list(body.get("humans")))
        if human is not None
    )
    return humans, SimulationState(
        simulation_state=_as_str(body.get("simulation_state")),
        playing=as_bool(body.get("playing")),
        paused=as_bool(body.get("paused")),
        stopped=as_bool(body.get("stopped")),
        scenario_applied=as_bool(body.get("scenario_applied")),
        sim_time_seconds=_as_optional_float(body.get("sim_time_seconds")),
        time_scale=_as_optional_float(body.get("time_scale")),
        scenario_id=_as_optional_str(body.get("scenario_id")),
        scenario_name=_as_optional_str(body.get("scenario_name")),
        environment=_as_optional_str(body.get("environment")),
        map_name=_as_optional_str(body.get("map_name")),
        map_width=_as_int(body.get("map_width")),
        map_height=_as_int(body.get("map_height")),
        map_resolution=_as_float(body.get("map_resolution")),
        map_origin_x=_as_float(body.get("map_origin_x")),
        map_origin_y=_as_float(body.get("map_origin_y")),
        human_count=_as_int(body.get("human_count")),
        robot_has_goal=as_bool(body.get("robot_has_goal")),
        robot_goal=_build_point(body.get("robot_goal")),
        camera_focus=_as_optional_str(body.get("camera_focus")),
        camera_focus_is_robot=as_bool(body.get("camera_focus_is_robot")),
        camera_focus_is_human=as_bool(body.get("camera_focus_is_human")),
        camera_focus_agent_id=_as_optional_int(body.get("camera_focus_agent_id")),
        camera_tool=_as_optional_str(body.get("camera_tool")),
        camera_mode=_as_optional_str(body.get("camera_mode")),
        camera_pose=_build_pose(body.get("camera_pose")),
        refreshed_at=body.get("refreshed_at"),
    )


def _build_human(entry) -> HumanState | None:
    """One world-frame entry of the ``humans`` list, or None when unusable."""
    if not isinstance(entry, Mapping) or entry.get("id") is None:
        return None
    try:
        human_id = int(entry["id"])
    except (TypeError, ValueError):
        return None
    return HumanState(
        human_id=human_id,
        x=_as_float(entry.get("x")),
        y=_as_float(entry.get("y")),
        z=_as_float(entry.get("z")),
        vx=_as_float(entry.get("vx")),
        vy=_as_float(entry.get("vy")),
        vz=_as_float(entry.get("vz")),
        speed=_as_float(entry.get("speed")),
        goal=_build_point(entry.get("goal")),
        group=_as_optional_str(entry.get("group")),
        controller=_as_optional_str(entry.get("controller")),
        end_behavior=_as_optional_str(entry.get("end_behavior")),
    )


def _build_point(value) -> PointState | None:
    """A ``{x, y, z}`` JSON object, or None for anything else including null."""
    if not isinstance(value, Mapping):
        return None
    return PointState(
        x=_as_float(value.get("x")),
        y=_as_float(value.get("y")),
        z=_as_float(value.get("z")),
    )


def _build_pose(value) -> PoseState | None:
    """A ``{x, y, z, yaw}`` JSON object, or None for anything else."""
    if not isinstance(value, Mapping):
        return None
    return PoseState(
        x=_as_float(value.get("x")),
        y=_as_float(value.get("y")),
        z=_as_float(value.get("z")),
        yaw=_as_float(value.get("yaw")),
    )


def _as_list(value) -> list:
    return value if isinstance(value, list) else []


def _as_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _as_optional_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _as_optional_int(value) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_str(value) -> str:
    return "" if value is None else str(value)


def _as_optional_str(value) -> str | None:
    return None if value is None else str(value)


def _yaw_from_quaternion(quaternion) -> float:
    x = float(quaternion.x)
    y = float(quaternion.y)
    z = float(quaternion.z)
    w = float(quaternion.w)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _stamp_from_header(header) -> float | None:
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return None
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _match(messages: dict[str, Any], name: str):
    """The message of one stream, resolving a fleet's names deterministically.

    A fleet publishes ``odom``, ``robot_1/odom``, ``robot_2/odom`` and so on for
    the same kind of stream. The bare name belongs to the primary robot, so it
    wins. When a scenario renamed its first robot the namespaced names are all
    equivalent, and the primary id is preferred, then the names sorted, so the
    snapshot never reports whichever robot happened to register first.
    """
    value = messages.get(name)
    if value is not None:
        return value

    namespaced = sorted(
        key
        for key in messages
        if isinstance(key, str) and key.lstrip("/").endswith("/" + name)
    )
    if not namespaced:
        return None

    preferred = f"{_PRIMARY}/{name}"
    return messages[preferred] if preferred in namespaced else messages[namespaced[0]]
