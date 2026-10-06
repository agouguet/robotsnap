"""One read of a running session, as arrays an environment can use.

This module is the seam between the simulator and a Gymnasium environment: it
takes a :class:`~robotsnap.client.RobotSNAPClient`, reads the streams of one
robot, and returns a :class:`World` - plain dataclasses and numpy arrays, no
Unity, no sockets, no policy. An environment built on top of it never touches a
rosbags message, and a test can build a world by hand.

Two frames meet here and are kept apart on purpose:

- every published pose is in the **ROS frame** the contract fixes, x forward,
  y left, yaw counter-clockwise in radians, so ``World`` is in that frame too;
- the agents of ``/simulation/agents`` are already in the **robot frame**, which
  is what a perception-driven observation wants.

The occupancy grid is kept as a small object of its own because a collision, an
out-of-bounds test and a local map all need the same lookup, and because the
cell order of that grid (columns along +x, rows along +y, from ``info.origin``)
is exactly the kind of thing that goes wrong silently.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from robotsnap import topics
from robotsnap.bridge.state import build_state

__all__ = [
    "Agent",
    "Human",
    "LidarScan",
    "OccupancyMap",
    "Pose2D",
    "World",
    "read_world",
    "wrap_angle",
]

#: Occupancy value from which a cell counts as an obstacle.
DEFAULT_OCCUPANCY_THRESHOLD = 50


def wrap_angle(angle: float) -> float:
    """Fold an angle into ``[-pi, pi)``, the range a heading is compared in."""
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class Pose2D:
    """A planar pose, in the ROS frame, yaw counter-clockwise in radians."""

    x: float
    y: float
    yaw: float

    def distance_to(self, x: float, y: float) -> float:
        return math.hypot(self.x - float(x), self.y - float(y))

    def bearing_to(self, x: float, y: float) -> float:
        """Heading of the point relative to this pose, in ``[-pi, pi)``.

        Zero means the point is straight ahead, positive means to the left, which
        is the convention the rest of the contract uses.
        """
        return wrap_angle(math.atan2(float(y) - self.y, float(x) - self.x) - self.yaw)


@dataclass(frozen=True)
class LidarScan:
    """Lidar geometry of one robot, with the raw ranges kept.

    ``angles`` is the bearing of each beam in the robot frame: beam ``i`` of the
    message sits at ``angle_min + i * angle_increment``, which is the order the
    publisher writes once its own mirroring is undone. A beam that saw nothing
    is ``inf`` in ``ranges``, as ROS sends it.
    """

    ranges: np.ndarray
    angles: np.ndarray
    range_min: float
    range_max: float
    stamp: float | None = None

    @classmethod
    def from_state(cls, scan: Any) -> "LidarScan":
        """Build from the ``LaserScanState`` of :mod:`robotsnap.bridge.state`."""
        ranges = np.asarray(scan.ranges, dtype=np.float32)
        angles = scan.angle_min + np.arange(ranges.size, dtype=np.float32) * np.float32(
            scan.angle_increment
        )
        return cls(
            ranges=ranges,
            angles=angles,
            range_min=float(scan.range_min),
            range_max=float(scan.range_max),
            stamp=scan.stamp,
        )

    @property
    def beams(self) -> int:
        return int(self.ranges.size)

    def hits(self) -> np.ndarray:
        """Ranges of the beams that returned something inside the sensor range."""
        ranges = self.ranges
        return ranges[np.isfinite(ranges) & (ranges >= self.range_min)]

    def min_range(self) -> float:
        """Shortest hit, or ``range_max`` when the sweep saw nothing."""
        finite = self.ranges[np.isfinite(self.ranges)]
        if finite.size == 0:
            return float(self.range_max)
        return float(min(float(finite.min()), float(self.range_max)))

    def points_2d(self) -> np.ndarray:
        """``(N, 2)`` hit points in the robot frame, beams that returned only."""
        ranges = self.ranges
        keep = np.isfinite(ranges) & (ranges >= self.range_min) & (
            ranges <= self.range_max
        )
        if not keep.any():
            return np.zeros((0, 2), dtype=np.float32)
        distance = ranges[keep]
        angle = self.angles[keep]
        return np.stack(
            (distance * np.cos(angle), distance * np.sin(angle)), axis=1
        ).astype(np.float32)

    def resample(self, bins: int) -> np.ndarray:
        """The sweep reduced to ``bins`` equal slices, the nearest hit of each.

        A slice that saw nothing reads :attr:`range_max`, so the vector means
        "distance to the closest obstacle in that direction, or the sensor's
        reach". This is what turns a scan of a few hundred beams into a fixed
        vector a network can take, without dropping a thin obstacle the way
        picking every n-th beam would.
        """
        if bins <= 0:
            raise ValueError("bins must be positive")
        out = np.full(int(bins), float(self.range_max), dtype=np.float32)
        usable = np.where(np.isfinite(self.ranges), self.ranges, np.inf)
        usable = np.where(usable >= self.range_min, usable, np.inf)
        total = usable.size
        if total == 0:
            return out
        edges = np.linspace(0, total, int(bins) + 1).astype(int)
        for index in range(int(bins)):
            start, stop = int(edges[index]), int(edges[index + 1])
            if stop <= start:
                continue
            slice_min = float(usable[start:stop].min())
            if math.isfinite(slice_min):
                out[index] = min(slice_min, float(self.range_max))
        return out


@dataclass(frozen=True)
class OccupancyMap:
    """The occupancy grid of the applied scenario, with its geometry.

    ``cells`` is ``(height, width)``: a column counts along ROS +x and a row
    along ROS +y, both growing away from ``origin_x``/``origin_y``, the corner
    the grid counts from. Walls are 100 and free cells 0, as the contract says.
    """

    cells: np.ndarray
    resolution: float
    origin_x: float
    origin_y: float

    @classmethod
    def from_message(cls, message: Any) -> "OccupancyMap":
        """Build from a decoded ``nav_msgs/OccupancyGrid``."""
        info = message.info
        width, height = int(info.width), int(info.height)
        cells = np.asarray(message.data, dtype=np.int8)
        if cells.size != width * height:
            raise ValueError(
                f"the grid holds {cells.size} cells but announces {width}x{height}"
            )
        if float(info.resolution) <= 0.0:
            raise ValueError("an occupancy grid needs a positive resolution")
        return cls(
            cells=cells.reshape(height, width),
            resolution=float(info.resolution),
            origin_x=float(info.origin.position.x),
            origin_y=float(info.origin.position.y),
        )

    @property
    def height(self) -> int:
        return int(self.cells.shape[0])

    @property
    def width(self) -> int:
        return int(self.cells.shape[1])

    def cell(self, x: float, y: float) -> tuple[int, int] | None:
        """``(column, row)`` of a point, or ``None`` when it is off the grid."""
        column = int(math.floor((float(x) - self.origin_x) / self.resolution))
        row = int(math.floor((float(y) - self.origin_y) / self.resolution))
        if 0 <= column < self.width and 0 <= row < self.height:
            return column, row
        return None

    def occupancy(self, x: float, y: float) -> int | None:
        """Occupancy of the cell of a point, or ``None`` when it is off the grid."""
        found = self.cell(x, y)
        if found is None:
            return None
        column, row = found
        return int(self.cells[row, column])

    def is_occupied(
        self, x: float, y: float, threshold: int = DEFAULT_OCCUPANCY_THRESHOLD
    ) -> bool:
        """Whether the cell of a point holds an obstacle."""
        value = self.occupancy(x, y)
        return value is not None and value >= int(threshold)

    def bounds(self) -> tuple[float, float, float, float]:
        """``(min x, min y, max x, max y)`` of the grid, in the ROS frame."""
        return (
            self.origin_x,
            self.origin_y,
            self.origin_x + self.width * self.resolution,
            self.origin_y + self.height * self.resolution,
        )

    def contains(self, x: float, y: float) -> bool:
        """Whether a point falls inside the grid."""
        return self.cell(x, y) is not None


@dataclass(frozen=True)
class Agent:
    """One agent as the robot sees it, in the robot frame."""

    agent_id: str
    x: float
    y: float
    vx: float
    vy: float
    visible: bool

    @property
    def distance(self) -> float:
        return math.hypot(self.x, self.y)

    @property
    def bearing(self) -> float:
        return wrap_angle(math.atan2(self.y, self.x))

    @property
    def speed(self) -> float:
        return math.hypot(self.vx, self.vy)


@dataclass(frozen=True)
class Human:
    """One simulated human of the snapshot, in the world (ROS) frame."""

    human_id: int
    x: float
    y: float
    vx: float
    vy: float
    speed: float
    controller: str | None = None

    @property
    def speed_from_velocity(self) -> float:
        return math.hypot(self.vx, self.vy)


@dataclass(frozen=True)
class World:
    """What one read of the session says about one robot and its surroundings.

    Every field is ``None`` or empty until the stream behind it has arrived:
    ``pose`` before the first ``/odom``, ``scan`` before the first ``/scan``,
    ``map`` before the first ``/map``. An environment waits for the streams it
    cannot do without at reset, and treats a missing one as a missing sensor
    afterwards rather than as an error.
    """

    robot_id: str | None = None
    pose: Pose2D | None = None
    linear_velocity: float = 0.0
    angular_velocity: float = 0.0
    goal: Pose2D | None = None
    scan: LidarScan | None = None
    agents: tuple[Agent, ...] = ()
    humans: tuple[Human, ...] = ()
    map: OccupancyMap | None = None
    sim_time_seconds: float | None = None
    time_scale: float | None = None
    scenario_id: str | None = None
    scenario_applied: bool = False
    wall_time: float = 0.0

    @property
    def distance_to_goal(self) -> float | None:
        if self.pose is None or self.goal is None:
            return None
        return self.pose.distance_to(self.goal.x, self.goal.y)

    @property
    def bearing_to_goal(self) -> float | None:
        if self.pose is None or self.goal is None:
            return None
        return self.pose.bearing_to(self.goal.x, self.goal.y)

    def nearest_agent(self) -> Agent | None:
        if not self.agents:
            return None
        return min(self.agents, key=lambda agent: agent.distance)


def _topic_for(robot: str | None, topic: str) -> str:
    return topic if robot is None else topics.robot_topic(robot, topic)


def read_world(client, robot: str | None = None) -> World:
    """Read one robot and its surroundings from a live session.

    Never raises on a missing stream: a topic that has not arrived leaves its
    field empty, which is what a caller that starts before the world is built
    needs. A malformed body is the client's business, and it is recorded in
    ``client.last_error`` there.
    """
    body = client.snapshot()
    if not isinstance(body, dict):
        body = {}

    odom_topic = _topic_for(robot, topics.ODOM)
    scan_topic = _topic_for(robot, topics.SCAN)
    typed = build_state(
        {
            topics.base(odom_topic): client.last_message(odom_topic),
            topics.base(scan_topic): client.last_message(scan_topic),
        }
    )

    entry = client.robot(robot)
    pose = None
    if isinstance(entry, dict) and entry.get("x") is not None:
        pose = Pose2D(
            float(entry.get("x") or 0.0),
            float(entry.get("y") or 0.0),
            float(entry.get("yaw") or 0.0),
        )
    elif typed.robot is not None:
        pose = Pose2D(typed.robot.x, typed.robot.y, typed.robot.yaw)

    goal = _read_goal(entry)

    return World(
        robot_id=(entry or {}).get("id") if isinstance(entry, dict) else None,
        pose=pose,
        linear_velocity=typed.robot.linear_x if typed.robot is not None else 0.0,
        angular_velocity=typed.robot.angular_z if typed.robot is not None else 0.0,
        goal=goal,
        scan=LidarScan.from_state(typed.laser) if typed.laser is not None else None,
        agents=_read_agents(client, robot),
        humans=_read_humans(body),
        map=_read_map(client),
        sim_time_seconds=_as_optional_float(body.get("sim_time_seconds")),
        time_scale=_as_optional_float(body.get("time_scale")),
        scenario_id=body.get("scenario_id"),
        scenario_applied=bool(body.get("scenario_applied")),
        wall_time=time.monotonic(),
    )


def _read_agents(client, robot: str | None) -> tuple[Agent, ...]:
    body = client.agents(robot=robot)
    entries = (body or {}).get("agents") if isinstance(body, dict) else None
    agents = []
    for entry in entries or ():
        if not isinstance(entry, dict):
            continue
        agents.append(
            Agent(
                agent_id=str(entry.get("id", "")),
                x=float(entry.get("x") or 0.0),
                y=float(entry.get("y") or 0.0),
                vx=float(entry.get("vx") or 0.0),
                vy=float(entry.get("vy") or 0.0),
                visible=bool(entry.get("visible", True)),
            )
        )
    return tuple(agents)


def _read_goal(entry: Any) -> Pose2D | None:
    """The goal of one robot, from the roster entry of the snapshot.

    Two keys can carry it, and the order matters. ``goal`` is the live one: the
    simulator fills it once the robot is driving towards something, whether a
    command set it or its own planner did. ``target_pose`` is what the scenario
    authored, and it is the one that is there at the start of an episode, since a
    scenario declares a destination without the robot having set off yet. A
    scenario with neither leaves the robot with no goal at all, which an
    environment reports rather than invents.
    """
    if not isinstance(entry, dict):
        return None
    for key in ("goal", "target_pose"):
        value = entry.get(key)
        if isinstance(value, dict) and value.get("x") is not None:
            return Pose2D(
                float(value.get("x") or 0.0),
                float(value.get("y") or 0.0),
                float(value.get("yaw") or 0.0),
            )
    return None


def _read_humans(body: dict[str, Any]) -> tuple[Human, ...]:
    humans = []
    for entry in body.get("humans") or ():
        if not isinstance(entry, dict):
            continue
        try:
            humans.append(
                Human(
                    human_id=int(entry["id"]),
                    x=float(entry.get("x") or 0.0),
                    y=float(entry.get("y") or 0.0),
                    vx=float(entry.get("vx") or 0.0),
                    vy=float(entry.get("vy") or 0.0),
                    speed=float(entry.get("speed") or 0.0),
                    controller=entry.get("controller"),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return tuple(humans)


def _read_map(client) -> OccupancyMap | None:
    message = client.last_message(topics.MAP)
    if message is None:
        return None
    try:
        return OccupancyMap.from_message(message)
    except (AttributeError, TypeError, ValueError):
        return None


def _as_optional_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)
