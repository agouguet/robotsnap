"""What the client holds, read as drawable rows.

The window never reads a message for itself: it comes here for the robots to
draw, the crowd with the frame its poses are in, and the lidar points of the
last scan. Each reader tolerates a topic that has not arrived yet and returns
an empty answer rather than raising.
"""

from __future__ import annotations

from robotsnap.topics import ODOM, robot_topic
from robotsnap.viewer.colours import ROBOT_BODY, ROBOT_FRAME, ROBOT_PALETTE, WORLD_FRAME
from robotsnap.viewer.geometry import _robot_pose, _state_pose


def _robots(client, snapshot: dict, legacy_pose):
    """Every robot to draw, as ``(id, pose, colour)`` triples.

    A snapshot that carries the ``robots`` roster is drawn robot by robot: each
    pose comes from the robot's own ``/robot_<id>/odom`` when that stream has
    arrived, and from its state entry otherwise. The primary robot keeps its
    colour; the others share :data:`ROBOT_PALETTE`. A snapshot from before the
    roster - the old single ``robot`` key - still draws that robot, unlabelled.
    """
    entries = snapshot.get("robots")
    if not isinstance(entries, list) or not entries:
        if legacy_pose is None:
            return []
        return [(None, legacy_pose, ROBOT_BODY)]

    dicts = [entry for entry in entries if isinstance(entry, dict)]
    any_primary = any(bool(entry.get("is_primary")) for entry in dicts)
    robots = []
    for index, entry in enumerate(dicts):
        robot_id = entry.get("id")
        primary = bool(entry.get("is_primary")) or (not any_primary and index == 0)
        pose = legacy_pose if primary and legacy_pose is not None else None
        if pose is None and robot_id is not None:
            pose = _robot_pose(client.last_message(robot_topic(robot_id, ODOM)))
        if pose is None:
            pose = _state_pose(entry)
        if pose is None:
            continue
        colour = (
            ROBOT_BODY if primary else ROBOT_PALETTE[index % len(ROBOT_PALETTE)]
        )
        robots.append((None if robot_id is None else str(robot_id), pose, colour))
    return robots


def _agents(client, snapshot: dict):
    """Every agent to draw, with the frame its poses are in.

    The crowd is read from the ``humans`` of the state snapshot, which is in the
    world frame and travels with the robot pose of the same instant, so the two
    cannot disagree. ``/simulation/agents`` carries the same crowd relative to
    the robot and is the only place the lidar-visibility flag comes from, but it
    is a ``std_msgs/String`` with no stamp: placing its entries with whatever
    pose is current when they arrive turns the whole crowd by however much the
    robot has turned since, which is exactly what it does when it spins. Its
    flags are folded in here, by id, instead.
    """
    payload = client.agents()
    visible = _agent_visibility(payload)
    humans = snapshot.get("humans")
    if isinstance(humans, list):
        entries = []
        for entry in humans:
            if not isinstance(entry, dict):
                continue
            entry = dict(entry)
            flag = visible.get(str(entry.get("id")))
            if flag is not None:
                entry["visible"] = flag
            entries.append(entry)
        if entries:
            return entries, WORLD_FRAME
    return _agent_entries(payload), _agent_frame(payload)


def _agent_entries(payload) -> list[dict]:
    """The ``agents`` of a ``/simulation/agents`` body, as a list of dicts."""
    entries = payload.get("agents") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        return []
    return [entry for entry in entries if isinstance(entry, dict)]


def _agent_frame(payload) -> str:
    """The frame a ``/simulation/agents`` body announces, robot by default."""
    frame = (payload or {}).get("frame") if isinstance(payload, dict) else None
    return str(frame or ROBOT_FRAME).lower()


def _agent_visibility(payload) -> dict[str, bool]:
    """The lidar-visibility flag of every agent of a payload, keyed by its id."""
    flags: dict[str, bool] = {}
    for entry in _agent_entries(payload):
        if entry.get("id") is None:
            continue
        flags[str(entry["id"])] = bool(entry.get("visible", True))
    return flags


def _laser_scan(client):
    """The last ``/scan`` the bridge decoded, or ``None``.

    The message itself is kept, not only its points: it is the scan that carries
    the stamp the placement needs.
    """
    return client.bridge.snapshot().laser


def _laser_points(scan) -> tuple[tuple[float, float], ...]:
    """Lidar ``(x, y)`` points of a scan, in the scan frame, possibly empty.

    The scan is fixed to the robot, so its points are placed with the robot
    pose like the agents. ``LaserScanState.points_2d()`` filters and projects
    the beams, so the viewer never re-reads a raw range array of its own.
    """
    return () if scan is None else tuple(scan.points_2d())
