"""Poses and frames: reading them off messages, and moving points between them.

Every function here answers one question about where something is: the
``(x, y)`` of a state entry, the yaw of a ROS quaternion, the sampled timestamp
of a header, the pose of ``/odom`` or of the ``robot`` entry of
``/simulation/state``, a robot speed, a robot-frame point in the world, and the
extent of an occupancy grid. Nothing here touches a screen.
"""

from __future__ import annotations

import math

from robotsnap.viewer.colours import WORLD_FRAME


def _as_float(value) -> float | None:
    """``value`` as a float, or ``None`` when it is not a number."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _ground_y(entry: dict) -> float | None:
    """Lateral coordinate of a state entry: ``y``, or the legacy ``z`` name."""
    y = _as_float(entry.get("y"))
    if y is not None:
        return y
    return _as_float(entry.get("z"))


def _world_point(entry):
    """World ``(x, y)`` of a state entry, or ``None`` when it carries no pose."""
    if not isinstance(entry, dict):
        return None
    x = _as_float(entry.get("x"))
    y = _ground_y(entry)
    if x is None or y is None:
        return None
    return (x, y)


def _yaw(quaternion) -> float:
    """Yaw of a ROS quaternion, around +z."""
    x = float(quaternion.x)
    y = float(quaternion.y)
    z = float(quaternion.z)
    w = float(quaternion.w)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _header_stamp(message) -> float | None:
    """Seconds of a message header stamp, or ``None`` when it carries none."""
    header = getattr(message, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return None
    try:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9
    except (TypeError, ValueError):
        return None


def _robot_pose(odom):
    """``(x, y, yaw)`` of the robot from a ``/odom`` message, or ``None``."""
    try:
        pose = odom.pose.pose
        return (float(pose.position.x), float(pose.position.y), _yaw(pose.orientation))
    except (AttributeError, TypeError, ValueError):
        return None


def _state_pose(robot):
    """``(x, y, yaw)`` of the ``robot`` entry of ``/simulation/state``."""
    point = _world_point(robot)
    if point is None:
        return None
    yaw = _as_float(robot.get("yaw"))
    return (point[0], point[1], yaw if yaw is not None else 0.0)


def _twist_speed(odom) -> float | None:
    """Forward speed of the robot in m/s, or ``None``."""
    try:
        linear = odom.twist.twist.linear
        return math.hypot(float(linear.x), float(linear.y))
    except (AttributeError, TypeError, ValueError):
        return None


def _to_world(x: float, y: float, pose):
    """Robot-frame ``(x, y)`` in the world frame, ``None`` without a pose."""
    if pose is None:
        return None
    robot_x, robot_y, yaw = pose
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)
    return (robot_x + cos_y * x - sin_y * y, robot_y + sin_y * x + cos_y * y)


def _place(point, frame: str, pose):
    """World ``(x, y)`` of a point given in ``frame``, or ``None``."""
    if point is None:
        return None
    if frame == WORLD_FRAME:
        return point
    return _to_world(point[0], point[1], pose)


def _map_bounds(message):
    """``(min_x, min_y, max_x, max_y)`` of an occupancy grid, or ``None``."""
    info = getattr(message, "info", None)
    width = _as_float(getattr(info, "width", None))
    height = _as_float(getattr(info, "height", None))
    resolution = _as_float(getattr(info, "resolution", None))
    origin = getattr(getattr(info, "origin", None), "position", None)
    origin_x = _as_float(getattr(origin, "x", None))
    origin_y = _as_float(getattr(origin, "y", None))
    if None in (width, height, resolution, origin_x, origin_y):
        return None
    if width < 1 or height < 1 or resolution <= 0.0:
        return None
    return (
        origin_x,
        origin_y,
        origin_x + width * resolution,
        origin_y + height * resolution,
    )
