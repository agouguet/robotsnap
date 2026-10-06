"""Topic names and message types exchanged with the Unity simulator.

This module is the single source of truth for both: the bridge seeds its
typestore from :data:`TOPIC_TYPES`, the client addresses its commands with the
name constants, and a viewer reads the same names instead of carrying strings
of its own. No other module in the package writes a topic name as a literal.

Names are the ROS2 ones, leading slash included, because that is what a Unity
peer registers and what a reader sees in ``rosbag``. The bridge keys its
storage by :func:`base`, the same name without the leading slash.

Every stream of one robot also exists under ``/robot_<id>/<topic>``, the
per-robot name :func:`robot_topic` builds; the unprefixed name above reaches the
primary robot only.

Every pose and velocity in these bodies is already in the ROS frame Unity
publishes in (x forward, y left, z up, yaw in radians); the conversion from
Unity's own frame happens inside the simulator.
"""

from __future__ import annotations

__all__ = [
    "CLOCK",
    "ODOM",
    "SCAN",
    "MAP",
    "CMD_VEL",
    "SIMULATION_STATE",
    "SIMULATION_AGENTS",
    "SIMULATION_CONTROL",
    "SIMULATION_CONTROL_RESULT",
    "RESET_DONE",
    "PRIMARY_ROBOT_ID",
    "STRING_TYPE",
    "BOOL_TYPE",
    "CLOCK_TYPE",
    "ODOMETRY_TYPE",
    "LASER_SCAN_TYPE",
    "OCCUPANCY_GRID_TYPE",
    "TWIST_TYPE",
    "TOPIC_TYPES",
    "JSON_TOPICS",
    "base",
    "base_types",
    "robot_topic",
]

# -- message types ----------------------------------------------------------
# rosbags' type names: "pkg/msg/Type", the ROS2 form.

#: ``std_msgs/String``; every JSON body below travels in one of these.
STRING_TYPE = "std_msgs/msg/String"
BOOL_TYPE = "std_msgs/msg/Bool"
CLOCK_TYPE = "rosgraph_msgs/msg/Clock"
ODOMETRY_TYPE = "nav_msgs/msg/Odometry"
LASER_SCAN_TYPE = "sensor_msgs/msg/LaserScan"
OCCUPANCY_GRID_TYPE = "nav_msgs/msg/OccupancyGrid"
TWIST_TYPE = "geometry_msgs/msg/Twist"

# -- topics -----------------------------------------------------------------

#: Simulation clock, published by the simulator.
CLOCK = "/clock"
#: Robot pose and twist.
ODOM = "/odom"
#: Lidar scan.
SCAN = "/scan"
#: Occupancy grid of the applied scenario: walls 100, free 0, image order.
MAP = "/map"
#: Robot velocity command, consumed by the simulator.
CMD_VEL = "/cmd_vel"
#: Whole session snapshot as JSON (see ``robotsnap.bridge.state``).
SIMULATION_STATE = "/simulation/state"
#: Every agent, in the robot frame, as JSON.
SIMULATION_AGENTS = "/simulation/agents"
#: Every command, session and crowd alike, as JSON.
SIMULATION_CONTROL = "/simulation/control"
#: The answer to one command, as JSON.
SIMULATION_CONTROL_RESULT = "/simulation/control_result"
#: Published once the world is ready, the handshake a client waits for.
RESET_DONE = "/reset_done"

#: Id the simulator gives the robot a bare topic name reaches. It mirrors
#: ``RobotRoster.PrimaryId`` on the Unity side, and it is what a fleet's
#: namespaced names are resolved against when the bare one is not published.
PRIMARY_ROBOT_ID = "robot_1"

#: The fixed surface of the simulator, topic -> message type.
TOPIC_TYPES: dict[str, str] = {
    CLOCK: CLOCK_TYPE,
    ODOM: ODOMETRY_TYPE,
    SCAN: LASER_SCAN_TYPE,
    MAP: OCCUPANCY_GRID_TYPE,
    CMD_VEL: TWIST_TYPE,
    SIMULATION_STATE: STRING_TYPE,
    SIMULATION_AGENTS: STRING_TYPE,
    SIMULATION_CONTROL: STRING_TYPE,
    SIMULATION_CONTROL_RESULT: STRING_TYPE,
    RESET_DONE: BOOL_TYPE,
}

#: Topics whose ``std_msgs/String`` body is a JSON object.
JSON_TOPICS: tuple[str, ...] = (
    SIMULATION_STATE,
    SIMULATION_AGENTS,
    SIMULATION_CONTROL,
    SIMULATION_CONTROL_RESULT,
)


def base(topic: str) -> str:
    """Return a topic name without its leading slash.

    The bridge stores and looks up topics under this form, so ``/odom`` and
    ``odom`` are the same key.
    """
    return str(topic).strip().lstrip("/")


def robot_topic(robot_id: str, topic: str) -> str:
    """Return the per-robot name of ``topic`` for ``robot_id``.

    The rule of the contract is ``/robot_<id>/<topic>``: every stream of a robot
    answers under that name, and the unprefixed name reaches the primary robot
    only. A leading or trailing slash on either argument is tolerated, and an id
    that already carries its ``robot_`` namespace is not prefixed twice, so
    ``robot_topic("/robot_2/", "/scan")`` and ``robot_topic("robot_2", "scan")``
    are both ``/robot_2/scan``.
    """
    name = str(robot_id).strip().strip("/")
    if name and not name.startswith("robot_"):
        name = "robot_" + name
    leaf = str(topic).strip().strip("/")
    return "/" + name + "/" + leaf


def base_types() -> dict[str, str]:
    """Return :data:`TOPIC_TYPES` keyed by :func:`base` name, for a bridge."""
    return {base(topic): msg_type for topic, msg_type in TOPIC_TYPES.items()}
