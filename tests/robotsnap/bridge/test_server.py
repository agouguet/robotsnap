"""Tests for the RobotSNAP Unity bridge server."""

import json
import math
import time

import numpy as np
import pytest

from robotsnap.bridge import codec
from robotsnap.bridge.server import RobotSNAPBridge
from unity_peer import UnityPeer

T = codec.TYPESTORE.types

HDR = "std_msgs/msg/Header"
TIME = "builtin_interfaces/msg/Time"
QUAT = "geometry_msgs/msg/Quaternion"
POSE = "geometry_msgs/msg/Pose"
VEC3 = "geometry_msgs/msg/Vector3"
TWIST = "geometry_msgs/msg/Twist"
ODOMETRY = "nav_msgs/msg/Odometry"
STRING = "std_msgs/msg/String"

_CONNECTION_TIMEOUT = 2.0
_POLL_INTERVAL = 0.01


# --- helpers ---------------------------------------------------------------


def _wait_for(predicate, timeout=_CONNECTION_TIMEOUT):
    """Poll ``predicate`` with short sleeps; returns its last value."""
    deadline = time.monotonic() + timeout
    while True:
        value = predicate()
        if value:
            return True
        if time.monotonic() >= deadline:
            return value
        time.sleep(_POLL_INTERVAL)


def _is_closed(sock):
    """True when the peer has closed the socket (EOF or error)."""
    try:
        sock.settimeout(0.1)
        return sock.recv(1) == b""
    except OSError:
        return True


def _header(frame_id="odom", sec=7, nanosec=500_000_000):
    return T[HDR](stamp=T[TIME](sec=sec, nanosec=nanosec), frame_id=frame_id)


def _quaternion_from_yaw(yaw):
    return T[QUAT](x=0.0, y=0.0, z=math.sin(yaw / 2.0), w=math.cos(yaw / 2.0))


def _odometry_message(x=4.0, y=-1.0, z=0.0, yaw=0.0):
    pose = T[POSE](
        position=T["geometry_msgs/msg/Point"](x=x, y=y, z=z),
        orientation=_quaternion_from_yaw(yaw),
    )
    twist = T[TWIST](
        linear=T[VEC3](x=0.4, y=0.0, z=0.0),
        angular=T[VEC3](x=0.0, y=0.0, z=0.15),
    )
    return T[ODOMETRY](
        header=_header(),
        child_frame_id="base_link",
        pose=T["geometry_msgs/msg/PoseWithCovariance"](
            pose=pose, covariance=np.zeros(36, dtype=np.float64)
        ),
        twist=T["geometry_msgs/msg/TwistWithCovariance"](
            twist=twist, covariance=np.zeros(36, dtype=np.float64)
        ),
    )


def _odometry_payload(**kwargs):
    return codec.encode(ODOMETRY, _odometry_message(**kwargs))


def _agents_json():
    """A ``/simulation/agents`` body: every agent, in the robot frame."""
    return json.dumps(
        {
            "agents": [
                {
                    "id": "human_0",
                    "x": 1.0,
                    "y": 2.0,
                    "z": 0.0,
                    "vx": 0.1,
                    "vy": 0.2,
                    "vz": 0.0,
                    "visible": True,
                },
                {
                    "id": "human_1",
                    "x": -3.0,
                    "y": 0.5,
                    "z": 0.0,
                    "vx": 0.0,
                    "vy": -0.1,
                    "vz": 0.0,
                    "visible": False,
                },
            ],
            "frame": "robot",
        }
    )


def _agents_payload():
    return codec.encode(STRING, T[STRING](data=_agents_json()))


@pytest.fixture
def bridge():
    server = RobotSNAPBridge(host="127.0.0.1", port=0)
    server.start()
    try:
        yield server
    finally:
        server.stop()


# --- tests -----------------------------------------------------------------


def test_start_binds_real_port_and_times_out(bridge):
    assert bridge.port > 0
    assert bridge.is_connected is False
    assert bridge.wait_for_connection(timeout=0.05) is False


def test_handshake_exposes_metadata(bridge):
    client = UnityPeer(bridge.port)
    try:
        assert bridge.wait_for_connection(timeout=_CONNECTION_TIMEOUT) is True
        client.handshake(version="2.0.1", metadata={"protocol": "ROS2"})
        assert _wait_for(lambda: bridge.handshake_metadata() is not None)
        assert bridge.handshake_metadata() == {"protocol": "ROS2"}
    finally:
        client.close()


def test_first_bytes_are_the_ros2_handshake(bridge):
    """Unity's reader thread requires the handshake as the first message."""
    client = UnityPeer(bridge.port)
    try:
        assert bridge.wait_for_connection(timeout=_CONNECTION_TIMEOUT) is True
        handshake = client.recv_handshake()
        assert handshake["version"].startswith("v0.7.")
        assert handshake["metadata"]["protocol"] == "ROS2"
    finally:
        client.close()


def test_odom_registration_and_snapshot(bridge):
    client = UnityPeer(bridge.port)
    try:
        client.register_publisher("/odom", "nav_msgs/Odometry")
        client.publish("/odom", _odometry_payload(x=4.0, y=-1.0, yaw=math.pi / 2.0))

        assert _wait_for(lambda: bridge.snapshot().robot is not None)
        robot = bridge.snapshot().robot
        assert robot.x == pytest.approx(4.0)
        assert robot.y == pytest.approx(-1.0)
        assert robot.z == pytest.approx(0.0)
        assert robot.yaw == pytest.approx(math.pi / 2.0, abs=1e-6)

        # Unity message names are learned and normalised to the ROS2 form.
        assert bridge.topic_types()["odom"] == ODOMETRY
        # "odom", "/odom" and a prefixed name resolve to the same message.
        assert bridge.latest("odom") is bridge.latest("/odom")
        assert bridge.latest("/robot0/odom") is bridge.latest("odom")
    finally:
        client.close()


def test_unity_ros2_frames_fill_the_robot_state(bridge):
    """End-to-end: consume the handshake, register, publish, decode to state."""
    client = UnityPeer(bridge.port)
    try:
        client.recv_handshake()
        client.register_publisher("/odom", "nav_msgs/Odometry")
        client.publish(
            "/odom", _odometry_payload(x=4.0, y=-1.0, z=0.0, yaw=math.pi / 2.0)
        )

        assert _wait_for(lambda: bridge.snapshot().robot is not None)
        robot = bridge.snapshot().robot
        assert robot.x == pytest.approx(4.0)
        assert robot.y == pytest.approx(-1.0)
        assert robot.z == pytest.approx(0.0)
        assert robot.yaw == pytest.approx(math.pi / 2.0, abs=1e-6)
        assert bridge.topic_types()["odom"] == ODOMETRY
    finally:
        client.close()


def test_agents_payload_populates_state(bridge):
    client = UnityPeer(bridge.port)
    try:
        client.register_publisher("/simulation/agents", "std_msgs/String")
        client.publish("/simulation/agents", _agents_payload())

        assert _wait_for(lambda: len(bridge.snapshot().agents) == 2)
        snapshot = bridge.snapshot()
        assert [agent.agent_id for agent in snapshot.agents] == ["human_0", "human_1"]
        assert snapshot.agents[0].x == pytest.approx(1.0)
        assert snapshot.agents[1].visible is False
        assert snapshot.agents_received_at is not None
        # The decoded String is kept as-is and handed to build_state.
        assert bridge.latest("/simulation/agents").data == _agents_json()
    finally:
        client.close()


def test_state_payload_populates_the_world_frame_humans(bridge):
    """``/simulation/state`` reaches the snapshot with its humans and run state."""
    state_json = json.dumps(
        {
            "simulation_state": "running",
            "playing": True,
            "sim_time_seconds": 3.5,
            "scenario_id": "corridor_1",
            "human_count": 1,
            "humans": [
                {
                    "id": 3,
                    "x": 1.0,
                    "y": 0.0,
                    "z": 2.0,
                    "vx": 0.1,
                    "vy": 0.0,
                    "vz": 0.0,
                    "speed": 0.1,
                    "goal": {"x": 5.0, "y": 0.0, "z": 2.0},
                    "group": "G1",
                    "controller": "sfm",
                    "end_behavior": "stop",
                }
            ],
            "robot_has_goal": False,
        }
    )
    client = UnityPeer(bridge.port)
    try:
        client.register_publisher("/simulation/state", "std_msgs/String")
        client.publish("/simulation/state", codec.encode(STRING, T[STRING](data=state_json)))

        assert _wait_for(lambda: len(bridge.snapshot().humans) == 1)
        snapshot = bridge.snapshot()
        assert snapshot.humans[0].human_id == 3
        assert snapshot.humans[0].z == pytest.approx(2.0)
        assert snapshot.simulation is not None
        assert snapshot.simulation.simulation_state == "running"
        assert snapshot.simulation.scenario_id == "corridor_1"
        assert snapshot.simulation.sim_time_seconds == pytest.approx(3.5)
        assert snapshot.simulation.robot_has_goal is False
    finally:
        client.close()


def test_send_cmd_vel_reaches_the_client(bridge):
    client = UnityPeer(bridge.port)
    try:
        assert bridge.wait_for_connection(timeout=_CONNECTION_TIMEOUT) is True
        client.recv_handshake()  # must be drained before the cmd_vel frame
        client.register_subscriber("/cmd_vel", "geometry_msgs/Twist")

        assert bridge.send_cmd_vel(0.7, -0.3) is True
        destination, payload = client.recv_frame()
        assert destination == "/cmd_vel"
        twist = codec.decode(TWIST, payload)
        assert twist.linear.x == pytest.approx(0.7)
        assert twist.angular.z == pytest.approx(-0.3)
    finally:
        client.close()


def test_send_cmd_vel_without_connection_is_false(bridge):
    assert bridge.send_cmd_vel(1.0, 0.0) is False

    client = UnityPeer(bridge.port)
    assert bridge.wait_for_connection(timeout=_CONNECTION_TIMEOUT) is True
    client.close()
    assert _wait_for(lambda: bridge.is_connected is False)
    assert bridge.send_cmd_vel(1.0, 0.0) is False


def test_send_cmd_vel_prefers_the_primary_name_of_a_fleet(bridge):
    """A fleet subscribes once per robot; a bare command still means the primary.

    The subscriptions are registered second robot first, so picking one by
    insertion order alone would drive the wrong robot.
    """
    client = UnityPeer(bridge.port)
    try:
        assert bridge.wait_for_connection(timeout=_CONNECTION_TIMEOUT) is True
        client.recv_handshake()
        client.register_subscriber("/robot_2/cmd_vel", "geometry_msgs/Twist")
        client.register_subscriber("/cmd_vel", "geometry_msgs/Twist")
        assert _wait_for(lambda: len(bridge.subscriptions()) == 2)

        assert bridge.send_cmd_vel(0.7, 0.0) is True
        destination, _ = client.recv_frame()
        assert destination == "/cmd_vel"
    finally:
        client.close()


def test_bad_frames_do_not_kill_the_session(bridge):
    client = UnityPeer(bridge.port)
    try:
        client.register_publisher("/odom", "nav_msgs/Odometry")
        client.publish("/mystery", b"\x01\x02\x03")  # unknown topic
        client.publish("/odom", b"\x01\x02\x03")  # malformed payload
        client.publish("/odom", _odometry_payload(x=2.0, y=3.0))

        assert _wait_for(lambda: bridge.snapshot().robot is not None)
        assert bridge.snapshot().robot.x == pytest.approx(2.0)
        assert bridge.is_connected is True
        assert bridge.last_error is not None
        assert bridge.latest_raw("/mystery") == b"\x01\x02\x03"
        assert bridge.latest("/odom") is not None
    finally:
        client.close()


def test_disconnect_then_reconnect(bridge):
    first = UnityPeer(bridge.port)
    first.handshake()
    assert bridge.wait_for_connection(timeout=_CONNECTION_TIMEOUT) is True
    assert _wait_for(lambda: bridge.handshake_metadata() is not None)

    first.close()
    assert _wait_for(lambda: bridge.is_connected is False)
    assert bridge.handshake_metadata() is None

    second = UnityPeer(bridge.port)
    try:
        second.register_publisher("/odom", "nav_msgs/Odometry")
        second.publish("/odom", _odometry_payload(x=1.0, y=2.0))
        assert _wait_for(lambda: bridge.is_connected is True)
        assert _wait_for(lambda: bridge.latest("/odom") is not None)
        assert bridge.snapshot().robot.x == pytest.approx(1.0)
    finally:
        second.close()


def test_second_connection_replaces_the_first(bridge):
    first = UnityPeer(bridge.port)
    try:
        assert bridge.wait_for_connection(timeout=_CONNECTION_TIMEOUT) is True
        first.recv_handshake()  # drain so EOF is visible once the peer closes

        second = UnityPeer(bridge.port)
        try:
            assert _wait_for(lambda: bridge.is_connected is True)
            assert _wait_for(lambda: _is_closed(first.sock))
            second.publish("/odom", _odometry_payload(x=9.0, y=0.0))
            assert _wait_for(lambda: bridge.snapshot().robot is not None)
            assert bridge.snapshot().robot.x == pytest.approx(9.0)
        finally:
            second.close()
    finally:
        first.close()
