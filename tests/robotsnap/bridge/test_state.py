"""Tests for the state aggregation layer."""

import json
import math

import numpy as np
import pytest

from robotsnap.bridge import state
from robotsnap.bridge.codec import TYPESTORE

T = TYPESTORE.types


def _header(frame_id="odom", sec=7, nanosec=500_000_000):
    return T["std_msgs/msg/Header"](
        stamp=T["builtin_interfaces/msg/Time"](sec=sec, nanosec=nanosec),
        frame_id=frame_id,
    )


def _quaternion_from_yaw(yaw):
    return T["geometry_msgs/msg/Quaternion"](
        x=0.0, y=0.0, z=math.sin(yaw / 2.0), w=math.cos(yaw / 2.0)
    )


def _odometry(yaw=0.0):
    pose = T["geometry_msgs/msg/Pose"](
        position=T["geometry_msgs/msg/Point"](x=4.0, y=-1.0, z=0.0),
        orientation=_quaternion_from_yaw(yaw),
    )
    twist = T["geometry_msgs/msg/Twist"](
        linear=T["geometry_msgs/msg/Vector3"](x=0.4, y=0.0, z=0.0),
        angular=T["geometry_msgs/msg/Vector3"](x=0.0, y=0.0, z=0.15),
    )
    return T["nav_msgs/msg/Odometry"](
        header=_header(),
        child_frame_id="base_link",
        pose=T["geometry_msgs/msg/PoseWithCovariance"](
            pose=pose, covariance=np.zeros(36, dtype=np.float64)
        ),
        twist=T["geometry_msgs/msg/TwistWithCovariance"](
            twist=twist, covariance=np.zeros(36, dtype=np.float64)
        ),
    )


def _laserscan():
    return T["sensor_msgs/msg/LaserScan"](
        header=_header("laser", sec=2, nanosec=0),
        angle_min=0.0,
        angle_max=math.pi,
        angle_increment=math.pi / 2.0,
        time_increment=0.0,
        scan_time=0.1,
        range_min=0.2,
        range_max=5.0,
        # beam 1 is inf (no return) and beam 3 is below range_min
        ranges=np.array([1.0, np.inf, 2.0, 0.1], dtype=np.float32),
        intensities=np.zeros(4, dtype=np.float32),
    )


def _agents_string():
    """``/simulation/agents``: every agent, in the robot frame."""
    body = {
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
    return T["std_msgs/msg/String"](data=json.dumps(body))


def _simulation_state_json():
    """A ``/simulation/state`` snapshot, with its two world-frame humans."""
    return json.dumps(
        {
            "simulation_state": "running",
            "playing": True,
            "paused": False,
            "stopped": False,
            "scenario_applied": True,
            "sim_time_seconds": 12.5,
            "time_scale": 1.0,
            "scenario_id": "corridor_1",
            "scenario_name": "corridor",
            "environment": "indoor",
            "map_name": "corridor_map",
            "map_width": 20,
            "map_height": 15,
            "map_resolution": 0.05,
            "map_origin_x": -1.0,
            "map_origin_y": 2.5,
            "human_count": 2,
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
                },
                {
                    "id": 4,
                    "x": -1.5,
                    "y": 0.0,
                    "z": 0.5,
                    "vx": 0.0,
                    "vy": 0.0,
                    "vz": -0.2,
                    "speed": 0.2,
                    "goal": None,
                    "group": None,
                    "controller": "external",
                    "end_behavior": "resume",
                },
            ],
            "robot": {"x": 0.5, "y": 0.0, "z": -0.25, "yaw": 0.1},
            "robot_has_goal": True,
            "robot_goal": {"x": 9.0, "y": 0.0, "z": 1.0},
            "camera_focus": "robot",
            "camera_focus_is_robot": True,
            "camera_focus_is_human": False,
            "camera_focus_agent_id": None,
            "camera_tool": "move",
            "camera_mode": "orbit",
            "camera_pose": {"x": -4.0, "y": 1.5, "z": 0.0, "yaw": 0.0},
            "refreshed_at": 41.25,
        }
    )


def _state_string():
    return T["std_msgs/msg/String"](data=_simulation_state_json())


def test_build_state_from_messages():
    messages = {
        "odom": _odometry(),
        "scan": _laserscan(),
        "simulation/agents": _agents_string(),
        "map": T["nav_msgs/msg/OccupancyGrid"](
            header=_header("map"),
            info=T["nav_msgs/msg/MapMetaData"](
                map_load_time=T["builtin_interfaces/msg/Time"](sec=0, nanosec=0),
                resolution=0.05,
                width=1,
                height=1,
                origin=T["geometry_msgs/msg/Pose"](
                    position=T["geometry_msgs/msg/Point"](x=0.0, y=0.0, z=0.0),
                    orientation=T["geometry_msgs/msg/Quaternion"](0.0, 0.0, 0.0, 1.0),
                ),
            ),
            data=np.array([0], dtype=np.int8),
        ),
        "simulation/state": _state_string(),
    }
    snapshot = state.build_state(messages)

    assert snapshot.robot is not None
    assert snapshot.robot.x == pytest.approx(4.0)
    assert snapshot.robot.y == pytest.approx(-1.0)
    assert snapshot.robot.stamp == pytest.approx(7.5)
    assert snapshot.map_received is True
    assert len(snapshot.agents) == 2
    assert snapshot.agents[0].agent_id == "human_0"
    assert snapshot.agents[0].visible is True
    assert snapshot.agents[1].visible is False
    assert snapshot.agents[1].x == pytest.approx(-3.0)
    assert snapshot.agents_received_at is not None

    # The humans of /simulation/state are the world-frame crowd.
    assert len(snapshot.humans) == 2
    first, second = snapshot.humans
    assert first.human_id == 3
    assert first.x == pytest.approx(1.0)
    assert first.z == pytest.approx(2.0)
    assert first.speed == pytest.approx(0.1)
    assert first.goal == state.PointState(x=5.0, y=0.0, z=2.0)
    assert first.group == "G1"
    assert first.controller == "sfm"
    assert first.end_behavior == "stop"
    assert second.human_id == 4
    assert second.goal is None
    assert second.controller == "external"

    simulation = snapshot.simulation
    assert simulation is not None
    assert simulation.simulation_state == "running"
    assert simulation.playing is True
    assert simulation.paused is False
    assert simulation.stopped is False
    assert simulation.scenario_applied is True
    assert simulation.sim_time_seconds == pytest.approx(12.5)
    assert simulation.time_scale == pytest.approx(1.0)
    assert simulation.scenario_id == "corridor_1"
    assert simulation.scenario_name == "corridor"
    assert simulation.environment == "indoor"
    assert simulation.map_name == "corridor_map"
    assert simulation.map_width == 20
    assert simulation.map_height == 15
    assert simulation.map_resolution == pytest.approx(0.05)
    assert simulation.map_origin_x == pytest.approx(-1.0)
    assert simulation.map_origin_y == pytest.approx(2.5)
    assert simulation.human_count == 2
    assert simulation.robot_has_goal is True
    assert simulation.robot_goal == state.PointState(x=9.0, y=0.0, z=1.0)
    assert simulation.camera_focus == "robot"
    assert simulation.camera_focus_is_robot is True
    assert simulation.camera_focus_is_human is False
    assert simulation.camera_focus_agent_id is None
    assert simulation.camera_tool == "move"
    assert simulation.camera_mode == "orbit"
    assert simulation.camera_pose == state.PoseState(x=-4.0, y=1.5, z=0.0, yaw=0.0)
    assert simulation.refreshed_at == pytest.approx(41.25)


def test_yaw_identity_is_zero():
    snapshot = state.build_state({"odom": _odometry(yaw=0.0)})
    assert snapshot.robot.yaw == pytest.approx(0.0, abs=1e-9)


def test_yaw_ninety_degrees_about_z():
    snapshot = state.build_state({"odom": _odometry(yaw=math.pi / 2.0)})
    assert snapshot.robot.yaw == pytest.approx(math.pi / 2.0, abs=1e-9)


def test_agents_accept_raw_string():
    """``build_state`` also takes the raw body, not only the decoded String."""
    raw = _agents_string().data
    snapshot = state.build_state({"simulation/agents": raw})
    assert [agent.agent_id for agent in snapshot.agents] == ["human_0", "human_1"]


def test_agents_accept_a_prefixed_topic():
    """A prefixed session (``/robot0/...``) resolves to the same topics."""
    snapshot = state.build_state(
        {"/robot0/simulation/agents": _agents_string(), "/robot0/scan": _laserscan()}
    )
    assert [agent.agent_id for agent in snapshot.agents] == ["human_0", "human_1"]
    assert snapshot.laser is not None


def test_a_fleet_resolves_the_bare_name_to_the_primary_robot():
    """``odom`` is the primary robot's stream, so it wins over the namespaced ones."""
    snapshot = state.build_state(
        {
            "odom": _odometry(yaw=0.0),
            "robot_1/odom": _odometry(yaw=1.0),
            "robot_2/odom": _odometry(yaw=2.0),
        }
    )
    assert snapshot.robot.yaw == pytest.approx(0.0, abs=1e-9)
    assert snapshot.laser is None


def test_a_renamed_fleet_resolves_the_primary_id_before_the_others():
    """Without the bare name, robot_1 wins over robot_2 whatever the arrival order."""
    snapshot = state.build_state(
        {"robot_2/odom": _odometry(yaw=2.0), "robot_1/odom": _odometry(yaw=1.0)}
    )
    assert snapshot.robot.yaw == pytest.approx(1.0, abs=1e-9)


def test_a_fleet_without_the_primary_id_is_resolved_by_sorted_name():
    """No bare name and no robot_1: the winner is the sorted name, not the first to arrive."""
    ascending = {"robot_3/odom": _odometry(yaw=3.0), "robot_2/odom": _odometry(yaw=2.0)}
    descending = {"robot_2/odom": _odometry(yaw=2.0), "robot_3/odom": _odometry(yaw=3.0)}
    assert state.build_state(ascending).robot.yaw == pytest.approx(2.0, abs=1e-9)
    assert state.build_state(descending).robot.yaw == pytest.approx(2.0, abs=1e-9)


def test_simulation_state_body_is_tolerated_when_broken():
    """A body that cannot be read empties its part instead of raising."""
    for body in ("not json at all", "[]", '{"humans": "not a list"}'):
        snapshot = state.build_state({"simulation/state": body})
        assert snapshot.humans == ()

    malformed = state.build_state({"simulation/state": T["std_msgs/msg/String"](data="{oops")})
    assert malformed.simulation is None

    # An unreadable agents body publishes no agents rather than an exception.
    assert state.build_state({"simulation/agents": '{"agents": [7]}'}).agents == ()


def test_simulation_state_partial_body_keeps_the_defaults():
    """Every key is optional: a state body alone still reads."""
    snapshot = state.build_state({"simulation/state": '{"simulation_state": "idle"}'})
    simulation = snapshot.simulation
    assert simulation is not None
    assert simulation.simulation_state == "idle"
    assert simulation.playing is False
    assert simulation.sim_time_seconds is None
    assert simulation.time_scale is None
    assert simulation.scenario_id is None
    assert simulation.map_width == 0
    assert simulation.robot_goal is None
    assert simulation.camera_pose is None
    assert simulation.refreshed_at is None
    assert snapshot.humans == ()


def test_laser_points_2d_skips_out_of_range():
    snapshot = state.build_state({"scan": _laserscan()})
    points = snapshot.laser.points_2d()
    assert len(points) == 2
    assert points[0][0] == pytest.approx(1.0)
    assert points[0][1] == pytest.approx(0.0)
    assert points[1][0] == pytest.approx(-2.0, abs=1e-6)
    assert points[1][1] == pytest.approx(0.0, abs=1e-6)


def test_empty_messages_produce_empty_state():
    snapshot = state.build_state({})
    assert snapshot.robot is None
    assert snapshot.laser is None
    assert snapshot.agents == ()
    assert snapshot.humans == ()
    assert snapshot.simulation is None
    assert snapshot.map_received is False
    assert snapshot.agents_received_at is None
