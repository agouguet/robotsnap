"""Tests for the CDR codecs."""

import json

import numpy as np
import pytest

from robotsnap.bridge import codec

T = codec.TYPESTORE.types

HDR = "std_msgs/msg/Header"
TIME = "builtin_interfaces/msg/Time"
POINT = "geometry_msgs/msg/Point"
QUAT = "geometry_msgs/msg/Quaternion"
POSE = "geometry_msgs/msg/Pose"
VEC3 = "geometry_msgs/msg/Vector3"
TWIST = "geometry_msgs/msg/Twist"


def _header(frame_id="map", sec=4, nanosec=250):
    return T[HDR](stamp=T[TIME](sec=sec, nanosec=nanosec), frame_id=frame_id)


def _point(x=1.0, y=2.0, z=3.0):
    return T[POINT](x=x, y=y, z=z)


def _odometry():
    pose = T[POSE](
        position=_point(1.5, -2.5, 0.0),
        orientation=T[QUAT](x=0.0, y=0.0, z=0.0, w=1.0),
    )
    twist = T[TWIST](
        linear=T[VEC3](x=0.6, y=0.0, z=0.0),
        angular=T[VEC3](x=0.0, y=0.0, z=-0.25),
    )
    return T["nav_msgs/msg/Odometry"](
        header=_header("odom"),
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
        header=_header("laser"),
        angle_min=-1.0,
        angle_max=1.0,
        angle_increment=0.5,
        time_increment=0.0,
        scan_time=0.1,
        range_min=0.2,
        range_max=5.0,
        ranges=np.array([1.0, 2.0, np.inf], dtype=np.float32),
        intensities=np.array([0.0, 0.0, 0.0], dtype=np.float32),
    )


def _occupancy_grid():
    info = T["nav_msgs/msg/MapMetaData"](
        map_load_time=T[TIME](sec=1, nanosec=0),
        resolution=0.05,
        width=2,
        height=2,
        origin=T[POSE](position=_point(0.0, 0.0, 0.0), orientation=T[QUAT](0.0, 0.0, 0.0, 1.0)),
    )
    return T["nav_msgs/msg/OccupancyGrid"](
        header=_header("map"),
        info=info,
        data=np.array([0, 100, -1, 0], dtype=np.int8),
    )


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
                    "vy": -0.2,
                    "vz": 0.0,
                    "visible": True,
                },
                {
                    "id": "human_1",
                    "x": -3.0,
                    "y": 0.5,
                    "z": 0.0,
                    "vx": 0.0,
                    "vy": 0.1,
                    "vz": 0.0,
                    "visible": False,
                },
            ],
            "frame": "robot",
        }
    )


def test_encode_decode_odometry():
    raw = codec.encode("nav_msgs/msg/Odometry", _odometry())
    msg = codec.decode_odometry(raw)
    assert msg.child_frame_id == "base_link"
    assert msg.pose.pose.position.x == pytest.approx(1.5)
    assert msg.pose.pose.position.y == pytest.approx(-2.5)
    assert msg.twist.twist.linear.x == pytest.approx(0.6)
    assert msg.twist.twist.angular.z == pytest.approx(-0.25)
    assert msg.header.frame_id == "odom"
    assert msg.header.stamp.sec == 4
    assert msg.header.stamp.nanosec == 250


def test_encode_decode_laserscan():
    raw = codec.encode("sensor_msgs/msg/LaserScan", _laserscan())
    msg = codec.decode_laserscan(raw)
    assert msg.angle_min == pytest.approx(-1.0)
    assert msg.angle_increment == pytest.approx(0.5)
    assert msg.range_max == pytest.approx(5.0)
    assert msg.header.frame_id == "laser"
    np.testing.assert_allclose(msg.ranges[:2], np.array([1.0, 2.0], dtype=np.float32))


def test_encode_decode_occupancy_grid():
    raw = codec.encode("nav_msgs/msg/OccupancyGrid", _occupancy_grid())
    msg = codec.decode_occupancy_grid(raw)
    assert msg.info.width == 2
    assert msg.info.height == 2
    assert msg.info.resolution == pytest.approx(0.05)
    assert list(msg.data) == [0, 100, -1, 0]


def test_encode_decode_bool():
    raw = codec.encode("std_msgs/msg/Bool", T["std_msgs/msg/Bool"](data=True))
    msg = codec.decode_bool(raw)
    assert msg.data is True


def test_encode_decode_string():
    raw = codec.encode("std_msgs/msg/String", T["std_msgs/msg/String"](data="hello"))
    msg = codec.decode_string(raw)
    assert msg.data == "hello"


def test_encode_twist_roundtrip():
    raw = codec.encode_twist(0.7, -0.3)
    twist = codec.decode("geometry_msgs/msg/Twist", raw)
    assert twist.linear.x == pytest.approx(0.7)
    assert twist.angular.z == pytest.approx(-0.3)
    assert twist.linear.y == pytest.approx(0.0)
    assert twist.angular.x == pytest.approx(0.0)


def test_decode_agents_json_valid():
    text = _agents_json()
    assert codec.decode_agents_json(text) == {
        "agents": [
            {
                "id": "human_0",
                "x": 1.0,
                "y": 2.0,
                "z": 0.0,
                "vx": 0.1,
                "vy": -0.2,
                "vz": 0.0,
                "visible": True,
            },
            {
                "id": "human_1",
                "x": -3.0,
                "y": 0.5,
                "z": 0.0,
                "vx": 0.0,
                "vy": 0.1,
                "vz": 0.0,
                "visible": False,
            },
        ],
        "frame": "robot",
    }


def test_decode_agents_json_accepts_a_parsed_body():
    """The client reads the body as JSON first and hands the object over."""
    body = json.loads(_agents_json())
    assert codec.decode_agents_json(body)["agents"][0]["id"] == "human_0"


def test_decode_agents_json_defaults_the_frame():
    """The only frame the contract defines is the robot frame."""
    body = {"agents": []}
    assert codec.decode_agents_json(json.dumps(body)) == {"agents": [], "frame": "robot"}


def test_decode_agents_json_malformed():
    with pytest.raises(codec.CodecError):
        codec.decode_agents_json("not json at all")
    with pytest.raises(codec.CodecError):
        codec.decode_agents_json('["human_0"]')  # not an object
    with pytest.raises(codec.CodecError):
        codec.decode_agents_json('{"frame": "robot"}')  # no agents list
    with pytest.raises(codec.CodecError):
        codec.decode_agents_json('{"agents": {}}')  # agents is not a list
    with pytest.raises(codec.CodecError):
        codec.decode_agents_json('{"agents": [7]}')  # entry is not an object
    with pytest.raises(codec.CodecError):
        codec.decode_agents_json('{"agents": [{"id": "a1"}]}')  # missing fields
    with pytest.raises(codec.CodecError):
        codec.decode_agents_json(
            '{"agents": [{"id": "a1", "x": "left", "y": 0, "z": 0,'
            ' "vx": 0, "vy": 0, "vz": 0, "visible": true}]}'
        )


def test_decode_unknown_type_raises_codec_error():
    with pytest.raises(codec.CodecError):
        codec.decode("no/msg/Type", b"\x00")
