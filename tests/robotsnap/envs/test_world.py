"""Tests for ``robotsnap.envs.world``: the typed read an environment works on."""

import math

import numpy as np
import pytest

from robotsnap.envs.world import (
    LidarScan,
    OccupancyMap,
    Pose2D,
    World,
    read_world,
    wrap_angle,
)
from unity_peer import unity_laserscan, unity_occupancy_grid, unity_odometry


class StubClient:
    """A client-shaped object holding canned streams, with no socket at all."""

    def __init__(self, state=None, odom=None, scan=None, grid=None, agents=None):
        self._state = state
        self._messages = {"odom": odom, "scan": scan, "map": grid}
        self._agents = agents

    def snapshot(self):
        return self._state

    def robot(self, robot_id=None):
        robots = (self._state or {}).get("robots") or []
        return robots[0] if robots else None

    def last_message(self, topic):
        return self._messages.get(str(topic).strip("/"))

    def agents(self, robot=None):
        return self._agents


def _state(x=1.0, y=2.0, yaw=0.25, goal=(9.0, 2.0)):
    entry = {
        "id": "robot_1",
        "type": "jackal",
        "is_primary": True,
        "x": x,
        "y": y,
        "z": 0.0,
        "yaw": yaw,
        "has_goal": goal is not None,
        "goal": None if goal is None else {"x": goal[0], "y": goal[1], "z": 0.0},
    }
    return {
        "scenario_id": "demo",
        "scenario_applied": True,
        "sim_time_seconds": 12.5,
        "time_scale": 1.0,
        "robots": [entry],
        "human_count": 0,
        "humans": [],
    }


# -- geometry ---------------------------------------------------------------=


def test_wrap_angle_folds_into_one_turn():
    assert wrap_angle(0.0) == pytest.approx(0.0)
    # Half a turn folds onto -pi: the range is half open, so no angle has two
    # spellings and an equality test on a heading is meaningful.
    assert wrap_angle(3 * math.pi) == pytest.approx(-math.pi)
    assert wrap_angle(-3 * math.pi) == pytest.approx(-math.pi)
    assert wrap_angle(2 * math.pi + 0.5) == pytest.approx(0.5)


def test_a_pose_measures_distance_and_bearing():
    pose = Pose2D(0.0, 0.0, 0.0)

    assert pose.distance_to(3.0, 4.0) == pytest.approx(5.0)
    assert pose.bearing_to(1.0, 1.0) == pytest.approx(math.pi / 4)
    # A heading of pi/2 puts a point straight ahead of the robot on the +y axis.
    turned = Pose2D(0.0, 0.0, math.pi / 2)
    assert turned.bearing_to(0.0, 1.0) == pytest.approx(0.0)
    assert turned.bearing_to(1.0, 1.0) == pytest.approx(-math.pi / 4)


# -- the lidar --------------------------------------------------------------=


def _scan(ranges, range_max=10.0):
    return LidarScan(
        ranges=np.asarray(ranges, dtype=np.float32),
        angles=np.linspace(-math.pi, math.pi, len(ranges), endpoint=False).astype(
            np.float32
        ),
        range_min=0.1,
        range_max=range_max,
        stamp=1.0,
    )


def test_the_shortest_hit_ignores_beams_that_saw_nothing():
    scan = _scan([np.inf, 3.0, np.nan, 1.5, np.inf])

    assert scan.min_range() == pytest.approx(1.5)
    assert _scan([np.inf, np.nan]).min_range() == pytest.approx(10.0)


def test_resampling_keeps_the_nearest_hit_of_each_slice():
    scan = _scan([np.nan, 1.0, np.inf, 2.0, 3.0, 4.0, np.inf, np.inf], range_max=5.0)

    assert scan.resample(4) == pytest.approx([1.0, 2.0, 3.0, 5.0])
    assert scan.resample(8) == pytest.approx([5.0, 1.0, 5.0, 2.0, 3.0, 4.0, 5.0, 5.0])
    # A slice with no hit reads as the sensor's own reach, not as a zero range.
    assert scan.resample(1) == pytest.approx([1.0])


def test_resampling_refuses_a_number_of_bins_that_is_not_one():
    with pytest.raises(ValueError):
        _scan([1.0, 2.0]).resample(0)


def test_the_points_are_the_beams_that_returned_inside_the_sensor_range():
    scan = LidarScan(
        ranges=np.array([np.inf, 2.0, 0.01], dtype=np.float32),
        angles=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        range_min=0.1,
        range_max=10.0,
    )

    points = scan.points_2d()

    assert points.shape == (1, 2)
    assert points[0] == pytest.approx([2.0, 0.0])


def test_a_scan_with_nothing_in_range_has_no_points():
    assert _scan([np.inf, np.inf]).points_2d().shape == (0, 2)


def test_the_scan_read_from_a_message_keeps_its_beam_bearings():
    message = unity_laserscan(
        [1.0, 2.0, 3.0], angle_min=-0.5, angle_increment=0.5, range_max=8.0
    )
    scan = LidarScan.from_state(
        _state_scan(message)
    )

    assert scan.beams == 3
    assert scan.angles == pytest.approx([-0.5, 0.0, 0.5])
    assert scan.range_max == pytest.approx(8.0)


def _state_scan(message):
    """The ``LaserScanState`` the bridge builds from a decoded scan."""
    from robotsnap.bridge.state import LaserScanState

    return LaserScanState(
        angle_min=message.angle_min,
        angle_max=message.angle_max,
        angle_increment=message.angle_increment,
        range_min=message.range_min,
        range_max=message.range_max,
        ranges=tuple(message.ranges),
        frame_id=message.header.frame_id,
        stamp=1.0,
    )


# -- the grid ---------------------------------------------------------------=


def _grid(cells, resolution=0.5, origin_x=0.0, origin_y=0.0):
    return OccupancyMap.from_message(
        unity_occupancy_grid(cells, resolution=resolution, origin_x=origin_x, origin_y=origin_y)
    )


def test_a_column_counts_along_x_and_a_row_along_y():
    """The order the publisher writes, which is the one easy to get wrong."""
    message = unity_occupancy_grid([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 100]], resolution=1.0)
    grid = OccupancyMap.from_message(message)

    assert (grid.height, grid.width) == (3, 4)
    # The wall is written at row 2, column 3: x = 3.5, y = 2.5.
    assert grid.cell(3.5, 2.5) == (3, 2)
    assert grid.occupancy(3.5, 2.5) == 100
    assert grid.is_occupied(3.5, 2.5) is True
    assert grid.is_occupied(2.5, 2.5) is False


def test_the_grid_origin_is_where_its_counting_starts():
    grid = _grid([[0, 100]], resolution=2.0, origin_x=-5.0, origin_y=-3.0)

    assert grid.bounds() == pytest.approx((-5.0, -3.0, -1.0, -1.0))
    assert grid.cell(-4.0, -2.0) == (0, 0)
    assert grid.is_occupied(-3.0, -2.0) is True
    assert grid.cell(-9.0, 0.0) is None
    assert grid.occupancy(-9.0, 0.0) is None
    assert grid.is_occupied(-9.0, 0.0) is False
    assert grid.contains(-5.5, -3.5) is False


def test_a_grid_that_does_not_match_its_own_size_is_refused():
    message = unity_occupancy_grid([[0, 0], [0, 0]])
    message.info.width = 3

    with pytest.raises(ValueError):
        OccupancyMap.from_message(message)


# -- reading a session ------------------------------------------------------=


def test_read_world_assembles_every_stream():
    client = StubClient(
        state=_state(1.0, 2.0, 0.25, goal=(9.0, 2.0)),
        odom=unity_odometry(x=1.0, y=2.0, yaw=0.25, linear_x=0.4, angular_z=-0.1),
        scan=unity_laserscan([1.0] * 4, angle_increment=0.25, range_max=6.0),
        grid=unity_occupancy_grid([[0, 0], [0, 0]], resolution=1.0),
        agents={"agents": [{"id": "h1", "x": 2.0, "y": 0.0, "vx": 0.0, "vy": 0.0, "visible": True}]},
    )

    world = read_world(client)

    assert world.robot_id == "robot_1"
    assert (world.pose.x, world.pose.y, world.pose.yaw) == pytest.approx([1.0, 2.0, 0.25])
    assert (world.goal.x, world.goal.y) == pytest.approx([9.0, 2.0])
    assert world.linear_velocity == pytest.approx(0.4)
    assert world.angular_velocity == pytest.approx(-0.1)
    assert world.scan is not None and world.scan.beams == 4
    assert world.map is not None and world.map.width == 2
    assert [agent.agent_id for agent in world.agents] == ["h1"]
    assert world.distance_to_goal == pytest.approx(8.0)
    # The goal is straight along +x and the robot is turned a quarter of a
    # radian to the left of it, so it sits to the right of the heading.
    assert world.bearing_to_goal == pytest.approx(-0.25)
    assert world.sim_time_seconds == pytest.approx(12.5)
    assert world.time_scale == pytest.approx(1.0)
    assert world.scenario_id == "demo"
    assert world.scenario_applied is True


def test_read_world_of_a_session_that_has_published_nothing_is_empty():
    world = read_world(StubClient())

    assert world.pose is None and world.goal is None and world.scan is None
    assert world.map is None and world.agents == () and world.humans == ()
    assert world.distance_to_goal is None and world.bearing_to_goal is None
    assert world.scenario_applied is False


def test_the_roster_pose_is_preferred_over_a_stale_odometry_frame():
    """The snapshot is rebuilt after a reset; the odometry may still be the old one."""
    client = StubClient(
        state=_state(5.0, 5.0, 0.0, goal=(0.0, 0.0)),
        odom=unity_odometry(x=-20.0, y=-20.0, yaw=0.0, linear_x=0.0, angular_z=0.0),
    )

    world = read_world(client)

    assert (world.pose.x, world.pose.y) == pytest.approx([5.0, 5.0])
    # The velocity still comes from odometry: the roster carries no twist.
    assert world.linear_velocity == pytest.approx(0.0)


def test_a_robot_without_a_goal_reports_none():
    world = read_world(StubClient(state=_state(goal=None)))

    assert world.goal is None
    assert world.distance_to_goal is None
    assert world.bearing_to_goal is None


def test_the_scenario_goal_is_read_from_the_authored_target():
    """A scenario declares a destination long before the robot sets off.

    ``goal`` only appears once the simulator has a live one - a command set it,
    or the planner did - so an episode that starts on a fresh scenario reads the
    destination from ``target_pose``, the point the file authored.
    """
    entry = _state(2.0, 3.0, 0.0, goal=None)["robots"][0]
    entry["goal"] = None
    entry["has_goal"] = False
    entry["target_pose"] = {"x": 8.0, "y": 3.0, "z": 0.0, "yaw": 0.0}
    state = _state(2.0, 3.0, 0.0, goal=None)
    state["robots"] = [entry]

    world = read_world(StubClient(state=state))

    assert world.goal is not None
    assert (world.goal.x, world.goal.y) == pytest.approx([8.0, 3.0])
    assert world.distance_to_goal == pytest.approx(6.0)


def test_a_live_goal_wins_over_the_authored_one():
    """Once the robot is driving somewhere, that is the goal of the episode."""
    entry = _state(2.0, 3.0, 0.0, goal=None)["robots"][0]
    entry["goal"] = {"x": 1.0, "y": 3.0, "z": 0.0}
    entry["has_goal"] = True
    entry["target_pose"] = {"x": 8.0, "y": 3.0, "z": 0.0, "yaw": 0.0}
    state = _state(2.0, 3.0, 0.0, goal=None)
    state["robots"] = [entry]

    world = read_world(StubClient(state=state))

    assert world.goal is not None
    assert (world.goal.x, world.goal.y) == pytest.approx([1.0, 3.0])
    assert world.distance_to_goal == pytest.approx(1.0)


def test_the_nearest_agent_is_the_closest_one_in_the_robot_frame():
    world = World(
        agents=(
            _agent("far", 4.0, 0.0),
            _agent("near", 0.5, 0.5),
        )
    )

    assert world.nearest_agent().agent_id == "near"
    assert world.nearest_agent().distance == pytest.approx(math.hypot(0.5, 0.5))


def _agent(agent_id, x, y):
    from robotsnap.envs.world import Agent

    return Agent(agent_id=agent_id, x=x, y=y, vx=0.0, vy=0.0, visible=True)
