"""Tests for the composable observation of :class:`RobotSNAPEnv`.

Nothing here touches a session: a :class:`World` is built by hand, which is what
the observation reads, and the parts are checked against it directly. The one
fixture used is the client, because an environment refuses to build without one.
"""

import math

import numpy as np
import pytest
from gymnasium import spaces

from robotsnap.envs import RobotSNAPEnv
from robotsnap.envs.base import MAX_AGENT_SPEED
from robotsnap.envs.observation import (
    DEFAULT_OBSERVATIONS,
    ObservationPart,
    describe_observation_parts,
    normalise_observation_names,
    register_observation_part,
    registered_observation_names,
)
from robotsnap.envs.world import Agent, Human, LidarScan, OccupancyMap, Pose2D, World

#: The size the historical flat observation has with the default sizes.
_DEFAULT_SIZE = 6 + 5 + 8 * 5 + 48


@pytest.fixture
def make_env(client):
    """Build environments over the fixture client, and close them afterwards."""
    created = []

    def build(**kwargs):
        environment = RobotSNAPEnv(client=client, scenario=None, **kwargs)
        created.append(environment)
        return environment

    yield build
    for environment in created:
        environment.close()


def _scan(ranges=(3.0, 3.0, 3.0, 3.0), range_max=5.0):
    values = np.asarray(ranges, dtype=np.float32)
    angles = np.linspace(-math.pi, math.pi, values.size, endpoint=False).astype(np.float32)
    return LidarScan(
        ranges=values, angles=angles, range_min=0.05, range_max=range_max, stamp=0.0
    )


def _grid(cells=None):
    if cells is None:
        cells = np.zeros((12, 12), dtype=np.int8)
    return OccupancyMap(
        cells=np.asarray(cells, dtype=np.int8),
        resolution=1.0,
        origin_x=0.0,
        origin_y=0.0,
    )


def _world(**overrides):
    """One world, with everything a part can read filled in."""
    fields = dict(
        robot_id="robot_1",
        pose=Pose2D(1.0, 2.0, 0.0),
        linear_velocity=0.5,
        angular_velocity=-0.25,
        goal=Pose2D(4.0, 2.0, 0.0),
        scan=_scan(),
        agents=(
            Agent("a1", 1.0, 0.0, 0.2, 0.0, True),
            Agent("a2", 5.0, 0.0, 0.0, 0.0, False),
        ),
        humans=(Human(0, 3.0, 2.0, 0.0, 0.0, 0.0),),
        map=_grid(),
        sim_time_seconds=12.0,
        time_scale=1.0,
        scenario_id="demo",
        scenario_applied=True,
    )
    fields.update(overrides)
    return World(**fields)


def _observe(environment, world=None):
    world = _world() if world is None else world
    return environment.observation(world, environment.task(world))


# -- the default, and what it is made of -----------------------------------


def test_the_default_observation_is_still_the_historical_vector(make_env):
    environment = make_env()
    world = _world()

    value = _observe(environment, world)

    assert environment.observation_names == DEFAULT_OBSERVATIONS
    assert value.shape == (_DEFAULT_SIZE,)
    assert value.dtype == np.float32
    assert environment.observation_space.contains(value)
    expected = np.concatenate(
        [
            environment.robot_features(world),
            environment.goal_features(world, environment.task(world)),
            environment.agent_features(world),
            environment.lidar_features(world),
        ]
    )
    assert np.array_equal(value, expected)


def test_the_slices_locate_every_part_of_the_default_vector(make_env):
    environment = make_env()

    assert environment.observation_slices == {
        "robot": slice(0, 6),
        "goal": slice(6, 11),
        "agents": slice(11, 51),
        "lidar": slice(51, 99),
    }


# -- choosing the parts ----------------------------------------------------


@pytest.mark.parametrize(
    "observations, names, size",
    [
        ("pose,goal", ("pose", "goal"), 9),
        (["goal", "lidar_stats"], ("goal", "lidar_stats"), 8),
        (("robot", "status", "time"), ("robot", "status", "time"), 10),
        ("none", (), 0),
        ([], (), 0),
        ("nearest_agent", ("nearest_agent",), 5),
    ],
)
def test_a_spec_picks_the_parts_and_their_order(make_env, observations, names, size):
    environment = make_env(observations=observations)

    value = _observe(environment)

    assert environment.observation_names == names
    assert value.shape == (size,)
    assert environment.observation_space.shape == (size,)
    assert environment.observation_space.contains(value)


def test_the_lidar_can_be_left_out(make_env):
    environment = make_env(observations=("robot", "goal", "agents"))

    value = _observe(environment)

    assert "lidar" not in environment.observation_names
    assert value.shape == (6 + 5 + 8 * 5,)
    assert environment.observation_space.contains(value)


def test_the_agents_can_be_left_out(make_env):
    environment = make_env(observations=("robot", "goal", "lidar"))

    value = _observe(environment)

    assert "agents" not in environment.observation_names
    assert value.shape == (6 + 5 + 48,)
    assert environment.observation_space.contains(value)


def test_both_can_be_left_out_and_the_rest_still_reads_the_same(make_env):
    """The point of the whole exercise: a part left out shifts nothing else."""
    full = make_env()
    trimmed = make_env(observations=("robot", "goal"))
    world = _world()

    whole = _observe(full, world)
    short = _observe(trimmed, world)

    assert short.shape == (11,)
    assert np.array_equal(short, whole[:11])
    assert trimmed.observation_space.contains(short)


def test_the_humans_can_be_observed_on_their_own(make_env):
    environment = make_env(
        observations="humans", observation_params={"humans": {"max": 3}}
    )

    value = _observe(environment)

    assert environment.observation_names == ("humans",)
    assert value.shape == (15,)
    assert value[4] == pytest.approx(1.0)  # the one human of the world, present
    assert environment.observation_space.contains(value)


def test_a_scalar_part_reads_the_world(make_env):
    environment = make_env(observations="goal_distance")

    value = _observe(environment)

    assert value.shape == (1,)
    assert value[0] == pytest.approx(3.0)  # (1, 2) to (4, 2)


def test_a_missing_goal_reads_as_no_goal(make_env):
    environment = make_env(observations="goal_distance")

    value = _observe(environment, _world(goal=None))

    assert value[0] == pytest.approx(-1.0)
    assert environment.observation_space.contains(value)


def test_the_occupancy_patch_is_cut_around_the_robot(make_env):
    cells = np.zeros((12, 12), dtype=np.int8)
    cells[5, 7] = 100  # one obstacle, one cell to the robot's right
    environment = make_env(
        observations="occupancy", observation_params={"occupancy": {"size": 5}}
    )

    patch = _observe(environment, _world(pose=Pose2D(5.5, 5.5, 0.0), map=_grid(cells)))

    assert patch.shape == (25,)
    grid = patch.reshape(5, 5)
    assert grid[2, 4] == pytest.approx(1.0)  # the obstacle, right of the centre
    assert grid[2, 2] == pytest.approx(0.0)  # the robot's own cell


def test_a_patch_that_leaves_the_grid_counts_the_outside_as_blocked(make_env):
    environment = make_env(
        observations="occupancy", observation_params={"occupancy": {"size": 5}}
    )

    patch = _observe(environment, _world(pose=Pose2D(0.5, 0.5, 0.0)))

    grid = patch.reshape(5, 5)
    assert grid[0, 0] == pytest.approx(1.0)
    assert grid[0, 1] == pytest.approx(1.0)
    assert grid[2, 2] == pytest.approx(0.0)


def test_the_humans_part_is_expressed_in_the_robot_frame(make_env):
    """A human two metres ahead of a robot facing +y reads as two metres left."""
    environment = make_env(
        observations="humans", observation_params={"humans": {"max": 1}}
    )

    value = _observe(
        environment,
        _world(
            pose=Pose2D(1.0, 2.0, math.pi / 2),
            humans=(Human(0, 3.0, 2.0, 0.0, 0.0, 0.0),),
        ),
    )

    assert value[0] == pytest.approx(0.0, abs=1e-6)
    assert value[1] == pytest.approx(-2.0, abs=1e-6)  # to the right, in the robot frame
    assert value[4] == pytest.approx(1.0)


def test_the_status_part_reports_the_episode_flags(make_env):
    environment = make_env(observations="status")

    value = _observe(environment)

    assert value.tolist() == [0.0, 0.0, 0.0]
    goal = _world(pose=Pose2D(4.0, 2.0, 0.0))  # standing on the goal
    assert _observe(environment, goal).tolist() == [1.0, 0.0, 0.0]


# -- sizes and parameters --------------------------------------------------


def test_parameters_size_the_parts_that_use_them(make_env):
    environment = make_env(
        observation_params={"lidar": {"bins": 16}, "agents": {"max": 2}}
    )

    value = _observe(environment)

    assert environment.lidar_bins == 16
    assert environment.max_agents == 2
    assert value.shape == (6 + 5 + 2 * 5 + 16,)
    assert environment.observation_space.contains(value)


def test_a_spec_carries_its_own_parameters(make_env):
    environment = make_env(observations={"agents": {"max": 0}, "lidar": {"bins": 8}})

    value = _observe(environment)

    assert environment.observation_names == ("agents", "lidar")
    assert value.shape == (8,)
    assert environment.observation_space.contains(value)


def test_a_parameter_the_spec_gives_wins_over_the_constructor(make_env):
    environment = make_env(lidar_bins=48, observations={"lidar": {"bins": 12}})

    assert environment.lidar_bins == 12
    assert _observe(environment).shape == (12,)


def test_a_zero_sized_part_still_has_a_space(make_env):
    environment = make_env(observations=("pose", "agents"), max_agents=0)

    value = _observe(environment)

    assert value.shape == (4,)
    assert environment.observation_space.contains(value)


# -- the structure ---------------------------------------------------------


def test_a_dict_observation_hands_back_one_array_per_part(make_env):
    environment = make_env(
        observations="pose,goal,lidar_stats", observation_structure="dict"
    )

    value = _observe(environment)

    assert isinstance(environment.observation_space, spaces.Dict)
    assert list(value) == ["pose", "goal", "lidar_stats"]
    assert value["pose"].shape == (4,)
    assert value["goal"].shape == (5,)
    assert value["lidar_stats"].shape == (3,)
    assert environment.observation_space.contains(value)


def test_the_structure_has_to_be_known(make_env):
    with pytest.raises(ValueError, match="observation_structure"):
        make_env(observation_structure="grid")


# -- subclassing and extension ---------------------------------------------


def test_a_subclass_feature_still_feeds_its_part(client):
    class Blind(RobotSNAPEnv):
        def lidar_features(self, world):
            return np.zeros(self.lidar_bins, dtype=np.float32)

    environment = Blind(client=client, scenario=None, observations="lidar")
    world = _world()
    try:
        value = environment.observation(world, environment.task(world))
    finally:
        environment.close()

    assert value.shape == (48,)
    assert not value.any()


def test_a_project_can_register_its_own_part(make_env):
    register_observation_part(
        "test_battery",
        ObservationPart(
            name="test_battery",
            size=1,
            low=np.zeros(1, dtype=np.float32),
            high=np.ones(1, dtype=np.float32),
            build=lambda context: np.array(
                [min(context.world.linear_velocity / 10.0, 1.0)], dtype=np.float32
            ),
            doc="a stand-in for a state a project adds",
        ),
    )

    environment = make_env(observations=("pose", "test_battery"))

    value = _observe(environment)

    assert "test_battery" in registered_observation_names()
    assert value.shape == (5,)
    assert value[4] == pytest.approx(0.05)
    assert environment.observation_space.contains(value)


def test_a_part_can_be_handed_in_without_being_registered(make_env):
    part = ObservationPart(
        name="inline",
        size=2,
        low=np.array([-1.0, -1.0], dtype=np.float32),
        high=np.array([1.0, 1.0], dtype=np.float32),
        build=lambda context: np.array([1.0, -1.0], dtype=np.float32),
        doc="built here, used here",
    )
    environment = make_env(observations=("pose", part))

    value = _observe(environment)

    assert environment.observation_names == ("pose", "inline")
    assert value.shape == (6,)
    assert value[4:].tolist() == [1.0, -1.0]


def test_a_subclass_can_declare_its_own_parts_by_name(client):
    """A part a task writes for itself is asked for by name, and stays local."""
    from robotsnap.envs.observation import ObservationPart

    class WithAFeeling(RobotSNAPEnv):
        DEFAULT_PARTS = ("pose", "mood")

        def __init__(self, **kwargs):
            self.extra_observation_parts = (
                ObservationPart(
                    name="mood",
                    size=1,
                    low=np.zeros(1, dtype=np.float32),
                    high=np.ones(1, dtype=np.float32),
                    build=lambda context: np.array([0.25], dtype=np.float32),
                    doc="a part this task owns",
                ),
            )
            kwargs.setdefault("observations", self.DEFAULT_PARTS)
            super().__init__(**kwargs)

    environment = WithAFeeling(client=client, scenario=None)
    try:
        value = _observe(environment)
        asked = _observe(WithAFeeling(client=client, scenario=None, observations={"mood": {}}))
    finally:
        environment.close()

    assert environment.observation_names == ("pose", "mood")
    assert value.shape == (5,)
    assert value[4] == pytest.approx(0.25)
    assert environment.observation_space.contains(value)
    # Reachable by name even when the spec names it explicitly, and the global
    # registry still knows nothing about it.
    assert asked.shape == (1,)
    assert "mood" not in registered_observation_names()


def test_a_part_that_returns_the_wrong_size_is_caught(make_env):
    part = ObservationPart(
        name="wrong",
        size=2,
        low=np.zeros(2, dtype=np.float32),
        high=np.ones(2, dtype=np.float32),
        build=lambda context: np.zeros(3, dtype=np.float32),
    )
    environment = make_env(observations=part)

    with pytest.raises(ValueError, match="expected 2"):
        _observe(environment)


def test_registering_over_a_part_needs_to_be_asked_for():
    part = ObservationPart(
        name="goal",
        size=1,
        low=np.zeros(1, dtype=np.float32),
        high=np.ones(1, dtype=np.float32),
        build=lambda context: np.zeros(1, dtype=np.float32),
    )
    with pytest.raises(ValueError, match="already registered"):
        register_observation_part("goal", part)


# -- what a caller gets wrong ----------------------------------------------


def test_an_unknown_part_is_refused_by_name(make_env):
    with pytest.raises(ValueError, match="unknown observation part 'lidarr'"):
        make_env(observations="lidarr")


def test_parameters_for_a_part_that_was_not_asked_for_are_refused(make_env):
    with pytest.raises(ValueError, match="not asked for"):
        make_env(observations="pose", observation_params={"lidar": {"bins": 8}})


def test_a_part_asked_for_twice_is_refused(make_env):
    with pytest.raises(ValueError, match="twice"):
        make_env(observations="pose,pose")


def test_an_empty_spec_is_not_an_empty_name_and_a_bad_one_says_so():
    assert normalise_observation_names("none") == ()
    assert normalise_observation_names("  ") == ()
    with pytest.raises(ValueError, match="unknown observation part"):
        normalise_observation_names("lidar,sonar")


def test_the_bounds_follow_the_parts(make_env):
    environment = make_env(observations="lidar_stats")

    low, high = environment.observation_bounds()

    assert low.tolist() == [0.0, 0.0, 0.0]
    assert high.tolist() == [1.0, 1.0, 1.0]


def test_the_package_describes_its_own_parts():
    described = dict(describe_observation_parts())

    assert "lidar" in described and "humans" in described
    assert all(name and doc for name, doc in described.items())
    # The historical speed bound is still published where it always was.
    assert MAX_AGENT_SPEED == 5.0
