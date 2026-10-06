"""Tests for the shared social task and its structured observation.

Every test here builds its world out of the dataclasses of
``robotsnap.envs.world`` and a ``TaskState`` by hand, so none of them needs a
session, a socket or a running Unity. The environment itself is built with a
stand-in client for the same reason: the base class does not touch its client
until an episode starts, which is what lets a caller size a network from the
observation and action spaces long before Unity is in the picture.
"""

import math

import numpy as np
import pytest

from robotsnap.envs.base import TaskState
from robotsnap.envs.world import Agent, Human, Pose2D, World
from robotsnap.models.social import (
    DEFAULT_MAX_NEIGHBOURS,
    EGO_KEY,
    EGO_SIZE,
    GOAL_KEY,
    GOAL_SIZE,
    MASK_KEY,
    NEIGHBOUR_SIZE,
    NEIGHBOURS_KEY,
    SocialNavEnv,
    build_structured_observation,
    pack_observation,
)


def _task(
    distance=5.0,
    bearing=0.0,
    *,
    goal_reached=False,
    collision=False,
    out_of_bounds=False,
    elapsed=0.0,
) -> TaskState:
    return TaskState(
        distance_to_goal=distance,
        bearing_to_goal=bearing,
        goal_reached=goal_reached,
        collision=collision,
        out_of_bounds=out_of_bounds,
        min_lidar=None,
        nearest_agent_distance=None,
        elapsed_seconds=elapsed,
        wall_seconds=elapsed,
        steps=0,
    )


def _world(agents=(), humans=(), pose=None, linear=0.0, angular=0.0) -> World:
    return World(
        pose=Pose2D(0.0, 0.0, 0.0) if pose is None else pose,
        linear_velocity=linear,
        angular_velocity=angular,
        agents=tuple(agents),
        humans=tuple(humans),
    )


def _env(**kwargs) -> SocialNavEnv:
    """An environment with a stand-in client, so no port is ever bound."""
    return SocialNavEnv(client=object(), **kwargs)


def test_the_shapes_and_the_mask():
    world = _world(
        agents=[
            Agent("a", 4.0, 0.0, 0.1, 0.0, True),
            Agent("b", 1.0, 0.0, 0.0, 0.2, True),
            Agent("c", 2.0, 1.0, 0.0, 0.0, True),
        ]
    )
    observation = build_structured_observation(world, _task(), 4)
    assert observation[EGO_KEY].shape == (EGO_SIZE,)
    assert observation[GOAL_KEY].shape == (GOAL_SIZE,)
    assert observation[NEIGHBOURS_KEY].shape == (4, NEIGHBOUR_SIZE)
    assert observation[MASK_KEY].shape == (4,)
    assert observation[MASK_KEY].dtype == bool
    assert observation[MASK_KEY].tolist() == [True, True, True, False]


def test_the_neighbours_are_nearest_first():
    world = _world(
        agents=[
            Agent("far", 9.0, 0.0, 0.0, 0.0, True),
            Agent("near", 1.5, 0.0, 0.3, 0.0, True),
        ]
    )
    observation = build_structured_observation(world, _task(), 4)
    assert observation[NEIGHBOURS_KEY][0][:2] == pytest.approx([1.5, 0.0])
    assert observation[NEIGHBOURS_KEY][0][2] == pytest.approx(0.3)


def test_more_neighbours_than_room_are_dropped():
    world = _world(
        agents=[Agent(f"a{index}", float(index + 1), 0.0, 0.0, 0.0, True) for index in range(6)]
    )
    observation = build_structured_observation(world, _task(), 2)
    assert observation[MASK_KEY].tolist() == [True, True]
    assert observation[NEIGHBOURS_KEY][0][0] == pytest.approx(1.0)
    assert observation[NEIGHBOURS_KEY][1][0] == pytest.approx(2.0)


def test_an_empty_neighbourhood_is_all_padding():
    observation = build_structured_observation(_world(), _task(), 3)
    assert not observation[MASK_KEY].any()
    assert np.all(observation[NEIGHBOURS_KEY] == 0.0)
    assert observation[NEIGHBOURS_KEY].shape == (3, NEIGHBOUR_SIZE)


def test_a_missing_goal_leaves_the_goal_fields_empty():
    observation = build_structured_observation(
        _world(), _task(distance=None, bearing=None), 2
    )
    assert np.all(observation[GOAL_KEY] == 0.0)
    assert observation[EGO_KEY][:3] == pytest.approx([0.0, 0.0, 0.0])


def test_the_ego_features_carry_the_error_to_the_goal():
    observation = build_structured_observation(
        _world(linear=0.5, angular=0.25), _task(distance=4.0, bearing=math.pi / 2), 2
    )
    ego = observation[EGO_KEY]
    assert ego[0] == pytest.approx(0.0, abs=1e-6)
    assert ego[1] == pytest.approx(4.0)
    assert ego[2] == pytest.approx(math.pi / 2)
    assert ego[3] == pytest.approx(0.5)
    assert ego[4] == pytest.approx(0.25)
    assert observation[GOAL_KEY][2] == pytest.approx(4.0)


def test_the_observation_packs_into_what_the_networks_take():
    world = _world(agents=[Agent("a", 1.0, 0.5, 0.0, 0.0, True)])
    observation = build_structured_observation(world, _task(), 4)
    ego, neighbours, mask = pack_observation(observation)
    assert ego.shape == (EGO_SIZE + GOAL_SIZE,)
    assert neighbours.shape == (4, NEIGHBOUR_SIZE)
    assert mask.tolist() == [True, False, False, False]


def test_the_environment_publishes_the_structured_observation_without_a_session():
    env = _env(max_neighbours=3)
    world = _world(agents=[Agent("a", 1.0, 0.0, 0.0, 0.0, True)])
    observation = env.observation(world, _task())
    assert env.observation_space["neighbours"].shape == (3, NEIGHBOUR_SIZE)
    assert env.observation_space["mask"].n == 3
    assert env.observation_space.contains(observation)


def test_the_environment_keeps_the_default_room_for_neighbours():
    assert _env().max_neighbours == DEFAULT_MAX_NEIGHBOURS == 8
    with pytest.raises(ValueError, match="max_neighbours"):
        _env(max_neighbours=0)


def test_an_action_index_becomes_a_command_through_the_base_class():
    env = _env(control_period=0.5, max_angular=4.0)
    assert env.action_space.n == len(env.social_actions) == 80
    assert env.action_to_command(0) == (0.0, 0.0)

    quarter_turn = env.social_actions.heading_offsets[4]
    index = env.social_actions.index_of(0.75, quarter_turn / 0.5)
    assert env.action_to_command(index) == pytest.approx((0.75, quarter_turn / 0.5))


def test_the_command_is_clipped_by_the_base_class():
    env = _env(control_period=0.5, max_angular=1.0)
    index = env.social_actions.index_of(0.75, env.social_actions.heading_offsets[4] / 0.5)
    assert env.action_to_command(index) == pytest.approx((0.75, 1.0))


def test_an_index_outside_the_action_set_is_refused():
    env = _env()
    with pytest.raises(ValueError, match="outside the"):
        env.action_to_command(len(env.social_actions))


def test_min_human_distance_reads_the_agents_and_the_crowd():
    env = _env()
    assert env.min_human_distance(_world()) is None
    assert env.min_human_distance(
        _world(agents=[Agent("a", 3.0, 4.0, 0.0, 0.0, True)])
    ) == pytest.approx(5.0)
    assert env.min_human_distance(
        _world(humans=[Human(1, 0.0, 0.5, 0.0, 0.0, 0.0)], pose=Pose2D(0.0, 0.0, 0.0))
    ) == pytest.approx(0.5)


def test_the_reward_is_the_social_reward_of_the_two_task_states():
    env = _env()
    world = _world(agents=[Agent("a", 3.0, 0.0, 0.0, 0.0, True)])
    previous, current = _task(distance=5.0), _task(distance=4.0, elapsed=0.2)
    expected = 1.0 - 0.1 * 0.2 - env.social_reward.social_penalty(3.0)
    assert env.reward(previous, current, (0.0, 0.0), world) == pytest.approx(expected)


def test_the_reward_charges_a_crash_and_pays_an_arrival():
    env = _env()
    previous, moving = _task(distance=5.0), _task(distance=4.0)
    crashed = _task(distance=4.0, collision=True)
    arrived = _task(distance=4.0, goal_reached=True)
    assert env.reward(previous, crashed, (0.0, 0.0), _world()) < env.reward(
        previous, moving, (0.0, 0.0), _world()
    )
    assert env.reward(previous, arrived, (0.0, 0.0), _world()) > env.reward(
        previous, moving, (0.0, 0.0), _world()
    )
