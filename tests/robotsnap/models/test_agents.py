"""Tests for the shared DQN and the agents each method builds on it."""

import numpy as np
import pytest

pytest.importorskip("torch")

from robotsnap.models import load_agent, make_agent  # noqa: E402 - after the torch skip
from robotsnap.models.cadrl import CadrlAgent  # noqa: E402
from robotsnap.models.sarl import SarlAgent  # noqa: E402
from robotsnap.models.social import (  # noqa: E402 - after the torch skip
    EGO_SIZE,
    GOAL_SIZE,
    NEIGHBOUR_SIZE,
    SocialActionSpace,
)

CLASSES = [CadrlAgent, SarlAgent]


def _observation(rng: np.random.Generator, neighbours: int = 4) -> dict:
    mask = np.zeros(neighbours, dtype=bool)
    mask[:2] = True
    return {
        "ego": rng.normal(size=EGO_SIZE).astype(np.float32),
        "goal": rng.normal(size=GOAL_SIZE).astype(np.float32),
        "neighbours": rng.normal(size=(neighbours, NEIGHBOUR_SIZE)).astype(np.float32),
        "mask": mask,
    }


@pytest.mark.parametrize("cls", CLASSES)
def test_acting_returns_an_index_of_the_action_set(cls):
    rng = np.random.default_rng(0)
    agent = cls(actions=5, hidden=8, seed=0)
    observation = _observation(rng)
    for _ in range(20):
        assert 0 <= agent.act(observation) < 5
        assert 0 <= agent.act(observation, greedy=True) < 5


@pytest.mark.parametrize("cls", CLASSES)
def test_a_few_learn_steps_run_and_do_not_crash(cls):
    rng = np.random.default_rng(1)
    agent = cls(actions=4, hidden=8, batch_size=4, buffer_size=32, seed=0)
    losses = []
    for _ in range(8):
        observation = _observation(rng)
        action = agent.act(observation)
        agent.observe(
            observation,
            action,
            float(rng.normal()),
            _observation(rng),
            False,
        )
        losses.append(agent.learn())
    assert any(loss is not None for loss in losses)
    assert all(np.isfinite(loss) for loss in losses if loss is not None)
    assert agent.epsilon < agent.epsilon_start


@pytest.mark.parametrize("cls", CLASSES)
def test_learn_waits_for_a_full_batch(cls):
    rng = np.random.default_rng(2)
    agent = cls(actions=3, hidden=8, batch_size=8, seed=0)
    observation = _observation(rng)
    for _ in range(3):
        agent.observe(observation, 0, 1.0, observation, False)
    assert agent.learn() is None


@pytest.mark.parametrize("cls", CLASSES)
def test_a_saved_agent_replays_the_same_greedy_action(cls, tmp_path):
    rng = np.random.default_rng(3)
    table = np.arange(12, dtype=np.float32).reshape(6, 2)
    agent = cls(actions=6, hidden=8, seed=1, action_table=table)
    observation = _observation(rng)
    # Give the running normaliser some statistics so the checkpoint has to
    # carry them, not just the weights.
    for _ in range(5):
        agent.observe(observation, 2, 1.0, _observation(rng), False)

    path = tmp_path / "agent.pt"
    agent.save(path)
    reloaded = load_agent(path)

    assert isinstance(reloaded, cls)
    assert reloaded.act(observation, greedy=True) == agent.act(observation, greedy=True)
    assert reloaded.epsilon == pytest.approx(agent.epsilon)
    assert reloaded.config()["action_table"] == agent.config()["action_table"]
    assert reloaded.config()["hidden"] == agent.config()["hidden"]


@pytest.mark.parametrize("cls", CLASSES)
def test_a_checkpoint_of_the_other_algorithm_is_refused(cls, tmp_path):
    other = CadrlAgent if cls is SarlAgent else SarlAgent
    path = tmp_path / "other.pt"
    other(actions=3, hidden=8).save(path)
    with pytest.raises(ValueError, match="not a"):
        cls.load(path)


def test_make_agent_sizes_the_agent_from_the_environment():
    class FakeEnv:
        social_actions = SocialActionSpace()

    env = FakeEnv()
    agent = make_agent("sarl", env, seed=0)
    assert isinstance(agent, SarlAgent)
    assert agent.actions == len(env.social_actions)
    assert agent.ego_features == EGO_SIZE + GOAL_SIZE
    assert agent.neighbour_features == NEIGHBOUR_SIZE
    with pytest.raises(ValueError, match="unknown algorithm"):
        make_agent("ppo", env)


def test_an_agent_consumes_what_the_environment_publishes():
    """The layout the environment writes and the one the networks read agree."""
    from robotsnap.envs.base import TaskState
    from robotsnap.envs.world import Agent, Pose2D, World
    from robotsnap.models.social import SocialNavEnv

    env = SocialNavEnv(client=object(), max_neighbours=3)
    world = World(
        pose=Pose2D(0.0, 0.0, 0.0),
        agents=(Agent("a", 1.0, 0.5, 0.2, 0.0, True),),
    )
    task = TaskState(
        distance_to_goal=4.0,
        bearing_to_goal=0.1,
        goal_reached=False,
        collision=False,
        out_of_bounds=False,
        min_lidar=None,
        nearest_agent_distance=1.0,
        elapsed_seconds=0.0,
        wall_seconds=0.0,
        steps=0,
    )
    observation = env.observation(world, task)
    agent = make_agent("cadrl", env, hidden=8, seed=0)
    action = agent.act(observation, greedy=True)
    assert 0 <= action < len(env.social_actions)

    # The two padded rows of the environment's observation are invisible to it,
    # whoever wrote them: the same value is read with them filled with noise.
    padded = dict(observation)
    padded["neighbours"] = observation["neighbours"].copy()
    padded["neighbours"][1:] = 1e4
    assert agent.act(padded, greedy=True) == action
