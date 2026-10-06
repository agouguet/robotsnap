"""Actor-critic semantics, paper reward and portable GA3C-CADRL checkpoints."""
import numpy as np
import pytest

pytest.importorskip("torch")

import inspect
from dataclasses import replace

import torch

from robotsnap.envs.task import TaskState
from robotsnap.envs.world import Agent, Pose2D, World
from robotsnap.models.ga3c_cadrl import (
    GA3C_CADRL_PAPER_COMMANDS, Ga3cAgent, Ga3cCadrlEnv, ga3c_cadrl_value,
)
from robotsnap.models.social import SocialNavEnv, build_structured_observation


def task(**kwargs):
    state = TaskState(1.0, 0.0, False, False, False, None, None, 0.0, 0.0, 0)
    return replace(state, **kwargs)


def world(distance=None):
    agents = () if distance is None else (Agent("human", distance, 0., 0., 0., True),)
    return World(pose=Pose2D(0., 0., 0.), agents=agents)


@pytest.fixture
def observation():
    return build_structured_observation(world(0.7), task(), 4)


@pytest.mark.parametrize("flags,distance,expected", [
    ({"goal_reached": True}, None, 1.0),
    ({"goal_reached": True, "collision": True}, None, 1.0),
    ({"collision": True}, None, -0.25),
    ({"out_of_bounds": True}, None, -0.25),
    ({}, 0.1, -0.095), ({}, 0.2, 0.0), ({}, 0.5, 0.0),
    ({}, 0.0, 0.0), ({}, None, 0.0),
])
def test_eq12_reward_has_no_time_penalty(flags, distance, expected):
    env = Ga3cCadrlEnv(client=object())
    current = task(**flags)
    for elapsed in (0., 19., 300.):
        assert env.reward(task(), replace(current, elapsed_seconds=elapsed),
                          (0., 0.), world(distance)) == pytest.approx(expected)


def test_environment_keeps_shared_contract():
    env = Ga3cCadrlEnv(client=object())
    shared = SocialNavEnv(client=object())
    for name in ("__init__", "make_observation_space", "observation", "action_to_command"):
        assert getattr(Ga3cCadrlEnv, name) is getattr(SocialNavEnv, name)
    assert Ga3cCadrlEnv.reward_parameters is SocialNavEnv.reward_parameters
    assert env.social_reward == shared.social_reward
    assert env.action_space.n == 80
    table = np.asarray(GA3C_CADRL_PAPER_COMMANDS)
    assert table.shape == (11, 2)
    np.testing.assert_allclose(table[:5, 1], np.linspace(-np.pi / 6, np.pi / 6, 5))
    np.testing.assert_array_equal(table[:, 0], [1.] * 5 + [0.5] * 3 + [0.] * 3)


def test_network_shapes_and_masked_padding_invariance():
    torch.manual_seed(4)
    model = ga3c_cadrl_value(ego_features=10, neighbour_features=4, actions=7, hidden=8)
    rng = np.random.default_rng(3)
    ego = rng.normal(size=(2, 10)).astype(np.float32)
    neighbours = rng.normal(size=(2, 5, 4)).astype(np.float32)
    mask = np.array([[True, False, True, False, False], [False] * 5])
    value, logits = model(ego, neighbours, mask)
    assert value.shape == (2,)
    assert logits.shape == (2, 7)
    permuted = neighbours.copy()
    permuted[0, [1, 3, 4]] = neighbours[0, [4, 1, 3]]
    permuted[1] = neighbours[1, ::-1]
    value2, logits2 = model(ego, permuted, mask)
    torch.testing.assert_close(logits, logits2)
    torch.testing.assert_close(value, value2)
    permuted[~mask] = np.nan
    torch.testing.assert_close(model(ego, permuted, mask)[1], logits)
    # A truly empty neighbour axis agrees with a fully masked sequence.
    empty = model(ego[1:], np.empty((1, 0, 4)), np.empty((1, 0), dtype=bool))
    torch.testing.assert_close(empty[0], value[1:])
    torch.testing.assert_close(empty[1], logits[1:])


def probability(agent, observation, action):
    ego, neighbours, mask = agent._state(observation)
    with torch.no_grad():
        _, logits = agent.online(ego[None], neighbours[None], mask[None])
        return float(logits.softmax(-1)[0, action])


@pytest.mark.parametrize("reward,direction", [(100., 1), (-100., -1)])
def test_policy_update_sign(observation, reward, direction):
    agent = Ga3cAgent(actions=3, hidden=16, normalise=False, seed=8,
                      learning_rate=0.01, entropy_coefficient=0., n_steps=1)
    action = agent.act(observation)
    before = probability(agent, observation, action)
    agent.observe(observation, action, reward, observation, True)
    assert np.isfinite(agent.learn())
    after = probability(agent, observation, action)
    assert direction * (after - before) > 0, (before, after)


def test_rollout_threshold_and_done_flush(observation):
    agent = Ga3cAgent(actions=3, hidden=8, normalise=False, seed=9, n_steps=3)
    assert agent.learn() is None
    for index in range(3):
        agent.observe(observation, 1, 1., observation, False)
        loss = agent.learn()
        if index < 2:
            assert loss is None
        else:
            assert isinstance(loss, float) and np.isfinite(loss)
    agent.observe(observation, 1, 1., observation, True)
    assert np.isfinite(agent.learn())
    assert agent.steps == 2
    assert not agent._rollout


def test_greedy_and_default_epsilon(observation):
    agent = Ga3cAgent(actions=3, hidden=8, seed=5)
    assert agent.epsilon == 0.0
    assert agent.act(observation, greedy=True) == agent.act(observation, greedy=True)


def test_pending_cache_copies_and_defensive_observe(observation):
    agent = Ga3cAgent(actions=3, normalise=False, seed=3, hidden=8)
    action = agent.act(observation)
    pending = agent._pending
    agent.observe(observation, action, 1., observation, False)
    assert agent._rollout[-1][2] is pending[2]
    assert agent._pending is None
    original_next = agent._rollout[-1][-1][0].copy()
    observation["ego"][:] += 2
    np.testing.assert_array_equal(agent._rollout[-1][-1][0], original_next)
    agent.act(observation)
    observation["ego"][:] += 2
    pending = agent._pending
    agent.observe(observation, 1, 1., observation, True)
    assert agent._rollout[-1][2] is not pending[2]
    assert np.isfinite(agent.learn())


def test_checkpoint_roundtrip(observation, tmp_path):
    agent = Ga3cAgent(actions=3, hidden=8, seed=7, explore_epsilon=0.25, n_steps=1)
    agent.observe(observation, 1, 1., observation, True)
    agent.learn()
    greedy = agent.act(observation, greedy=True)
    path = tmp_path / "ga3c.pt"
    assert agent.save(path) == str(path)
    document = torch.load(path, map_location="cpu")
    assert document["algorithm"] == "ga3c_cadrl"
    restored = Ga3cAgent.load(path)
    assert isinstance(restored, Ga3cAgent)
    assert restored.epsilon == agent.epsilon == 0.25
    assert restored.steps == agent.steps
    assert restored.norm.state() == agent.norm.state()
    assert restored.act(observation, greedy=True) == greedy
    torch.testing.assert_close(restored.online.action_table, agent.online.action_table)
    assert restored.config() == agent.config()


def test_checkpoint_rejects_other_formats_and_algorithms(tmp_path):
    agent = Ga3cAgent(actions=2, hidden=8)
    path = tmp_path / "invalid.pt"
    agent.save(path)
    document = torch.load(path, map_location="cpu")
    document["algorithm"] = "cadrl"
    torch.save(document, path)
    with pytest.raises(ValueError, match="not a 'ga3c_cadrl' one"):
        Ga3cAgent.load(path)
    document["format"] = "other"
    torch.save(document, path)
    with pytest.raises(ValueError, match="is not a .* checkpoint"):
        Ga3cAgent.load(path)


def test_paper_signature_defaults():
    signature = inspect.signature(Ga3cAgent.__init__)
    for name, expected in {"learning_rate": 2e-5, "entropy_coefficient": 1e-4,
                           "gamma": 0.97, "batch_size": 100, "n_steps": 5}.items():
        assert signature.parameters[name].default == expected
