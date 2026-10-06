"""Tests for the shared ``V(s)`` core: the model, the reward, and the learner.

None of the tests here needs a session, a socket or a running Unity: the
environment is built with a stand-in client (the base class does not touch its
client until an episode starts), and the geometry is checked on arrays written
by hand. What is under test is the fidelity of the rewrite: the propagation
model, the immediate reward the lookahead reads, the ``argmax_a [R_hat +
gamma * V]`` rule of the greedy action, and the ``r + gamma * V(s')`` target
with no maximum over the actions.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from robotsnap.models.cadrl import CadrlAgent  # noqa: E402 - after the torch skip
from robotsnap.models.lookahead import (  # noqa: E402
    LookaheadAgent,
    propagate,
    state_value_base,
)
from robotsnap.models.sarl import SarlAgent  # noqa: E402
from robotsnap.models.social import (  # noqa: E402
    EGO_SIZE,
    GOAL_SIZE,
    NEIGHBOUR_SIZE,
    SocialNavEnv,
)


def _env(**kwargs) -> SocialNavEnv:
    """An environment with a stand-in client, so no port is ever bound."""
    return SocialNavEnv(client=object(), **kwargs)


def _ego(goal_x: float, goal_y: float, *, heading_error: float = 0.0) -> np.ndarray:
    """One packed ego vector with the goal at ``(goal_x, goal_y)``."""
    distance = float(np.hypot(goal_x, goal_y))
    bearing = float(np.arctan2(goal_y, goal_x))
    return np.array(
        [
            goal_x,
            goal_y,
            heading_error,
            0.0,
            0.0,
            goal_x,
            goal_y,
            distance,
            float(np.cos(bearing)),
            float(np.sin(bearing)),
        ],
        dtype=np.float32,
    )


def _observation(goal_x: float, goal_y: float, neighbours=(), mask=()) -> dict:
    goal = _ego(goal_x, goal_y)
    rows = np.zeros((max(1, len(neighbours)), NEIGHBOUR_SIZE), dtype=np.float32)
    flags = np.zeros(max(1, len(neighbours)), dtype=bool)
    for index, row in enumerate(neighbours):
        rows[index] = row
        flags[index] = True
    for index, flag in enumerate(mask):
        flags[index] = bool(flag)
    return {
        "ego": goal[:EGO_SIZE].copy(),
        "goal": goal[EGO_SIZE:].copy(),
        "neighbours": rows,
        "mask": flags,
    }


# ---------------------------------------------------------------------------
# The model: one step of the unicycle, in the robot frame.
# ---------------------------------------------------------------------------


def test_a_straight_command_keeps_a_neighbour_ahead_and_closes_the_gap():
    ego = _ego(5.0, 0.0)[None]
    neighbours = np.zeros((1, 1, NEIGHBOUR_SIZE), dtype=np.float32)
    neighbours[0, 0] = (2.0, 0.0, 0.0, 0.0)
    mask = np.ones((1, 1), dtype=bool)

    ego2, neighbours2, mask2 = propagate(
        ego, neighbours, mask, np.array([[0.5, 0.0]]), 0.1
    )

    # The robot moved 0.05 m forward, so a fixed neighbour is that much closer
    # and still straight ahead.
    assert neighbours2[0, 0, 0, 0] == pytest.approx(1.95, abs=1e-6)
    assert neighbours2[0, 0, 0, 1] == pytest.approx(0.0, abs=1e-6)
    assert np.hypot(*neighbours2[0, 0, 0, :2]) == pytest.approx(1.95, abs=1e-6)
    assert ego2[0, 0, EGO_SIZE + 1] == pytest.approx(0.0, abs=1e-6)
    assert mask2[0, 0, 0]


def test_a_pure_rotation_arcs_a_neighbour_without_changing_its_distance():
    angle = 0.1
    ego = _ego(5.0, 0.0)[None]
    neighbours = np.zeros((1, 1, NEIGHBOUR_SIZE), dtype=np.float32)
    neighbours[0, 0] = (2.0, 0.0, 0.0, 0.0)
    mask = np.ones((1, 1), dtype=bool)

    _, neighbours2, _ = propagate(
        ego, neighbours, mask, np.array([[0.0, angle / 0.1]]), 0.1
    )

    x, y = neighbours2[0, 0, 0, :2]
    # The frame turned left by `angle`; the neighbour, which was on the axis,
    # sweeps an arc of that angle to the right and its distance is untouched.
    assert float(np.hypot(x, y)) == pytest.approx(2.0, abs=1e-6)
    assert float(np.arctan2(y, x)) == pytest.approx(-angle, abs=1e-6)


def test_a_neighbour_velocity_turns_with_the_frame():
    angle = 0.2
    ego = _ego(5.0, 0.0)[None]
    neighbours = np.zeros((1, 1, NEIGHBOUR_SIZE), dtype=np.float32)
    neighbours[0, 0] = (3.0, 0.0, 0.0, 0.5)
    mask = np.ones((1, 1), dtype=bool)

    _, neighbours2, _ = propagate(
        ego, neighbours, mask, np.array([[0.0, angle / 0.1]]), 0.1
    )

    vx, vy = neighbours2[0, 0, 0, 2:4]
    assert vx == pytest.approx(0.5 * np.sin(angle), abs=1e-6)
    assert vy == pytest.approx(0.5 * np.cos(angle), abs=1e-6)


def test_the_goal_follows_the_same_transform_as_a_neighbour():
    ego = _ego(4.0, 0.0)[None]
    neighbours = np.zeros((1, 1, NEIGHBOUR_SIZE), dtype=np.float32)
    neighbours[0, 0] = (4.0, 0.0, 0.0, 0.0)
    mask = np.ones((1, 1), dtype=bool)
    command = np.array([[0.4, 0.7]])

    ego2, neighbours2, _ = propagate(ego, neighbours, mask, command, 0.1)

    assert ego2[0, 0, EGO_SIZE + 0 : EGO_SIZE + 2] == pytest.approx(
        neighbours2[0, 0, 0, :2], abs=1e-6
    )


def test_the_mask_and_the_masked_rows_are_left_alone():
    ego = np.stack([_ego(5.0, 0.0), _ego(5.0, 0.0)])
    neighbours = np.full((2, 3, NEIGHBOUR_SIZE), 1e6, dtype=np.float32)
    neighbours[:, 0] = (2.0, 0.5, -0.2, 0.1)
    mask = np.zeros((2, 3), dtype=bool)
    mask[:, 0] = True
    commands = np.array([[0.5, 0.0], [0.0, 1.0]])

    _, neighbours2, mask2 = propagate(ego, neighbours, mask, commands, 0.1)

    assert mask2.shape == (2, 2, 3)
    assert not mask2[:, :, 1:].any()
    assert np.all(neighbours2[:, :, 1:, :] == 0.0)
    assert mask2[:, :, 0].all()


# ---------------------------------------------------------------------------
# The immediate reward the lookahead reads.
# ---------------------------------------------------------------------------


def _propagated(goal_distance: float, neighbour=None, previous=None) -> dict:
    """A propagated state, in the mapping the environment's reward expects."""
    neighbours = (
        np.zeros((1, NEIGHBOUR_SIZE), dtype=np.float64)
        if neighbour is None
        else np.asarray(neighbour, dtype=np.float64).reshape(1, NEIGHBOUR_SIZE)
    )
    mask = np.array([neighbour is not None])
    return {
        "ego": np.zeros(EGO_SIZE, dtype=np.float64),
        "goal": np.array(
            [goal_distance, 0.0, goal_distance, 1.0, 0.0], dtype=np.float64
        ),
        "neighbours": neighbours,
        "mask": mask,
        "previous_distance_to_goal": previous,
    }


def test_a_propagated_step_towards_the_goal_is_paid_positive_progress():
    env = _env()

    closer = _propagated(4.0, previous=5.0)
    farther = _propagated(6.0, previous=5.0)

    assert env.lookahead_reward(closer, 1) > 0.0
    assert env.lookahead_reward(farther, 1) < 0.0
    assert env.lookahead_reward(closer, 1) > env.lookahead_reward(farther, 1)


def test_a_propagated_neighbour_inside_the_collision_distance_is_penalised():
    # Isolate the collision term: no progress, no time charge and no comfort
    # charge, so the only difference between the two states is the flat price.
    env = _env(
        social_parameters={
            "progress_weight": 0.0,
            "time_penalty": 0.0,
            "personal_space_weight": 0.0,
            "too_close_distance": 0.0,
            "too_close_penalty": 0.0,
        }
    )
    inside = _propagated(5.0, neighbour=(0.1, 0.0, 0.0, 0.0))
    outside = _propagated(5.0, neighbour=(3.0, 0.0, 0.0, 0.0))

    assert env.lookahead_reward(inside, 0) == pytest.approx(-env.social_reward.collision_penalty)
    assert env.lookahead_reward(outside, 0) == pytest.approx(0.0)


def test_the_collision_threshold_is_the_environment_threshold():
    env = _env(social_parameters={"progress_weight": 0.0, "time_penalty": 0.0})
    just_inside = _propagated(5.0, neighbour=(env.collision_distance - 0.01, 0.0, 0.0, 0.0))
    just_outside = _propagated(5.0, neighbour=(env.collision_distance + 0.01, 0.0, 0.0, 0.0))

    assert env.lookahead_reward(just_inside, 0) < env.lookahead_reward(just_outside, 0)


# ---------------------------------------------------------------------------
# The greedy action: argmax_a [ R_hat + gamma * V ].
# ---------------------------------------------------------------------------


def _constant_value_class(constant: float) -> type:
    """A state-value module that ignores its input and returns ``constant``."""
    import torch.nn as nn

    class ConstantValue(nn.Module):
        def __init__(self, *, ego_features: int, neighbour_features: int, hidden: int = 64, **_: object):
            super().__init__()
            self.bias = nn.Parameter(torch.tensor(float(constant)))

        def forward(self, ego, neighbours, mask):
            batch = torch.as_tensor(np.asarray(ego)).shape[0]
            return self.bias.expand(batch)

    return ConstantValue


class _ConstantAgent(LookaheadAgent):
    name = "constant"
    value_class = _constant_value_class(0.0)


def test_a_greedy_action_maximises_the_estimated_reward_over_the_candidates():
    env = _env(max_neighbours=1)
    observation = _observation(3.0, 0.0)
    commands = np.array([[0.0, 0.0], [0.5, 0.0], [0.0, 0.75]], dtype=np.float32)
    agent = _ConstantAgent(
        actions=len(commands),
        hidden=8,
        seed=0,
        action_table=commands,
        control_period=env.control_period,
        reward_callable=env.lookahead_reward,
    )

    # The value net is constant, so `gamma * V` is the same for every candidate
    # and the argmax can only be driven by `R_hat`: standing still and turning in
    # place pay only the time charge, and driving towards the goal pays progress.
    chosen = agent.act(observation, greedy=True)

    ego, neighbours, mask = agent._pack(observation)
    ego2, neighbours2, mask2 = propagate(
        ego[None], neighbours[None], mask[None], commands, env.control_period
    )
    previous = agent._goal_distance(ego)
    scores = [
        env.lookahead_reward(
            agent.propagated_state(ego2[0, index], neighbours2[0, index], mask2[0, index], previous),
            index,
        )
        for index in range(len(commands))
    ]

    assert chosen == int(np.argmax(scores)) == 1


def test_the_value_of_a_state_is_a_single_scalar():
    """No action axis exists, so no maximum over the actions can be taken."""
    import torch.nn as nn

    base = state_value_base()

    class MeanPooled(base):
        def pool(self, encoded, keep):
            weights = keep.to(encoded.dtype).unsqueeze(-1)
            return (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)

    model = MeanPooled(
        ego_features=EGO_SIZE + GOAL_SIZE, neighbour_features=NEIGHBOUR_SIZE, hidden=8
    )
    ego = np.stack([_ego(5.0, 0.0), _ego(1.0, 2.0)])
    neighbours = np.zeros((2, 3, NEIGHBOUR_SIZE), dtype=np.float32)
    mask = np.zeros((2, 3), dtype=bool)
    mask[0, 0] = True
    values = model(ego, neighbours, mask)
    assert tuple(values.shape) == (2,)
    assert isinstance(model, nn.Module)


# ---------------------------------------------------------------------------
# The TD target: y = r + gamma * V(s'), no max.
# ---------------------------------------------------------------------------


def _transition(agent, observation, action, reward, next_observation, done) -> tuple:
    current = [entry.copy() for entry in agent._pack(observation)]
    following = [entry.copy() for entry in agent._pack(next_observation)]
    return (*current, int(action), float(reward), *following, bool(done))


@pytest.mark.parametrize("cls", [CadrlAgent, SarlAgent])
def test_the_update_target_is_r_plus_gamma_v_of_the_next_state(cls):
    """The loss matches a manually computed ``r + gamma * (1-done) * V_target(s')``."""
    agent = cls(actions=3, hidden=8, gamma=0.5, normalise=False, seed=0, batch_size=2)
    batch = [
        _transition(agent, _observation(5.0, 0.0), 1, 1.5, _observation(1.0, 0.0), False),
        _transition(agent, _observation(2.0, 1.0), 2, -0.5, _observation(2.0, 1.0), True),
    ]

    ego = torch.as_tensor(np.stack([row[0] for row in batch]))
    neighbours = torch.as_tensor(np.stack([row[1] for row in batch]))
    mask = torch.as_tensor(np.stack([row[2] for row in batch])).bool()
    next_ego = torch.as_tensor(np.stack([row[5] for row in batch]))
    next_neighbours = torch.as_tensor(np.stack([row[6] for row in batch]))
    next_mask = torch.as_tensor(np.stack([row[7] for row in batch])).bool()
    rewards = torch.as_tensor([row[4] for row in batch], dtype=torch.float32)
    dones = torch.as_tensor([float(row[8]) for row in batch], dtype=torch.float32)

    with torch.no_grad():
        values = agent.online(ego, neighbours, mask)
        future = agent.target(next_ego, next_neighbours, next_mask)
        expected = torch.nn.functional.mse_loss(
            values, rewards + agent.gamma * (1.0 - dones) * future
        )

    loss = agent.update(batch)

    assert loss == pytest.approx(float(expected), rel=1e-4, abs=1e-5)


@pytest.mark.parametrize("cls", [CadrlAgent, SarlAgent])
def test_with_gamma_zero_the_value_converges_to_the_reward(cls):
    """``gamma = 0`` leaves ``y = r``, so the learned ``V(s)`` is the reward."""
    agent = cls(
        actions=3,
        hidden=8,
        learning_rate=1e-2,
        gamma=0.0,
        normalise=False,
        seed=0,
        batch_size=2,
        target_update=10_000,
    )
    rich = _observation(5.0, 0.0)
    poor = _observation(1.0, 2.0)
    batch = [
        _transition(agent, rich, 1, 3.0, rich, False),
        _transition(agent, poor, 2, -2.0, poor, False),
    ]

    for _ in range(300):
        agent.update(batch)

    ego = torch.as_tensor(np.stack([row[0] for row in batch]))
    neighbours = torch.as_tensor(np.stack([row[1] for row in batch]))
    mask = torch.as_tensor(np.stack([row[2] for row in batch])).bool()
    with torch.no_grad():
        values = agent.online(ego, neighbours, mask).tolist()

    assert values[0] == pytest.approx(3.0, abs=0.5)
    assert values[1] == pytest.approx(-2.0, abs=0.5)


# ---------------------------------------------------------------------------
# Save and load.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [CadrlAgent, SarlAgent])
def test_save_load_round_trip_carries_the_algorithm_and_the_policy(cls, tmp_path):
    from robotsnap.models import load_agent

    observation = _observation(4.0, 0.5, neighbours=[(2.0, 0.5, 0.0, 0.0)])
    agent = cls(actions=6, hidden=8, seed=1, action_table=np.arange(12, dtype=np.float32).reshape(6, 2))
    for _ in range(5):
        agent.observe(observation, 2, 1.0, _observation(1.0, 1.0), False)

    path = tmp_path / "agent.pt"
    agent.save(path)

    document = torch.load(path, map_location="cpu")
    assert document["algorithm"] == cls.name
    assert document["format"] == "robotsnap-social-agent-1"

    reloaded = load_agent(path)
    assert isinstance(reloaded, cls)
    assert reloaded.act(observation, greedy=True) == agent.act(observation, greedy=True)

    rebuilt = cls._from_document(document)
    assert rebuilt.act(observation, greedy=True) == agent.act(observation, greedy=True)


@pytest.mark.parametrize("cls", [CadrlAgent, SarlAgent])
def test_a_checkpoint_of_the_other_algorithm_is_refused(cls, tmp_path):
    other = SarlAgent if cls is CadrlAgent else CadrlAgent
    path = tmp_path / "other.pt"
    other(actions=3, hidden=8).save(path)
    with pytest.raises(ValueError, match="not a"):
        cls.load(path)
