"""Tests for RGL: the relational graph, the predictor, and the depth-d lookahead.

No test here needs a session, a socket or a running Unity: environments are
built with a stand-in client and every network is exercised on arrays written by
hand. What is under test is the fidelity of the method to Chen et al. (2020):
the graph over robot and masked neighbours, the separate prediction network
``fP``, and the width-clipped depth-``d`` lookahead whose value target keeps no
maximum over the actions.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from robotsnap.models.lookahead import propagate  # noqa: E402 - after the torch skip
from robotsnap.models.rgl import (  # noqa: E402
    AGENT,
    ENVIRONMENT,
    RGL_DEPTH,
    RGL_REWARD_PARAMETERS,
    RGL_WIDTH,
    RglAgent,
    RglEnv,
    rgl_class,
    rgl_predictor,
    rgl_value,
)
from robotsnap.models.social import (  # noqa: E402
    EGO_SIZE,
    GOAL_SIZE,
    NEIGHBOUR_SIZE,
    SocialActionSpace,
)

#: Small networks everywhere: the tests are about the plumbing, not the width.
TINY = dict(hidden=8, latent=8, mlp_hidden=(8,), value_hidden=(8,), predictor_hidden=(8,))


def _env(**kwargs) -> RglEnv:
    """The RGL task with a stand-in client, so no port is ever bound."""
    return RglEnv(client=object(), **kwargs)


def _state(batch: int = 3, count: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    ego = rng.normal(size=(batch, EGO_SIZE + GOAL_SIZE)).astype(np.float32)
    neighbours = rng.normal(size=(batch, count, NEIGHBOUR_SIZE)).astype(np.float32)
    mask = np.ones((batch, count), dtype=bool)
    return ego, neighbours, mask


def _model(**kwargs):
    options = {
        key: value
        for key, value in {**TINY, **kwargs}.items()
        if key != "predictor_hidden"
    }
    return rgl_value(
        ego_features=EGO_SIZE + GOAL_SIZE,
        neighbour_features=NEIGHBOUR_SIZE,
        **options,
    )


def _observation(goal_x: float, goal_y: float, neighbours=(), mask=()) -> dict:
    """One structured observation, with the goal in the robot frame."""
    distance = float(np.hypot(goal_x, goal_y))
    bearing = float(np.arctan2(goal_y, goal_x))
    goal = np.array(
        [goal_x, goal_y, distance, float(np.cos(bearing)), float(np.sin(bearing))],
        dtype=np.float32,
    )
    rows = np.zeros((max(1, len(neighbours)), NEIGHBOUR_SIZE), dtype=np.float32)
    flags = np.zeros(max(1, len(neighbours)), dtype=bool)
    for index, row in enumerate(neighbours):
        rows[index] = row
        flags[index] = True
    for index, flag in enumerate(mask):
        flags[index] = bool(flag)
    return {
        "ego": np.array([goal_x, goal_y, bearing, 0.0, 0.0], dtype=np.float32),
        "goal": goal,
        "neighbours": rows,
        "mask": flags,
    }


def _transition(agent, observation, action, reward, next_observation, done=False) -> tuple:
    current = [entry.copy() for entry in agent._pack(observation)]
    following = [entry.copy() for entry in agent._pack(next_observation)]
    return (*current, int(action), float(reward), *following, bool(done))


def _constant_value_class(constant: float) -> type:
    """A state-value module that ignores its input and returns ``constant``."""
    import torch.nn as nn

    class ConstantValue(nn.Module):
        def __init__(self, **_: object):
            super().__init__()
            self.bias = nn.Parameter(torch.tensor(float(constant)))

        def forward(self, ego, neighbours, mask):
            batch = torch.as_tensor(np.asarray(ego)).shape[0]
            return self.bias.expand(batch)

    return ConstantValue


class _PropagateAgent(RglAgent):
    """An RGL agent whose lookahead rolls the exact analytic model.

    The predictor is replaced by :func:`propagate`, so a lookahead test can
    control the geometry exactly and count the expansion calls through
    :attr:`fp_calls`.
    """

    value_class = _constant_value_class(0.0)

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.fp_calls = 0

    def _predict_batch(self, ego, neighbours, mask, actions):
        self.fp_calls += 1
        indices = np.asarray(list(actions), dtype=np.int64)
        ego2, neighbours2, mask2 = propagate(
            np.asarray(ego, dtype=np.float32)[None],
            np.asarray(neighbours, dtype=np.float32)[None],
            np.asarray(mask)[None],
            self.commands[indices],
            self.control_period,
        )
        return ego2[0], neighbours2[0], mask2[0]


# ---------------------------------------------------------------------------
# The relational value network.
# ---------------------------------------------------------------------------


def test_the_value_is_one_scalar_per_state():
    ego, neighbours, mask = _state(batch=2, count=4)
    values = _model()(ego, neighbours, mask)
    assert tuple(values.shape) == (2,)
    assert torch.isfinite(values).all()


def test_a_padded_neighbour_is_ignored_whatever_it_holds():
    ego, neighbours, mask = _state(batch=2, count=4, seed=2)
    mask = mask.copy()
    mask[:, 2:] = False
    model = _model()
    quiet = model(ego, neighbours, mask)

    changed = neighbours.copy()
    changed[:, 2:] = -12345.0
    noisy = model(ego, changed, mask)
    assert torch.allclose(quiet, noisy, atol=1e-6)


def test_a_state_with_no_neighbour_ignores_the_padded_rows():
    ego, neighbours, _ = _state(batch=2, count=4, seed=3)
    model = _model()
    empty = np.zeros((2, 4), dtype=bool)
    quiet = model(ego, neighbours, empty)
    noisy = model(ego, np.full_like(neighbours, -7.0), empty)
    assert torch.isfinite(quiet).all()
    assert torch.allclose(quiet, noisy, atol=1e-6)


def test_the_value_does_not_depend_on_the_neighbour_row_order():
    """Swapping two agents, padded rows included, cannot change the value."""
    ego, neighbours, mask = _state(batch=3, count=4, seed=4)
    mask[:, 3] = False
    model = _model()
    order = np.array([3, 0, 2, 1])
    straight = model(ego, neighbours, mask)
    permuted = model(ego, neighbours[:, order], mask[:, order])
    assert torch.allclose(straight, permuted, atol=1e-5)


def test_the_value_depends_on_the_couple_of_neighbours_not_only_each_one():
    """Information passes between neighbours, so the value is not additive.

    For a value that pooled independently encoded neighbours, the mixed
    difference ``v(A, B) - v(A, B') - v(A', B) + v(A', B')`` would vanish. It is
    the signature of the second relational layer, in which the robot's message
    comes from a neighbour whose own features already mixed in the others. The
    test is run in double precision and with a sharpened attention so the
    interaction is well above the floating-point floor.
    """
    # A fixed initialisation, so the interaction does not depend on the order
    # the rest of the suite left the global torch RNG in.
    torch.manual_seed(0)
    model = _model().double()
    for layer in model.graph.layers:
        layer.attention[-1].weight.data *= 5.0
        layer.attention[-1].bias.data *= 5.0

    rng = np.random.default_rng(1)
    ego = rng.normal(size=(1, EGO_SIZE + GOAL_SIZE)).astype(np.float64)
    mask = np.ones((1, 2), dtype=bool)

    def value(first, second):
        rows = np.concatenate([first, second], axis=1)
        with torch.no_grad():
            return float(model(ego, rows, mask)[0])

    largest = 0.0
    for _ in range(3):
        a = rng.normal(size=(1, 1, NEIGHBOUR_SIZE)).astype(np.float64) * 2.0
        a2 = rng.normal(size=(1, 1, NEIGHBOUR_SIZE)).astype(np.float64) * 2.0
        b = rng.normal(size=(1, 1, NEIGHBOUR_SIZE)).astype(np.float64) * 2.0
        b2 = rng.normal(size=(1, 1, NEIGHBOUR_SIZE)).astype(np.float64) * 2.0
        interaction = value(a, b) - value(a, b2) - value(a2, b) + value(a2, b2)
        largest = max(largest, abs(interaction))

    assert largest > 1e-5


# ---------------------------------------------------------------------------
# The state predictor.
# ---------------------------------------------------------------------------


def _predictor():
    return rgl_predictor(
        ego_features=EGO_SIZE + GOAL_SIZE,
        neighbour_features=NEIGHBOUR_SIZE,
        latent=16,
        layers=2,
        mlp_hidden=(16,),
        hidden=(16,),
    )


def _synthetic_batch(count, rng, commands, neighbours=3):
    goal = rng.uniform(-6.0, 6.0, size=(count, 2))
    goal[:, 0] = np.where(np.abs(goal[:, 0]) < 0.5, 0.5, goal[:, 0])
    distance = np.hypot(goal[:, 0], goal[:, 1])
    bearing = np.arctan2(goal[:, 1], goal[:, 0])
    ego = np.zeros((count, EGO_SIZE + GOAL_SIZE), dtype=np.float32)
    ego[:, 0] = goal[:, 0]
    ego[:, 1] = goal[:, 1]
    ego[:, 2] = bearing
    ego[:, 3] = rng.uniform(0.0, 1.0, size=count)
    ego[:, 4] = rng.uniform(-1.0, 1.0, size=count)
    ego[:, EGO_SIZE + 0] = goal[:, 0]
    ego[:, EGO_SIZE + 1] = goal[:, 1]
    ego[:, EGO_SIZE + 2] = distance
    ego[:, EGO_SIZE + 3] = goal[:, 0] / distance
    ego[:, EGO_SIZE + 4] = goal[:, 1] / distance
    rows = np.zeros((count, neighbours, NEIGHBOUR_SIZE), dtype=np.float32)
    rows[:, :, 0] = rng.uniform(-5.0, 5.0, size=(count, neighbours))
    rows[:, :, 1] = rng.uniform(-5.0, 5.0, size=(count, neighbours))
    rows[:, :, 2] = rng.uniform(-1.0, 1.0, size=(count, neighbours))
    rows[:, :, 3] = rng.uniform(-1.0, 1.0, size=(count, neighbours))
    mask = rng.random((count, neighbours)) < 0.75
    mask[:, 0] = True
    indices = rng.integers(0, len(commands), size=count)
    next_ego, next_neighbours, next_mask = propagate(ego, rows, mask, commands, 0.1)
    return (
        ego,
        rows,
        mask,
        commands[indices],
        next_ego[np.arange(count), indices],
        next_neighbours[np.arange(count), indices],
        next_mask[np.arange(count), indices],
    )


def _prediction_mse(predictor, batch):
    import torch.nn.functional as functional

    ego, rows, mask, actions, next_ego, next_neighbours, next_mask = batch
    predicted_ego, predicted_neighbours, _ = predictor(
        torch.as_tensor(ego),
        torch.as_tensor(rows),
        torch.as_tensor(mask),
        torch.as_tensor(actions),
    )
    weights = torch.as_tensor(next_mask).bool().unsqueeze(-1)
    ego_loss = functional.mse_loss(predicted_ego, torch.as_tensor(next_ego))
    squared = (predicted_neighbours - torch.as_tensor(next_neighbours)) ** 2
    float_weights = weights.to(squared.dtype)
    neighbour_loss = (squared * float_weights).sum() / (
        float_weights.sum().clamp(min=1.0) * predicted_neighbours.shape[-1]
    )
    return ego_loss + neighbour_loss


def test_the_predictor_returns_the_next_state_and_copies_the_mask():
    predictor = _predictor()
    ego, neighbours, mask = _state(batch=2, count=4, seed=5)
    actions = np.zeros((2, 2), dtype=np.float32)
    next_ego, next_neighbours, next_mask = predictor(ego, neighbours, mask, actions)
    assert tuple(next_ego.shape) == (2, EGO_SIZE + GOAL_SIZE)
    assert tuple(next_neighbours.shape) == (2, 4, NEIGHBOUR_SIZE)
    assert np.array_equal(next_mask.numpy(), mask)
    assert torch.isfinite(next_ego).all() and torch.isfinite(next_neighbours).all()


def test_the_predictor_learns_the_analytic_transition():
    """A few hundred L2 steps on propagate-labelled transitions must pay off."""
    commands = SocialActionSpace().commands
    predictor = _predictor()
    optimiser = torch.optim.Adam(predictor.parameters(), lr=1e-2)
    rng = np.random.default_rng(0)
    evaluation = _synthetic_batch(128, np.random.default_rng(999), commands)

    with torch.no_grad():
        before = float(_prediction_mse(predictor, evaluation))
    for _ in range(300):
        loss = _prediction_mse(predictor, _synthetic_batch(32, rng, commands))
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    with torch.no_grad():
        after = float(_prediction_mse(predictor, evaluation))

    assert after < before * 0.5


# ---------------------------------------------------------------------------
# The depth-d width-w lookahead.
# ---------------------------------------------------------------------------


def test_the_default_lookahead_is_depth_two_and_width_two():
    agent = RglAgent(actions=5, **TINY)
    assert (agent.depth, agent.width) == (RGL_DEPTH, RGL_WIDTH) == (2, 2)
    assert agent.config()["depth"] == 2 and agent.config()["width"] == 2


def test_a_depth_one_width_one_lookahead_maximises_the_immediate_reward():
    env = _env(max_neighbours=1)
    observation = _observation(3.0, 0.0)
    commands = np.array([[0.0, 0.0], [0.5, 0.0], [0.0, 0.75]], dtype=np.float32)
    agent = _PropagateAgent(
        actions=len(commands),
        depth=1,
        width=1,
        seed=0,
        action_table=commands,
        control_period=env.control_period,
        reward_callable=env.lookahead_reward,
        **TINY,
    )
    chosen = agent.act(observation, greedy=True)

    # With a constant value net the lookahead score is the immediate reward
    # alone, so the greedy action is the argmax over `R_hat`.
    ego, neighbours, mask = agent._pack(observation)
    ego2, neighbours2, mask2 = propagate(
        ego[None], neighbours[None], mask[None], commands, env.control_period
    )
    previous = agent._goal_distance(ego)
    scores = [
        env.lookahead_reward(
            agent.propagated_state(
                ego2[0, index], neighbours2[0, index], mask2[0, index], previous
            ),
            index,
        )
        for index in range(len(commands))
    ]
    assert chosen == int(np.argmax(scores))
    assert agent.fp_calls == 1


def test_the_search_visits_at_most_width_branches_per_level():
    env = _env(max_neighbours=1)
    observation = _observation(4.0, 0.5)
    commands = SocialActionSpace(speeds=2, headings=4).commands
    agent = _PropagateAgent(
        actions=len(commands),
        depth=2,
        width=2,
        seed=0,
        action_table=commands,
        control_period=env.control_period,
        reward_callable=env.lookahead_reward,
        **TINY,
    )
    assert 0 <= agent.act(observation, greedy=True) < len(commands)

    # The root is one node; every level after it expands at most `width` nodes.
    assert agent.last_search["expanded"] == [1, 2]
    assert all(size <= agent.width for size in agent.last_search["expanded"])
    assert agent.fp_calls == 1 + (agent.depth - 1) * agent.width
    assert agent.last_search["leaves"] == agent.width * len(commands)

    shallow = _PropagateAgent(
        actions=len(commands),
        depth=1,
        width=1,
        seed=0,
        action_table=commands,
        control_period=env.control_period,
        reward_callable=env.lookahead_reward,
        **TINY,
    )
    shallow.act(observation, greedy=True)
    assert shallow.last_search["expanded"] == [1]
    assert shallow.fp_calls == 1


# ---------------------------------------------------------------------------
# The learner: no max over the actions, plus the prediction loss.
# ---------------------------------------------------------------------------


def test_the_update_target_is_the_reward_with_no_maximum_over_the_actions():
    """``gamma = 0`` leaves ``y = r``: no action maximum can be hiding in it."""
    agent = RglAgent(
        actions=3,
        gamma=0.0,
        learning_rate=1e-2,
        normalise=False,
        seed=0,
        batch_size=2,
        target_update=10_000,
        **TINY,
    )
    rich = _observation(5.0, 0.0)
    poor = _observation(1.0, 2.0)
    batch = [
        _transition(agent, rich, 1, 3.0, rich),
        _transition(agent, poor, 2, -2.0, poor),
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


def test_the_update_carries_a_finite_prediction_loss():
    agent = RglAgent(actions=4, normalise=False, seed=0, **TINY)
    observation = _observation(3.0, 0.0, neighbours=[(2.0, 0.5, 0.0, 0.0)])
    batch = [
        _transition(agent, observation, 1, 1.0, _observation(1.0, 0.0)),
        _transition(agent, observation, 2, -1.0, _observation(1.0, 0.0)),
    ]
    before = [parameter.detach().clone() for parameter in agent.predictor.parameters()]
    loss = agent.update(batch)
    after = list(agent.predictor.parameters())

    assert np.isfinite(loss)
    # The prediction half actually stepped the predictor's weights.
    assert any(
        not torch.allclose(old, new.detach()) for old, new in zip(before, after)
    )


# ---------------------------------------------------------------------------
# Save and load.
# ---------------------------------------------------------------------------


def test_save_load_round_trip_carries_depth_and_width(tmp_path):
    observation = _observation(4.0, 0.5, neighbours=[(2.0, 0.5, 0.0, 0.0)])
    agent = RglAgent(
        actions=6,
        seed=1,
        action_table=np.arange(12, dtype=np.float32).reshape(6, 2),
        **TINY,
    )
    for _ in range(5):
        agent.observe(observation, 2, 1.0, _observation(1.0, 1.0), False)

    path = tmp_path / "agent.pt"
    agent.save(path)

    document = torch.load(path, map_location="cpu")
    assert document["algorithm"] == "rgl"
    assert document["format"] == "robotsnap-social-agent-1"
    assert document["config"]["depth"] == 2
    assert document["config"]["width"] == 2

    reloaded = RglAgent.load(path)
    assert isinstance(reloaded, RglAgent)
    assert (reloaded.depth, reloaded.width) == (2, 2)
    assert reloaded.act(observation, greedy=True) == agent.act(observation, greedy=True)

    rebuilt = RglAgent._from_document(document)
    assert rebuilt.act(observation, greedy=True) == agent.act(observation, greedy=True)


def test_a_checkpoint_of_another_algorithm_is_refused(tmp_path):
    from robotsnap.models.cadrl import CadrlAgent

    path = tmp_path / "cadrl.pt"
    CadrlAgent(actions=3, hidden=8).save(path)
    with pytest.raises(ValueError, match="not a"):
        RglAgent.load(path)


# ---------------------------------------------------------------------------
# The catalogue's sizing hook and the environment.
# ---------------------------------------------------------------------------


def test_the_rgl_environment_names_its_own_reward_constants():
    env = _env()
    assert RglEnv.reward_parameters is RGL_REWARD_PARAMETERS
    assert env.social_reward.parameters() == RGL_REWARD_PARAMETERS
    assert ENVIRONMENT is RglEnv and AGENT is RglAgent
    assert RglAgent.name == "rgl"
    assert rgl_class() is not None


def test_make_agent_and_load_agent_accept_rgl(monkeypatch, tmp_path):
    """The registration line the catalogue owner adds is what wires these up."""
    import robotsnap.models as catalogue

    monkeypatch.setitem(catalogue.METHODS, "rgl", "RGL")
    monkeypatch.setitem(catalogue._METHOD_MODULES, "rgl", "robotsnap.models.rgl")
    monkeypatch.setattr(catalogue, "_REGISTRY", None)

    class FakeEnv:
        social_actions = SocialActionSpace(speeds=2, headings=4)

    env = FakeEnv()
    agent = catalogue.make_agent("rgl", env, seed=0, **TINY)
    assert isinstance(agent, RglAgent)
    assert agent.actions == len(env.social_actions)
    assert agent.ego_features == EGO_SIZE + GOAL_SIZE
    assert agent.neighbour_features == NEIGHBOUR_SIZE
    assert (agent.depth, agent.width) == (2, 2)

    path = tmp_path / "rgl.pt"
    agent.save(path)
    reloaded = catalogue.load_agent(path)
    assert isinstance(reloaded, RglAgent)
    assert reloaded.actions == agent.actions
