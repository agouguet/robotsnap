"""RGL, whole: relational graph learning for crowd navigation.

Chen, Y. F., Hu, S., Nikdel, P., Mori, G. and Savva, M. (2020), "Relational
Graph Learning for Crowd Navigation", IROS 2020 (arXiv:1909.13165), and its
official implementation, ``ChanganVR/RelationalGraphLearning``. The learner is
the value-based family CADRL and SARL belong to - a state value ``V(s)`` trained
by temporal difference with an experience replay, a duplicated target network
and an epsilon-greedy behaviour policy, where the TD target has *no maximum over
the actions*::

    y_i = r_i + gamma^(dt * v_pref) * V_hat(s_{i+1})          (Algorithm 1, line 11)

What RGL adds is a state-action lookahead built on two learned models. It learns
a value network ``fV`` and, separately, a state-prediction network ``fP`` - the
paper is explicit that the two use separate graph models - and picks its command
by searching a tree of imagined futures::

    a_t = argmax_a [ R_hat(S_t, a, fP(S_t, a)) + gamma^(dt*v_pref) * V^d(fP(S_t, a)) ]

where ``V^d`` is the value of a depth-``d`` lookahead over ``fP``: the search
keeps the ``w`` best actions at each node (the paper's *action-space clipping*)
and reads ``fV`` once the depth is reached. The paper's reported configuration
is ``d = 2`` and ``w = 2`` ("RGL"), against the ``d = 1`` one-step variant
("MP-RGL-Onestep"); both are the defaults exposed here.

The two networks reason over a *graph* rather than a list. Every agent - the
robot and each masked neighbour - is a node, encoded by a shared MLP ``f_h``;
the directed edges carry features computed by a shared MLP ``f_r`` from the two
endpoints' latent features. Each node then aggregates its incoming neighbours by
a relational convolution, ``message = f_m([h_i, h_j, e_ij])``, weighted by an
attention head ``W_a``, and two such layers are stacked with a latent width of
32. The robot's node after the last layer is what ``fV`` reads. Two layers are
what make the value depend on neighbour-to-neighbour relations: in the second
layer the robot's message comes from a neighbour whose own features already
mixed in the other neighbours, which is the "higher-order interaction" the
paper is after.

What is faithful here
    the relational graph over robot and masked neighbours, two layers of width
    32; the separate ``fP`` and ``fV`` networks; the TD target with no max; the
    depth- and width-clipped lookahead; the replay buffer, target network and
    epsilon-greedy exploration, inherited from
    :class:`~robotsnap.models.lookahead.LookaheadAgent`.

What is not
    the paper initialises ``fP`` and ``fV`` by imitation learning from ORCA
    demonstrations before the RL phase, because the reward is sparse. No ORCA
    dataset is available in this repository, so that phase is *not* reproduced
    and is documented rather than faked. As a partial substitute, ``fP`` can be
    warm-started against the analytic one-step model
    :func:`~robotsnap.models.lookahead.propagate` with
    ``predictor_warm_start``, which is an exact, if less rich, teacher. The
    official code also computes the robot's next state with the kinematic model
    and learns only the humans' motion (with a linear fallback); ``fP`` here
    learns the whole next state, which is a documented simplification. Finally,
    the official adjacency is an embedded-Gaussian softmax over the latent
    features; the attention head ``W_a`` here is an MLP over the same pairwise
    features plus an explicit edge feature ``f_r``, a relational variant that
    keeps the property that matters - information flows between neighbours, not
    only from each neighbour to the robot.

Reference: Chen, Y. F., Hu, S., Nikdel, P., Mori, G. and Savva, M. (2020),
"Relational Graph Learning for Crowd Navigation", IROS 2020.

RGL navigates the same task CADRL and SARL do, so
:data:`RGL_REWARD_PARAMETERS` repeats the shared social cost of the CADRL
lineage (Chen et al. 2017, inherited by Chen et al. 2019 and by RGL). The
constants are defined here rather than imported from ``cadrl.py`` so a study can
reshape RGL without touching the shared task or its siblings.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from robotsnap.models.lookahead import LookaheadAgent, propagate, state_value_base
from robotsnap.models.social import (
    EGO_SIZE,
    FORMAT,
    GOAL_SIZE,
    NEIGHBOUR_SIZE,
    SocialNavEnv,
    _RunningNorm,
)

__all__ = [
    "AGENT",
    "ENVIRONMENT",
    "RGL_DEPTH",
    "RGL_LATENT",
    "RGL_LAYERS",
    "RGL_MLP_HIDDEN",
    "RGL_REWARD_PARAMETERS",
    "RGL_VALUE_HIDDEN",
    "RGL_WIDTH",
    "RglAgent",
    "RglEnv",
    "rgl_class",
    "rgl_predictor",
    "rgl_value",
]

#: The social cost of the CADRL lineage (Chen et al. 2017; Chen et al. 2019),
#: which RGL (Chen et al. 2020) inherits. The values are the shared defaults of
#: :class:`~robotsnap.models.social.SocialReward`.
RGL_REWARD_PARAMETERS: dict[str, float] = {
    "progress_weight": 1.0,
    "collision_penalty": 10.0,
    "goal_bonus": 10.0,
    "time_penalty": 0.1,
    "comfort_distance": 1.0,
    "personal_space_weight": 1.0,
    "too_close_distance": 0.5,
    "too_close_penalty": 5.0,
}

#: The relational graph: two convolutional layers of latent width 32 (the
#: paper's ``num_layer = 2`` and ``X_dim = 32``).
RGL_LAYERS = 2
RGL_LATENT = 32
#: The hidden widths of the edge, node and message MLPs ``f_r``, ``f_h``, ``f_m``
#: - the paper's ``wr_dims = wh_dims = [64, 32]``.
RGL_MLP_HIDDEN = (64, 32)
#: The hidden widths of the value head ``f_v``, the paper's
#: ``value_network_dims`` ending in the scalar output.
RGL_VALUE_HIDDEN = (150, 100, 100)
#: The hidden widths of the state predictor's heads.
RGL_PREDICTOR_HIDDEN = (64, 32)
#: The paper's best planning configuration: depth 2, width 2, action clipping on.
RGL_DEPTH = 2
RGL_WIDTH = 2
#: How many neighbour rows the synthetic warm-start batch reserves. The graph is
#: indifferent to the count, so this is only a teacher's batch shape.
RGL_WARM_START_NEIGHBOURS = 4


class RglEnv(SocialNavEnv):
    """The RGL task: the shared social task with the CADRL-lineage cost.

    RGL navigates the same crowd, with the same eighty commands and the same
    structured observation as CADRL and SARL, so there is nothing new here but
    the named shaping. The learning method lives in :class:`RglAgent`.
    """

    reward_parameters = RGL_REWARD_PARAMETERS


def _mlp(input_dim: int, dims: Sequence[int], last_relu: bool = False):
    """The official ``helpers.mlp``: linear layers with a ReLU between them."""
    import torch.nn as nn

    widths = [int(input_dim)] + [int(width) for width in dims]
    layers: list[Any] = []
    for index in range(len(widths) - 1):
        layers.append(nn.Linear(widths[index], widths[index + 1]))
        if index != len(widths) - 2 or last_relu:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


_GRAPH_CLASSES: tuple[type, type] | None = None


def graph_classes() -> tuple[type, type]:
    """The relational layer and the relational graph, built on the first call.

    Kept lazy for the same reason every other network in the catalogue is: a
    caller that only wants to read what RGL is must not pay for PyTorch.
    """
    global _GRAPH_CLASSES
    if _GRAPH_CLASSES is None:
        import torch
        import torch.nn as nn

        class RelationalLayer(nn.Module):
            """One relational convolution over the agent graph.

            Every ordered pair ``(i, j)`` gets an edge feature ``e_ij =
            f_r([h_i, h_j])`` and a message ``f_m([h_i, h_j, e_ij])``. Node
            ``i`` then aggregates the messages of its incoming neighbours with
            an attention weight ``W_a([h_i, h_j, e_ij])``. The attention is a
            softmax over the valid neighbours only: a masked key is filled with
            minus infinity before the softmax and contributes nothing
            afterwards, so padding cannot leak a value wherever the padded row
            sits. The sum is over the set of neighbours, which is why the layer
            is permutation-equivariant and a state's value cannot depend on the
            row order.
            """

            def __init__(self, latent: int, hidden: Sequence[int]):
                super().__init__()
                self.edge = _mlp(2 * latent, [*hidden, latent], last_relu=True)
                self.message = _mlp(3 * latent, [*hidden, latent], last_relu=True)
                self.attention = _mlp(3 * latent, [*hidden, 1])
                self.update = nn.Linear(latent, latent)

            def forward(self, nodes, node_mask):
                batch, count, latent = nodes.shape
                hi = nodes.unsqueeze(2).expand(batch, count, count, latent)
                hj = nodes.unsqueeze(1).expand(batch, count, count, latent)
                edge = self.edge(torch.cat([hi, hj], dim=-1))
                joint = torch.cat([hi, hj, edge], dim=-1)
                message = self.message(joint)
                logits = self.attention(joint).squeeze(-1)
                keys = node_mask.unsqueeze(1).expand(batch, count, count)
                logits = logits.masked_fill(~keys, float("-inf"))
                weights = torch.softmax(logits, dim=-1)
                # Rows that are not valid keys get a zero weight; every query
                # has at least itself as a valid key, so no row is all-masked.
                weights = weights.masked_fill(~keys, 0.0)
                aggregated = (weights.unsqueeze(-1) * message).sum(dim=2)
                # A residual, the official ``skip_connection = True``: a layer
                # refines a node instead of replacing what it already knew.
                updated = torch.relu(self.update(aggregated) + nodes)
                return updated * node_mask.unsqueeze(-1).to(updated.dtype)

        class RelationalGraph(nn.Module):
            """Encode ``(ego, neighbours, mask)`` into one latent per agent.

            The robot is node 0 and is always valid; each masked neighbour is a
            node after it. The returned matrix has shape ``(batch, 1 + count,
            latent)`` and the returned mask the same ``(batch, 1 + count)``
            boolean layout, so a caller reads the robot's row at index 0.
            """

            def __init__(
                self,
                *,
                ego_features: int,
                neighbour_features: int,
                latent: int,
                layers: int,
                hidden: Sequence[int],
            ):
                super().__init__()
                self.latent = int(latent)
                self.ego_encoder = _mlp(
                    int(ego_features), [*hidden, self.latent], last_relu=True
                )
                self.neighbour_encoder = _mlp(
                    int(neighbour_features), [*hidden, self.latent], last_relu=True
                )
                self.layers = nn.ModuleList(
                    [RelationalLayer(self.latent, hidden) for _ in range(int(layers))]
                )

            def forward(self, ego, neighbours, mask):
                keep = mask.bool()
                safe = torch.where(
                    keep.unsqueeze(-1), neighbours, torch.zeros_like(neighbours)
                )
                robot = self.ego_encoder(ego).unsqueeze(1)
                rows = self.neighbour_encoder(safe)
                nodes = torch.cat([robot, rows], dim=1)
                node_mask = torch.cat([torch.ones_like(keep[:, :1]), keep], dim=1)
                for layer in self.layers:
                    nodes = layer(nodes, node_mask)
                return nodes, node_mask

        _GRAPH_CLASSES = (RelationalLayer, RelationalGraph)
    return _GRAPH_CLASSES


_VALUE_CLASS: type | None = None


def rgl_class() -> type:
    """RGL's state-value network, built once on the first call.

    ``V(s)`` is the robot's node after the relational graph, read by the ``f_v``
    MLP. There is no action input: the candidate command enters the lookahead,
    not the value network.
    """
    global _VALUE_CLASS
    if _VALUE_CLASS is None:
        base = state_value_base()
        _, RelationalGraph = graph_classes()

        class RglValue(base):
            """RGL's state value: a two-layer relational graph over the agents.

            The robot and every masked neighbour are nodes; the value is ``f_v``
            of the robot's node after the relational convolutions. A false row
            is zeroed before the encoders and masked out of every attention, so
            it can never reach the value, and the aggregation is a sum over the
            neighbour set, so the value does not depend on the row order.
            """

            def __init__(
                self,
                *,
                latent: int = RGL_LATENT,
                layers: int = RGL_LAYERS,
                mlp_hidden: Sequence[int] = RGL_MLP_HIDDEN,
                value_hidden: Sequence[int] = RGL_VALUE_HIDDEN,
                **kwargs: Any,
            ):
                super().__init__(**kwargs)
                self.latent = int(latent)
                self.graph_layers = int(layers)
                if self.latent <= 0:
                    raise ValueError("the graph latent width must be positive")
                if self.graph_layers <= 0:
                    raise ValueError("the graph needs at least one layer")
                self.graph = RelationalGraph(
                    ego_features=self.ego_features,
                    neighbour_features=self.neighbour_features,
                    latent=self.latent,
                    layers=self.graph_layers,
                    hidden=tuple(mlp_hidden),
                )
                self.value_head = _mlp(self.latent, [*tuple(value_hidden), 1])

            def _tensor(self, value: Any):
                import torch

                if not torch.is_tensor(value):
                    value = torch.as_tensor(np.asarray(value))
                reference = next(self.parameters())
                return value.to(device=reference.device, dtype=reference.dtype)

            def forward(self, ego: Any, neighbours: Any, mask: Any):
                """The ``(batch,)`` values of a batch of states."""
                import torch

                ego = self._tensor(ego)
                neighbours = self._tensor(neighbours)
                if ego.dim() != 2:
                    raise ValueError(
                        f"ego is (batch, features), got shape {tuple(ego.shape)}"
                    )
                if neighbours.dim() != 3:
                    raise ValueError(
                        f"neighbours are (batch, count, features), got shape "
                        f"{tuple(neighbours.shape)}"
                    )
                batch, count = neighbours.shape[0], neighbours.shape[1]
                if ego.shape[0] != batch or ego.shape[1] != self.ego_features:
                    raise ValueError(
                        f"ego is ({batch}, {self.ego_features}) to match {batch} "
                        "neighbour row(s)"
                    )
                if neighbours.shape[2] != self.neighbour_features:
                    raise ValueError(
                        f"neighbour features are {self.neighbour_features}, got "
                        f"{neighbours.shape[2]}"
                    )
                keep = self._tensor(mask).bool()
                if tuple(keep.shape) != (batch, count):
                    raise ValueError(
                        f"the mask is ({batch}, {count}), got shape {tuple(keep.shape)}"
                    )
                nodes, _ = self.graph(ego, neighbours, keep)
                return self.value_head(nodes[:, 0, :]).squeeze(-1)

            def pool(self, encoded, keep):
                """A masked mean, so the inherited contract still has an answer.

                The relational value net replaces :meth:`forward` with the graph
                and never calls this; it is here so a caller that pokes the base
                class finds something meaningful rather than an exception.
                """
                weights = keep.to(encoded.dtype).unsqueeze(-1)
                return (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp(
                    min=1.0
                )

        _VALUE_CLASS = RglValue
    return _VALUE_CLASS


def __getattr__(name: str):
    """``RglValue`` and ``RglPredictor`` on request, without torch at import."""
    if name == "RglValue":
        return rgl_class()
    if name == "RglPredictor":
        return rgl_predictor_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def rgl_value(**kwargs: Any):
    """Build a fresh RGL state-value network.

    Keyword arguments are the constructor's: ``ego_features``,
    ``neighbour_features``, ``hidden`` plus ``latent``, ``layers``,
    ``mlp_hidden`` and ``value_hidden``. Calling this pulls PyTorch in;
    importing the module does not.
    """
    return rgl_class()(**kwargs)


_PREDICTOR_CLASS: type | None = None


def rgl_predictor_class() -> type:
    """RGL's state predictor ``fP``, built once on the first call."""
    global _PREDICTOR_CLASS
    if _PREDICTOR_CLASS is None:
        import torch
        import torch.nn as nn

        _, RelationalGraph = graph_classes()

        class RglPredictor(nn.Module):
            """The state predictor ``fP(s, a) -> s'``, a second graph model.

            A separate relational graph - the paper is explicit that ``fP`` and
            ``fV`` do not share one - encodes the current state, the candidate
            command ``(v, w)`` is embedded by an MLP, and two residual heads
            produce the robot's next packed state (goal position error, heading
            error, body velocity and the goal's derived fields) and every
            neighbour's next ``(dx, dy, vx, vy)`` row. The mask is copied, never
            predicted: whether a neighbour exists is a fact of the observation,
            not something to regress. Rows that are masked are zeroed, so the
            lookahead can never read a value out of padding.

            The training target is the transition the buffer actually observed,
            ``S_{i+1}`` (Algorithm 1, line 14) - exactly the supervised line the
            paper writes. The analytic
            :func:`~robotsnap.models.lookahead.propagate` never replaces an
            observed transition; it is the teacher of the optional warm start
            (:meth:`RglAgent.warm_start_predictor`), the documented substitute
            for the paper's ORCA imitation phase.
            """

            def __init__(
                self,
                *,
                ego_features: int,
                neighbour_features: int,
                action_features: int = 2,
                latent: int = RGL_LATENT,
                layers: int = RGL_LAYERS,
                mlp_hidden: Sequence[int] = RGL_MLP_HIDDEN,
                hidden: Sequence[int] = RGL_PREDICTOR_HIDDEN,
                **extras: Any,
            ):
                super().__init__()
                del extras
                self.ego_features = int(ego_features)
                self.neighbour_features = int(neighbour_features)
                self.action_features = int(action_features)
                self.latent = int(latent)
                self.graph = RelationalGraph(
                    ego_features=self.ego_features,
                    neighbour_features=self.neighbour_features,
                    latent=self.latent,
                    layers=int(layers),
                    hidden=tuple(mlp_hidden),
                )
                self.action_encoder = _mlp(
                    self.action_features, [*tuple(hidden), self.latent], last_relu=True
                )
                self.ego_head = _mlp(
                    2 * self.latent, [*tuple(hidden), self.ego_features]
                )
                self.neighbour_head = _mlp(
                    3 * self.latent, [*tuple(hidden), self.neighbour_features]
                )

            def _tensor(self, value: Any):
                import torch

                if not torch.is_tensor(value):
                    value = torch.as_tensor(np.asarray(value))
                reference = next(self.parameters())
                return value.to(device=reference.device)

            def forward(self, ego: Any, neighbours: Any, mask: Any, actions: Any):
                """``(ego2, neighbours2, mask2)`` for a batch of ``(S, a)``.

                ``actions`` is a ``(batch, action_features)`` float block - the
                commands themselves, so the network stays indifferent to the
                numbering of an action table.
                """
                import torch

                ego = self._tensor(ego)
                neighbours = self._tensor(neighbours)
                actions = self._tensor(actions).to(dtype=ego.dtype)
                keep = self._tensor(mask).bool()
                batch, count = neighbours.shape[0], neighbours.shape[1]
                nodes, _ = self.graph(ego, neighbours, keep)
                robot = nodes[:, 0, :]
                command = self.action_encoder(actions)
                delta_ego = self.ego_head(torch.cat([robot, command], dim=-1))
                robot_rows = robot.unsqueeze(1).expand(batch, count, self.latent)
                command_rows = command.unsqueeze(1).expand(batch, count, self.latent)
                pair = torch.cat(
                    [nodes[:, 1:, :], robot_rows, command_rows], dim=-1
                )
                delta_neighbours = self.neighbour_head(pair)
                next_ego = ego + delta_ego
                next_neighbours = neighbours + delta_neighbours
                next_neighbours = torch.where(
                    keep.unsqueeze(-1),
                    next_neighbours,
                    torch.zeros_like(next_neighbours),
                )
                return next_ego, next_neighbours, keep

        _PREDICTOR_CLASS = RglPredictor
    return _PREDICTOR_CLASS


def rgl_predictor(**kwargs: Any):
    """Build a fresh RGL state predictor.

    Keyword arguments are ``ego_features``, ``neighbour_features``,
    ``action_features`` plus ``latent``, ``layers``, ``mlp_hidden`` and
    ``hidden``.
    """
    return rgl_predictor_class()(**kwargs)


class RglAgent(LookaheadAgent):
    """RGL's learner: ``fV`` by TD, ``fP`` by regression, and a depth-d lookahead.

    The replay buffer, the target network, the epsilon schedule, the TD target
    ``y = r + gamma * (1 - done) * V_target(s')`` with no maximum over the
    actions, and the outer ``act/observe/learn`` loop are the shared ones of
    :class:`~robotsnap.models.lookahead.LookaheadAgent`. RGL adds:

    - a state predictor ``fP``, trained by the supervised L2 of Algorithm 1
      line 14 on the transitions in the buffer, and used by the lookahead;
    - :meth:`_search`, the depth-``d`` width-``w`` lookahead of Algorithm 1 line
      7: at every node the ``w`` best actions are kept (action-space clipping)
      and the children are expanded through ``fP``; once the depth is reached
      the value network is read. With ``d = 1`` this is the one-step rule CADRL
      and SARL use.

    The value target bootstraps the *target network without lookahead*,
    ``V_target(s')``. That is an honest approximation of the paper's
    ``fV_hat^d(s_{i+1})`` (Algorithm 1, line 11), which asks for the depth-``d``
    lookahead value of the next state; the shared learner regresses the plain
    target value, and the greedy policy is where the lookahead lives. The
    discount is the shared ``lookahead_discount``, by default ``gamma``; the
    paper's ``gamma^(dt * v_pref)`` is that value for a caller who sets it.

    Exploration draws and the environment's reward callable are unchanged from
    the shared agent.
    """

    name = "rgl"

    @property
    def value_class(self) -> type:
        """RGL's ``fV`` network class, built on the first use."""
        return rgl_class()

    def __init__(
        self,
        *,
        depth: int = RGL_DEPTH,
        width: int = RGL_WIDTH,
        prediction_weight: float = 1.0,
        latent: int = RGL_LATENT,
        layers: int = RGL_LAYERS,
        mlp_hidden: Sequence[int] = RGL_MLP_HIDDEN,
        value_hidden: Sequence[int] = RGL_VALUE_HIDDEN,
        predictor_hidden: Sequence[int] = RGL_PREDICTOR_HIDDEN,
        predictor_lr: float | None = None,
        predictor_warm_start: int = 0,
        **kwargs: Any,
    ):
        # Everything the value network is sized by has to exist before the base
        # constructor builds it below.
        self.depth = int(depth)
        self.width = int(width)
        if self.depth < 1:
            raise ValueError("the lookahead depth must be at least one")
        if self.width < 1:
            raise ValueError("the lookahead width must be at least one")
        self.prediction_weight = float(prediction_weight)
        self.latent = int(latent)
        self.layers = int(layers)
        self.mlp_hidden = tuple(int(entry) for entry in mlp_hidden)
        self.value_hidden = tuple(int(entry) for entry in value_hidden)
        self.predictor_hidden = tuple(int(entry) for entry in predictor_hidden)
        self.predictor_lr = None if predictor_lr is None else float(predictor_lr)
        self.predictor_warm_start = int(predictor_warm_start)

        super().__init__(**kwargs)
        import torch

        self.predictor = self._build_predictor().to(self.device)
        self.predictor_optimiser = torch.optim.Adam(
            self.predictor.parameters(),
            lr=self.predictor_lr if self.predictor_lr is not None else self.learning_rate,
        )
        #: What the last greedy search did, for a log line or a test.
        self.last_search: dict[str, Any] = {}
        if self.predictor_warm_start > 0:
            self.warm_start_predictor(self.predictor_warm_start)

    # -- the two models ----------------------------------------------------

    def _build_model(self):
        """RGL's ``fV``, sized by this agent's graph configuration."""
        return self.value_class(
            ego_features=self.ego_features,
            neighbour_features=self.neighbour_features,
            hidden=self.hidden,
            latent=self.latent,
            layers=self.layers,
            mlp_hidden=self.mlp_hidden,
            value_hidden=self.value_hidden,
        )

    def _build_predictor(self):
        """RGL's ``fP``, a second graph model of the same shape."""
        return rgl_predictor(
            ego_features=self.ego_features,
            neighbour_features=self.neighbour_features,
            latent=self.latent,
            layers=self.layers,
            mlp_hidden=self.mlp_hidden,
            hidden=self.predictor_hidden,
        )

    # -- the lookahead -----------------------------------------------------

    def _lookahead_action(self, observation: Any) -> int:
        """The greedy command, by the depth-``d`` width-``w`` lookahead."""
        ego, neighbours, mask = self._pack(observation)
        action, _ = self._search(
            ego,
            neighbours,
            mask,
            self._goal_distance(ego),
            depth=self.depth,
            width=self.width,
        )
        return int(action)

    def _search(
        self,
        ego: np.ndarray,
        neighbours: np.ndarray,
        mask: np.ndarray,
        previous_distance: float | None,
        *,
        depth: int,
        width: int,
    ) -> tuple[int, float]:
        """The best root action and its estimated return, by clipped lookahead.

        A breadth-first expansion of ``fP``: the root expands every command, and
        every node that survives pruning expands every command again while only
        its ``w`` best children are kept. The child that survives the last level
        carries the root action to return. Scores are the discounted return
        estimate ``cumulative_reward + gamma^level * fV(state)``, which is
        ``R_hat + gamma * V`` at its one-step special case, exactly the shared
        agent's convention. The number of nodes expanded per level is recorded
        in :attr:`last_search`.
        """
        gamma = self.gamma if self.lookahead_discount is None else self.lookahead_discount
        # (cumulative reward, ego, neighbours, mask, distance to goal, first action)
        frontier: list[
            tuple[float, np.ndarray, np.ndarray, np.ndarray, float | None, int | None]
        ] = [(0.0, ego, neighbours, mask, previous_distance, None)]
        stats: dict[str, Any] = {"expanded": [], "fP_calls": 0}
        best_action = 0
        best_score = float("-inf")
        children: list[
            tuple[float, np.ndarray, np.ndarray, np.ndarray, float | None, int | None]
        ] = []

        for level in range(1, depth + 1):
            stats["expanded"].append(len(frontier))
            children = []
            for (
                cumulative,
                node_ego,
                node_neighbours,
                node_mask,
                node_distance,
                first,
            ) in frontier:
                next_ego, next_neighbours, next_mask = self._predict_batch(
                    node_ego, node_neighbours, node_mask, range(self.actions)
                )
                stats["fP_calls"] += 1
                for index in range(self.actions):
                    reward = self._immediate(
                        next_ego[index],
                        next_neighbours[index],
                        next_mask[index],
                        node_distance,
                        index,
                    )
                    child_cumulative = cumulative + (gamma ** (level - 1)) * reward
                    child_first = index if first is None else first
                    children.append(
                        (
                            child_cumulative,
                            next_ego[index],
                            next_neighbours[index],
                            next_mask[index],
                            self._goal_distance(next_ego[index]),
                            child_first,
                        )
                    )

            values = self._value_batch(
                np.stack([child[1] for child in children]),
                np.stack([child[2] for child in children]),
                np.stack([child[3] for child in children]),
            )
            scores = np.asarray(
                [
                    children[index][0] + (gamma ** level) * float(values[index])
                    for index in range(len(children))
                ],
                dtype=np.float64,
            )

            if level == depth:
                winner = int(np.argmax(scores))
                best_action = int(children[winner][5])
                best_score = float(scores[winner])
            else:
                kept = min(int(width), len(children))
                order = np.argsort(-scores, kind="stable")[:kept]
                frontier = [children[int(index)] for index in order]

        stats["leaves"] = len(children)
        stats["action"] = best_action
        stats["score"] = best_score
        self.last_search = stats
        return best_action, best_score

    def _immediate(
        self,
        ego: np.ndarray,
        neighbours: np.ndarray,
        mask: np.ndarray,
        previous_distance: float | None,
        action: int,
    ) -> float:
        """``R_hat(S, a, S_hat')`` for one predicted transition."""
        if self.reward_callable is None:
            return 0.0
        propagated = self.propagated_state(ego, neighbours, mask, previous_distance)
        return float(self.reward_callable(propagated, int(action)))

    def _value_batch(
        self, ego: np.ndarray, neighbours: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """``fV`` on a batch of states, with the running normaliser applied."""
        import torch

        view_ego, view_neighbours = ego, neighbours
        if self.norm is not None:
            view_ego, view_neighbours = self.norm.look(ego, neighbours)
        device = next(self.online.parameters()).device
        with torch.no_grad():
            values = self.online(
                torch.as_tensor(view_ego, dtype=torch.float32, device=device),
                torch.as_tensor(view_neighbours, dtype=torch.float32, device=device),
                torch.as_tensor(mask, device=device).bool(),
            )
        return np.asarray(values.detach().cpu().numpy(), dtype=np.float64).reshape(-1)

    def _predict_batch(
        self,
        ego: np.ndarray,
        neighbours: np.ndarray,
        mask: np.ndarray,
        actions: Iterable[int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``fP`` for one state under every action in ``actions``.

        Returns ``(count, ego_features)``, ``(count, neighbours, features)`` and
        ``(count, neighbours)`` arrays: one imagined next state per command. This
        is the one place the lookahead calls the predictor, so a test can count
        the ``fP`` calls of a search by watching this method.
        """
        import torch

        indices = np.asarray(list(actions), dtype=np.int64)
        count = int(indices.size)
        commands = self.commands[indices].astype(np.float32)
        ego = np.asarray(ego, dtype=np.float32).reshape(1, -1)
        neighbours = np.asarray(neighbours, dtype=np.float32)
        mask = np.asarray(mask).astype(bool)
        device = next(self.predictor.parameters()).device
        with torch.no_grad():
            next_ego, next_neighbours, next_mask = self.predictor(
                torch.as_tensor(np.repeat(ego, count, axis=0), device=device),
                torch.as_tensor(
                    np.repeat(neighbours[None], count, axis=0), device=device
                ),
                torch.as_tensor(np.repeat(mask[None], count, axis=0), device=device),
                torch.as_tensor(commands, device=device),
            )
        return (
            np.asarray(next_ego.detach().cpu().numpy(), dtype=np.float64),
            np.asarray(next_neighbours.detach().cpu().numpy(), dtype=np.float64),
            np.asarray(next_mask.detach().cpu().numpy(), dtype=bool),
        )

    def predict_next(self, observation: Any, action: int) -> dict[str, Any]:
        """``fP(S, a)`` as the mapping a reward callable or a caller reads."""
        ego, neighbours, mask = self._pack(observation)
        next_ego, next_neighbours, next_mask = self._predict_batch(
            ego, neighbours, mask, [int(action)]
        )
        return self.propagated_state(
            next_ego[0],
            next_neighbours[0],
            next_mask[0],
            self._goal_distance(ego),
        )

    # -- learning ----------------------------------------------------------

    def update(self, batch: Sequence[tuple[Any, ...]]) -> float:
        """One value step plus one predictor step on the transitions ``batch``.

        The value half is the shared TD regression, ``y = r + gamma * (1 - done)
        * V_target(s')`` with no maximum over the actions. The prediction half is
        the supervised L2 of ``fP(S_i, a_i)`` against the observed ``S_{i+1}``
        (Algorithm 1, line 14), masked so padded rows never enter the loss. The
        returned number is ``value_loss + lambda * prediction_loss``, the total
        the trainer displays.
        """
        import torch

        transitions = list(batch)
        if not transitions:
            raise ValueError("an update needs at least one transition")
        value_loss = super().update(transitions)
        prediction_loss = self._prediction_loss(transitions)

        self.predictor_optimiser.zero_grad()
        prediction_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), 10.0)
        self.predictor_optimiser.step()
        return float(
            value_loss + self.prediction_weight * float(prediction_loss.item())
        )

    def _prediction_loss(self, transitions: Sequence[tuple[Any, ...]]):
        """The masked L2 of ``fP`` against the observed next states."""
        import torch
        import torch.nn.functional as functional

        def stacked(index: int) -> np.ndarray:
            return np.stack([np.asarray(row[index]) for row in transitions])

        ego, neighbours, mask = stacked(0), stacked(1), stacked(2)
        next_ego, next_neighbours, next_mask = stacked(5), stacked(6), stacked(7)
        indices = np.asarray([int(row[3]) for row in transitions], dtype=np.int64)
        commands = self.commands[indices].astype(np.float32)

        device = next(self.predictor.parameters()).device
        predicted_ego, predicted_neighbours, _ = self.predictor(
            torch.as_tensor(ego, dtype=torch.float32, device=device),
            torch.as_tensor(neighbours, dtype=torch.float32, device=device),
            torch.as_tensor(mask, device=device).bool(),
            torch.as_tensor(commands, device=device),
        )
        target_ego = torch.as_tensor(next_ego, dtype=torch.float32, device=device)
        target_neighbours = torch.as_tensor(
            next_neighbours, dtype=torch.float32, device=device
        )
        weights = torch.as_tensor(next_mask, device=device).bool().unsqueeze(-1)
        ego_loss = functional.mse_loss(predicted_ego, target_ego)
        squared = (predicted_neighbours - target_neighbours) ** 2
        float_weights = weights.to(squared.dtype)
        total_weight = float_weights.sum().clamp(min=1.0)
        neighbour_loss = (squared * float_weights).sum() / (
            total_weight * predicted_neighbours.shape[-1]
        )
        return ego_loss + neighbour_loss

    def warm_start_predictor(self, steps: int, batch_size: int = 32) -> float:
        """Fit ``fP`` to :func:`~robotsnap.models.lookahead.propagate`.

        The paper's imitation phase is not reproducible here (no ORCA dataset),
        so the exact one-step model stands in as the teacher: random synthetic
        states and commands are rolled through ``propagate`` and the predictor
        regresses them. This is a documented substitute for the paper's step 1,
        not a reproduction of it. Returns the last batch's prediction loss.
        """
        import torch

        steps = int(steps)
        if steps <= 0:
            raise ValueError("warm_start_predictor needs a positive number of steps")
        rng = np.random.default_rng(self.seed)
        loss = 0.0
        for _ in range(steps):
            transitions = self._synthetic_transitions(int(batch_size), rng)
            batch_loss = self._prediction_loss(transitions)
            self.predictor_optimiser.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), 10.0)
            self.predictor_optimiser.step()
            loss = float(batch_loss.item())
        return loss

    def _synthetic_transitions(
        self, count: int, rng: np.random.Generator
    ) -> list[tuple[Any, ...]]:
        """``count`` random ``(S, a, S')`` rows whose ``S'`` is ``propagate``'s."""
        neighbours = int(RGL_WARM_START_NEIGHBOURS)
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

        indices = rng.integers(0, self.actions, size=count)
        next_ego, next_neighbours, next_mask = propagate(
            ego, rows, mask, self.commands, self.control_period
        )
        # ``propagate`` carries an action axis; each sample takes its own draw.
        return [
            (
                ego[index],
                rows[index],
                mask[index],
                int(indices[index]),
                0.0,
                next_ego[index, int(indices[index])],
                next_neighbours[index, int(indices[index])],
                next_mask[index, int(indices[index])],
                False,
            )
            for index in range(count)
        ]

    # -- carrying a trained agent ------------------------------------------

    @staticmethod
    def size_from(env: Any) -> dict[str, Any]:
        """The observation and action sizes ``make_agent`` reads off ``env``.

        The optional hook ``robotsnap.models.make_agent`` looks for on an agent
        class. RGL's observation is the shared structured one, so this is the
        same sizing every sibling uses; naming it here keeps the agent buildable
        from an environment without repeating the widths at the call site.
        """
        defaults: dict[str, Any] = {
            "ego_features": EGO_SIZE + GOAL_SIZE,
            "neighbour_features": NEIGHBOUR_SIZE,
        }
        space = getattr(env, "social_actions", None)
        if space is not None:
            defaults["actions"] = len(space)
            defaults["action_table"] = np.asarray(space.commands)
        elif hasattr(env, "action_space"):
            defaults["actions"] = int(env.action_space.n)
        return defaults

    def config(self) -> dict[str, Any]:
        """The constructor arguments this agent was built with."""
        document = super().config()
        document.update(
            {
                "depth": self.depth,
                "width": self.width,
                "prediction_weight": self.prediction_weight,
                "latent": self.latent,
                "layers": self.layers,
                "mlp_hidden": list(self.mlp_hidden),
                "value_hidden": list(self.value_hidden),
                "predictor_hidden": list(self.predictor_hidden),
                "predictor_lr": self.predictor_lr,
                "predictor_warm_start": self.predictor_warm_start,
            }
        )
        return document

    def save(self, path: str | Path) -> str:
        """Write both networks, the configuration and the statistics to ``path``."""
        import torch

        document = {
            "format": FORMAT,
            "algorithm": self.name,
            "config": self.config(),
            "state_dict": self.online.state_dict(),
            "predictor_state_dict": self.predictor.state_dict(),
            "epsilon": float(self.epsilon),
            "steps": int(self.steps),
            "normaliser": None if self.norm is None else self.norm.state(),
        }
        torch.save(document, path)
        return str(path)

    @classmethod
    def _from_document(cls, document: Mapping[str, Any], *, device: str | None = None):
        config = dict(document["config"])
        if device is not None:
            config["device"] = str(device)
        agent = cls(**config)
        state = document["state_dict"]
        agent.online.load_state_dict(state)
        agent.target.load_state_dict(state)
        if document.get("predictor_state_dict") is not None:
            agent.predictor.load_state_dict(document["predictor_state_dict"])
        agent.online.eval()
        agent.target.eval()
        agent.predictor.eval()
        agent.epsilon = float(document.get("epsilon", agent.epsilon))
        agent.steps = int(document.get("steps", 0))
        if agent.norm is not None and document.get("normaliser"):
            agent.norm = _RunningNorm.from_state(document["normaliser"])
        return agent


#: What the catalogue registers for ``rgl``: the task and the agent on it.
ENVIRONMENT = RglEnv
AGENT = RglAgent
