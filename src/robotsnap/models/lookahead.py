"""The shared value-based core of CADRL and SARL: ``V(s)`` and one-step lookahead.

Chen et al. (2017) and Chen et al. (2019) do not learn a state-action value.
They learn the value of a *state*, ``V(s)``, and choose a command by asking
their model what one step under that command would lead to, then taking the
argmax of a collision cost plus the discounted value of the state it reaches -
their Eq. (5)-(7) and Algorithm 1, "Deep V-learning", which SARL inherits
unchanged (:math:`y = r + \\gamma \\hat V(s')`, with no max over the actions).

This module holds the three pieces both methods agree on, so neither repeats a
line of them:

- :func:`state_value_base` - the ``nn.Module`` that turns a state ``(ego,
  neighbours, mask)`` into a single scalar, with the per-neighbour encoder
  shared and the pooling left to the method, exactly as
  :func:`robotsnap.models.social.value_net_base` does it for action values;
- :func:`propagate` - the model the lookahead rolls forward: the robot's exact
  unicycle displacement and the constant-velocity neighbours, in the robot
  frame;
- :class:`LookaheadAgent` - the replay-buffer, target-network, epsilon-greedy TD
  learner that regresses ``V(s)`` and picks its action by one-step lookahead.

Nothing here imports torch at module import: the network class is built on the
first call that needs it, and :class:`LookaheadAgent` pulls the framework in its
constructor only, so reading the catalogue - or importing a method module to see
what it is - stays free of it.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from robotsnap.envs.base import DEFAULT_CONTROL_PERIOD
from robotsnap.models.social import (
    EGO_KEY,
    EGO_SIZE,
    FORMAT,
    GOAL_KEY,
    GOAL_SIZE,
    MASK_KEY,
    NEIGHBOURS_KEY,
    NEIGHBOUR_SIZE,
    _RunningNorm,
    pack_observation,
)

__all__ = [
    "LookaheadAgent",
    "propagate",
    "state_value_base",
]

#: The key the propagated state carries the goal distance *before* propagation
#: under, so :meth:`SocialNavEnv.lookahead_reward` can price progress without
#: the simulator having run the step. The lookahead agent writes it and the
#: environment reads it; keeping the spelling in one documented place is what
#: stops the two from drifting apart.
PREVIOUS_DISTANCE_KEY = "previous_distance_to_goal"

#: Below this angular speed the unicycle's arc degenerates to a straight line
#: and the exact formulas divide by a zero; the limit of both expressions is the
#: straight displacement, which is what the branch returns.
_TURNING_EPSILON = 1e-6


def propagate(
    ego: Any,
    neighbours: Any,
    mask: Any,
    commands: Any,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Roll one batch of states forward one step under every candidate command.

    The model is the one both papers use for the lookahead, written in the robot
    frame. ``ego`` is the packed ``(batch, EGO_SIZE + GOAL_SIZE)`` vector, so its
    first five entries are the robot's own situation and the next five are the
    goal: position error ``x, y``, distance, and the cos/sin of its bearing. A
    command is a ``(linear_x, angular_z)`` pair and ``commands`` holds one row
    per candidate; the returned arrays add an axis for it, shaped
    ``(batch, actions, ...)``.

    The robot moves by its exact unicycle displacement over ``dt`` seconds, read
    in the frame it starts in::

        dx = v/w * sin(w*dt),  dy = v/w * (1 - cos(w*dt))   if |w| > 1e-6
        dx = v*dt,             dy = 0                       otherwise

    and turns by ``dtheta = w*dt``. Everything else is fixed in the world, so its
    relative pose is rotated back by the robot's own turn: a point that was at
    relative position ``p`` is, in the new frame, at ``R(-dtheta) * (p - D)``
    where ``D = (dx, dy)``. A constant-velocity neighbour keeps its world
    velocity, so only the frame changed and its relative velocity is
    ``R(-dtheta) * v``. The goal follows the neighbours' rule - it is fixed in
    the world too - and its derived fields (distance, cos and sin bearing) are
    recomputed from the new relative position.

    The neighbour order is preserved exactly as it came in, nearest first, and
    the mask travels with it: a row whose mask is false stays false and is
    zeroed, so a padded row can never leak a number into the lookahead. The ego
    is rebuilt consistently - the goal position error, the heading error and the
    commanded speeds - because the value network that reads the propagated state
    saw the same layout during training.

    The arithmetic is done in double precision and cast back to the incoming
    dtype, so a float32 observation comes back float32 without losing the
    geometry of the propagation on the way.
    """
    ego = np.asarray(ego)
    neighbours = np.asarray(neighbours)
    keep = np.asarray(mask).astype(bool)
    commands = np.asarray(commands, dtype=np.float64)

    if ego.ndim != 2:
        raise ValueError(f"ego is (batch, features), got shape {tuple(ego.shape)}")
    batch, features = ego.shape
    if features != EGO_SIZE + GOAL_SIZE:
        raise ValueError(
            f"ego is (batch, {EGO_SIZE + GOAL_SIZE}) to carry the goal, got shape "
            f"{tuple(ego.shape)}"
        )
    if neighbours.ndim != 3:
        raise ValueError(
            f"neighbours are (batch, count, features), got shape {tuple(neighbours.shape)}"
        )
    if neighbours.shape[0] != batch:
        raise ValueError(
            f"neighbours carry {neighbours.shape[0]} batch entries but ego carries {batch}"
        )
    if neighbours.shape[2] != NEIGHBOUR_SIZE:
        raise ValueError(
            f"neighbour features are {NEIGHBOUR_SIZE}, got {neighbours.shape[2]}"
        )
    if keep.shape != neighbours.shape[:2]:
        raise ValueError(
            f"the mask is {tuple(neighbours.shape[:2])}, got shape {tuple(keep.shape)}"
        )
    if commands.ndim != 2 or commands.shape[1] != 2:
        raise ValueError(
            f"commands are (actions, 2) as (linear_x, angular_z), got shape "
            f"{tuple(commands.shape)}"
        )

    count = neighbours.shape[1]
    actions = commands.shape[0]
    dt = float(dt)
    if not np.isfinite(dt) or dt < 0.0:
        raise ValueError("dt must be a finite, non-negative number of seconds")

    speed = commands[:, 0]
    turn_rate = commands[:, 1]
    turning = np.abs(turn_rate) > _TURNING_EPSILON
    ratio = np.where(turning, speed / np.where(turning, turn_rate, 1.0), 0.0)
    angle = turn_rate * dt
    dx = np.where(turning, ratio * np.sin(angle), speed * dt)
    dy = np.where(turning, ratio * (1.0 - np.cos(angle)), 0.0)
    cos_turn = np.cos(angle)
    sin_turn = np.sin(angle)

    neighbour_dtype = neighbours.dtype if neighbours.dtype.kind == "f" else np.float64
    work = neighbours.astype(np.float64, copy=False)
    ego_double = ego.astype(np.float64, copy=False)

    # The goal is fixed in the world, so it transforms exactly like a neighbour:
    # subtract the robot's displacement, then rotate the frame back by its turn.
    rel_goal_x = ego_double[:, EGO_SIZE + 0][:, None] - dx[None, :]
    rel_goal_y = ego_double[:, EGO_SIZE + 1][:, None] - dy[None, :]
    goal_x = rel_goal_x * cos_turn[None, :] + rel_goal_y * sin_turn[None, :]
    goal_y = -rel_goal_x * sin_turn[None, :] + rel_goal_y * cos_turn[None, :]

    distance = np.hypot(goal_x, goal_y)
    present = distance > 0.0
    safe_distance = np.where(present, distance, 1.0)
    cos_bearing = np.where(present, goal_x / safe_distance, 1.0)
    sin_bearing = np.where(present, goal_y / safe_distance, 0.0)
    bearing = np.arctan2(goal_y, goal_x)
    heading_error = np.arctan2(np.sin(bearing), np.cos(bearing))

    dtype = ego.dtype if ego.dtype.kind == "f" else np.float64
    ego2 = np.zeros((batch, actions, features), dtype=np.float64)
    ego2[:, :, 0] = goal_x
    ego2[:, :, 1] = goal_y
    ego2[:, :, 2] = heading_error
    ego2[:, :, 3] = speed[None, :]
    ego2[:, :, 4] = turn_rate[None, :]
    ego2[:, :, EGO_SIZE + 0] = goal_x
    ego2[:, :, EGO_SIZE + 1] = goal_y
    ego2[:, :, EGO_SIZE + 2] = distance
    ego2[:, :, EGO_SIZE + 3] = cos_bearing
    ego2[:, :, EGO_SIZE + 4] = sin_bearing

    rel_x = work[:, :, 0][:, :, None] - dx[None, None, :]
    rel_y = work[:, :, 1][:, :, None] - dy[None, None, :]
    new_x = rel_x * cos_turn[None, None, :] + rel_y * sin_turn[None, None, :]
    new_y = -rel_x * sin_turn[None, None, :] + rel_y * cos_turn[None, None, :]
    new_vx = (
        work[:, :, 2][:, :, None] * cos_turn[None, None, :]
        + work[:, :, 3][:, :, None] * sin_turn[None, None, :]
    )
    new_vy = (
        -work[:, :, 2][:, :, None] * sin_turn[None, None, :]
        + work[:, :, 3][:, :, None] * cos_turn[None, None, :]
    )
    rotated = np.stack([new_x, new_y, new_vx, new_vy], axis=-1)
    neighbours2 = np.transpose(rotated, (0, 2, 1, 3))
    mask2 = np.broadcast_to(keep[:, None, :], (batch, actions, count)).copy()
    neighbours2 = np.where(mask2[..., None], neighbours2, 0.0)

    return (
        ego2.astype(dtype, copy=False),
        neighbours2.astype(neighbour_dtype, copy=False),
        mask2,
    )


_STATE_VALUE_BASE: type | None = None


def state_value_base() -> type:
    """The shared state-value network, built on the first call that needs it.

    ``(ego, neighbours, mask)`` in; one scalar ``V(s)`` per state out. The
    per-neighbour encoder is shared over the neighbours and the mask is applied
    before it - a false row is zeroed rather than fed to the encoder - so a
    padded row can never turn into a number the pooling then spreads. The
    pooling itself is the one thing a method replaces, exactly as
    :class:`~robotsnap.models.social.value_net_base` leaves it to CADRL's LSTM
    and SARL's attention.

    There is deliberately no action table here: a state-value net scores a
    state, and the command that reaches it is the lookahead's business, not the
    network's. ``actions`` and ``action_features`` are accepted and ignored so
    the shared factory can hand every method the same arguments.

    Building the class here rather than at module import is what keeps torch out
    of an interpreter that only reads the catalogue.
    """
    global _STATE_VALUE_BASE
    if _STATE_VALUE_BASE is None:
        import torch
        import torch.nn as nn

        class StateValueNet(nn.Module):
            """Ego, neighbours and a mask in; one state value out.

            A subclass replaces :meth:`pool` only. Everything else - the input
            contract, the shared encoder and the head - is the same for every
            method, which is what makes two value functions comparable.
            """

            def __init__(
                self,
                *,
                ego_features: int,
                neighbour_features: int,
                hidden: int = 64,
                actions: int = 0,
                action_features: int = 2,
            ):
                super().__init__()
                if int(ego_features) <= 0 or int(neighbour_features) <= 0:
                    raise ValueError("ego_features and neighbour_features must be positive")
                self.ego_features = int(ego_features)
                self.neighbour_features = int(neighbour_features)
                self.hidden = int(hidden)
                # Unused, and kept only so the shared factory can pass the same
                # arguments to a state-value net as to an action-value one.
                self.actions = int(actions)
                self.action_features = int(action_features)

                self.encoder = nn.Sequential(
                    nn.Linear(self.neighbour_features, self.hidden),
                    nn.ReLU(),
                    nn.Linear(self.hidden, self.hidden),
                    nn.ReLU(),
                )
                self.head = nn.Sequential(
                    nn.Linear(self.ego_features + self.hidden, self.hidden),
                    nn.ReLU(),
                    nn.Linear(self.hidden, 1),
                )

            def _tensor(self, value: Any):
                """``value`` as a float tensor on the module's own device."""
                if not torch.is_tensor(value):
                    value = torch.as_tensor(np.asarray(value))
                reference = next(self.parameters())
                return value.to(device=reference.device, dtype=reference.dtype)

            def forward(self, ego: Any, neighbours: Any, mask: Any):
                """The ``(batch,)`` values of a batch of states.

                ``neighbours`` is ``(batch, count, features)`` and ``mask`` is
                ``(batch, count)``; a row whose mask is false is padding and is
                ignored everywhere, whatever it holds.
                """
                ego = self._tensor(ego)
                neighbours = self._tensor(neighbours)
                if ego.dim() != 2:
                    raise ValueError(f"ego is (batch, features), got shape {tuple(ego.shape)}")
                if neighbours.dim() != 3:
                    raise ValueError(
                        f"neighbours are (batch, count, features), got shape "
                        f"{tuple(neighbours.shape)}"
                    )
                batch, count = neighbours.shape[0], neighbours.shape[1]
                if ego.shape[0] != batch or ego.shape[1] != self.ego_features:
                    raise ValueError(
                        f"ego is ({batch}, {self.ego_features}) to match {batch} neighbour row(s)"
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

                safe = torch.where(
                    keep.unsqueeze(-1), neighbours, torch.zeros_like(neighbours)
                )
                pooled = self.pool(self.encoder(safe), keep)
                return self.head(torch.cat([ego, pooled], dim=1)).squeeze(-1)

            def pool(self, encoded, keep):
                """Fold ``(batch, count, hidden)`` neighbours into ``(batch, hidden)``."""
                raise NotImplementedError

        _STATE_VALUE_BASE = StateValueNet
    return _STATE_VALUE_BASE


class LookaheadAgent:
    """A value-based TD agent whose policy is a one-step model lookahead.

    The learner is the one Chen et al. (2017) describe and Chen et al. (2019)
    inherit: a replay buffer so one step of experience is used many times, a
    target network so the regression target does not chase the weights it
    grades, an epsilon that decays so the agent explores widely and then
    settles, and a regression of ``V(s)`` on ``y = r + gamma * V_target(s')``
    with *no max over the actions* - Algorithm 1, line 10 of SARL, which is
    CADRL's Algorithm 1 "Deep V-learning".

    The policy is the model lookahead the value function exists for. From one
    observation, :func:`propagate` rolls every candidate command one step
    forward, and the agent takes::

        argmax_a [ R_hat(S_t, a) + gamma * V(S_hat_{t+1}(a)) ]

    the Eq. (5)-(7) rule of CADRL. The value network is asked once, batched over
    the propagated states; ``R_hat`` is whatever immediate-reward callable the
    caller supplied, or zero when none did - the reward of a social-navigation
    step is the environment's business, so it arrives as a callable rather than
    being guessed here.

    An exploration step is drawn exactly as the shared action-value agent draws
    it: the
    do-nothing action with ``near_stop_probability`` of its own, and a uniform
    index otherwise, because a robot that never tries stopping learns that
    stopping is safe only late.

    The interface is deliberately framework-light, and deliberately identical to
    the shared action-value agent of ``robotsnap.models.social``: an agent reads
    a mapping of numpy arrays, keeps numpy in its buffer, and touches torch only
    in the forward and backward passes. The shared trainer and the ``play``
    loader drive it without knowing which learner they hold. Save and load carry
    the same document ``robotsnap.models.load_agent`` knows how to read.

    ``reward_callable`` is not written to a checkpoint - a function is not a
    number - so a checkpoint reloaded without one plays the argmax of the
    discounted value alone; a study that wants the paper's reward at playback
    time passes the environment method again.
    """

    #: The name the algorithm registers under, and the one a checkpoint carries.
    name = "value"

    #: The state-value network class this agent builds; a subclass names it.
    value_class: Any = None

    def __init__(
        self,
        *,
        ego_features: int = EGO_SIZE + GOAL_SIZE,
        neighbour_features: int = NEIGHBOUR_SIZE,
        actions: int,
        action_table: Any = None,
        hidden: int = 64,
        learning_rate: float = 1e-3,
        gamma: float = 0.95,
        batch_size: int = 32,
        buffer_size: int = 20000,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        target_update: int = 200,
        normalise: bool = True,
        seed: int | None = None,
        device: str = "cpu",
        lookahead_discount: float | None = None,
        reward_callable: Callable[[Any, int], float] | None = None,
        **extras: Any,
    ):
        import torch

        self.ego_features = int(ego_features)
        self.neighbour_features = int(neighbour_features)
        self.actions = int(actions)
        if self.actions <= 0:
            raise ValueError("an agent needs at least one action")
        self.hidden = int(hidden)
        self.learning_rate = float(learning_rate)
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.buffer_size = int(buffer_size)
        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay = float(epsilon_decay)
        self.target_update = max(1, int(target_update))
        self.normalise = bool(normalise)
        self.seed = seed
        self.device = str(device)
        self.lookahead_discount = (
            None if lookahead_discount is None else float(lookahead_discount)
        )
        self.reward_callable = reward_callable

        # The shared runner hands every method the same options. The three a
        # lookahead agent understands are read here; anything else a caller
        # passes is ignored rather than raised, so a run that names an option of
        # a sibling method still builds.
        self.near_stop_probability = float(extras.pop("near_stop_probability", 0.1))
        self.control_period = float(extras.pop("control_period", DEFAULT_CONTROL_PERIOD))
        self.action_features = int(extras.pop("action_features", 2))
        del extras

        if action_table is None:
            # A table that is not the commands of a real action set still has to
            # say which action is which, or every candidate would look the same
            # to the lookahead; an index table does that for a synthetic model.
            table = np.zeros((self.actions, 2), dtype=np.float32)
            table[:, 0] = np.arange(self.actions)
        else:
            table = np.asarray(action_table, dtype=np.float32)
        table = table.reshape(self.actions, 2)
        self.commands = table.astype(np.float64)

        self.online = self._build_model().to(self.device)
        self.target = self._build_model().to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimiser = torch.optim.Adam(self.online.parameters(), lr=self.learning_rate)

        self.buffer: deque[tuple[Any, ...]] = deque(maxlen=self.buffer_size)
        self.epsilon = self.epsilon_start
        self.steps = 0
        self.rng = np.random.default_rng(seed)
        self.norm = (
            _RunningNorm(self.ego_features, self.neighbour_features)
            if self.normalise
            else None
        )

    # -- the model, supplied by the subclass --------------------------------

    def _build_model(self):
        """A fresh state-value network for this agent's shape."""
        value_class = self.value_class
        if value_class is None:
            raise NotImplementedError("a lookahead agent needs a value_class")
        return value_class(
            ego_features=self.ego_features,
            neighbour_features=self.neighbour_features,
            hidden=self.hidden,
        )

    def _pack(self, observation: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The ``(ego, neighbours, mask)`` triple the network takes."""
        return pack_observation(observation)

    def set_action_table(self, table: Any) -> None:
        """Replace the candidate commands of the lookahead."""
        values = np.asarray(table, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError(
                f"an action table is (actions, 2), got shape {tuple(values.shape)}"
            )
        if values.shape[0] != self.actions:
            raise ValueError(
                f"the table holds {values.shape[0]} actions but the agent was built "
                f"for {self.actions}"
            )
        self.commands = values

    # -- acting and learning ----------------------------------------------

    def act(self, observation: Any, greedy: bool = False) -> int:
        """The index of the command to take from one observation.

        Not greedy draws an exploration move with ``epsilon`` probability, the
        do-nothing action with ``near_stop_probability`` of its own and a uniform
        index otherwise. Greedy - and any other explore draw - plays
        ``argmax_a [ R_hat + gamma * V(propagated_a) ]``, the one-step lookahead
        of Eq. (7) of CADRL: one model rollout per candidate and one batched
        forward pass over the states they reach.
        """
        if not greedy and float(self.rng.random()) < self.epsilon:
            if float(self.rng.random()) < self.near_stop_probability:
                return 0
            return int(self.rng.integers(self.actions))
        return self._lookahead_action(observation)

    def _lookahead_action(self, observation: Any) -> int:
        """The greedy action, by one-step lookahead over every candidate."""
        import torch

        ego, neighbours, mask = self._pack(observation)
        previous_distance = self._goal_distance(ego)
        ego2, neighbours2, mask2 = propagate(
            ego[None], neighbours[None], mask[None], self.commands, self.control_period
        )

        view_ego, view_neighbours = ego2[0], neighbours2[0]
        if self.norm is not None:
            view_ego, view_neighbours = self.norm.look(view_ego, view_neighbours)
        with torch.no_grad():
            values = self.online(
                torch.as_tensor(view_ego),
                torch.as_tensor(view_neighbours),
                torch.as_tensor(mask2[0]).bool(),
            )
        values = np.asarray(values.detach().cpu().numpy(), dtype=np.float64).reshape(-1)

        if self.reward_callable is None:
            immediate = np.zeros(self.actions, dtype=np.float64)
        else:
            immediate = np.asarray(
                [
                    float(
                        self.reward_callable(
                            self.propagated_state(
                                ego2[0, index],
                                neighbours2[0, index],
                                mask2[0, index],
                                previous_distance,
                            ),
                            index,
                        )
                    )
                    for index in range(self.actions)
                ],
                dtype=np.float64,
            )

        discount = self.gamma if self.lookahead_discount is None else self.lookahead_discount
        score = immediate + discount * values
        return int(np.argmax(score))

    def observe(
        self,
        observation: Any,
        action: int,
        reward: float,
        next_observation: Any,
        done: bool,
    ) -> None:
        """Record one transition and take the next epsilon step."""
        current = [entry.copy() for entry in self._pack(observation)]
        following = [entry.copy() for entry in self._pack(next_observation)]
        transition = (
            *current,
            int(action),
            float(reward),
            *following,
            bool(done),
        )
        self.buffer.append(transition)
        if self.norm is not None:
            self.norm.observe(current[0], current[1], current[2])
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def learn(self) -> float | None:
        """One gradient step on a sampled batch, or ``None`` while the buffer fills."""
        if len(self.buffer) < self.batch_size:
            return None
        indices = self.rng.integers(0, len(self.buffer), size=self.batch_size)
        return self.update([self.buffer[int(index)] for index in indices])

    def update(self, batch: Sequence[tuple[Any, ...]]) -> float:
        """One gradient step of the value regression on the transitions ``batch``.

        Each transition is the tuple :meth:`observe` writes:
        ``(ego, neighbours, mask, action, reward, next_ego, next_neighbours,
        next_mask, done)``. The target is ``y = r + gamma * (1 - done) *
        V_target(s')`` - the TD target of SARL's Algorithm 1 line 10, with no
        maximum over the candidate actions, which is what separates this learner
        from one that regresses an action value. The target network is refreshed
        every ``target_update`` steps.
        """
        import torch
        import torch.nn.functional as functional

        transitions = list(batch)
        if not transitions:
            raise ValueError("an update needs at least one transition")

        def stacked(index: int) -> np.ndarray:
            return np.stack([np.asarray(row[index]) for row in transitions])

        ego, neighbours, mask = stacked(0), stacked(1), stacked(2)
        rewards = np.asarray([row[4] for row in transitions], dtype=np.float32)
        next_ego, next_neighbours, next_mask = stacked(5), stacked(6), stacked(7)
        dones = np.asarray([row[8] for row in transitions], dtype=np.float32)
        if self.norm is not None:
            ego, neighbours = self.norm.look(ego, neighbours)
            next_ego, next_neighbours = self.norm.look(next_ego, next_neighbours)

        device = next(self.online.parameters()).device

        def to_tensor(value: Any, dtype=None):
            return torch.as_tensor(value, dtype=dtype, device=device)

        ego_t = to_tensor(ego, torch.float32)
        neighbours_t = to_tensor(neighbours, torch.float32)
        mask_t = to_tensor(mask, torch.bool)
        next_ego_t = to_tensor(next_ego, torch.float32)
        next_neighbours_t = to_tensor(next_neighbours, torch.float32)
        next_mask_t = to_tensor(next_mask, torch.bool)
        rewards_t = to_tensor(rewards, torch.float32)
        dones_t = to_tensor(dones, torch.float32)

        self.online.train()
        values = self.online(ego_t, neighbours_t, mask_t)
        with torch.no_grad():
            future = self.target(next_ego_t, next_neighbours_t, next_mask_t)
            target = rewards_t + self.gamma * (1.0 - dones_t) * future
        loss = functional.mse_loss(values, target)

        self.optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimiser.step()

        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())

    # -- the propagated state the reward callable reads ---------------------

    @staticmethod
    def _goal_distance(ego: Any) -> float | None:
        """The distance to the goal of one packed ego vector."""
        vector = np.asarray(ego, dtype=np.float64).reshape(-1)
        if vector.size < EGO_SIZE + 2:
            return None
        return float(np.hypot(vector[EGO_SIZE + 0], vector[EGO_SIZE + 1]))

    @staticmethod
    def propagated_state(
        ego: Any,
        neighbours: Any,
        mask: Any,
        previous_distance: float | None,
    ) -> dict[str, Any]:
        """One propagated state, as the mapping a reward callable reads.

        The propagated ``ego`` here is the full packed vector, so the goal's own
        row is the tail of it; it is split back out into the structured layout
        the environment's reward expects, and the goal distance before the step
        travels under :data:`PREVIOUS_DISTANCE_KEY` because progress is the
        difference between the two and only the caller knows the first.
        """
        vector = np.asarray(ego, dtype=np.float64).reshape(-1)
        return {
            EGO_KEY: vector[:EGO_SIZE].copy(),
            GOAL_KEY: vector[EGO_SIZE : EGO_SIZE + GOAL_SIZE].copy(),
            NEIGHBOURS_KEY: np.asarray(neighbours, dtype=np.float64).copy(),
            MASK_KEY: np.asarray(mask).astype(bool).copy(),
            PREVIOUS_DISTANCE_KEY: previous_distance,
        }

    # -- carrying a trained agent ------------------------------------------

    def config(self) -> dict[str, Any]:
        """The constructor arguments this agent was built with."""
        return {
            "ego_features": self.ego_features,
            "neighbour_features": self.neighbour_features,
            "actions": self.actions,
            "action_table": self.commands.astype(np.float32).tolist(),
            "hidden": self.hidden,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "batch_size": self.batch_size,
            "buffer_size": self.buffer_size,
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay": self.epsilon_decay,
            "target_update": self.target_update,
            "normalise": self.normalise,
            "seed": self.seed,
            "device": self.device,
            "lookahead_discount": self.lookahead_discount,
            "near_stop_probability": self.near_stop_probability,
            "control_period": self.control_period,
        }

    def save(self, path: str | Path) -> str:
        """Write the weights, the configuration and the statistics to ``path``."""
        import torch

        document = {
            "format": FORMAT,
            "algorithm": self.name,
            "config": self.config(),
            "state_dict": self.online.state_dict(),
            "epsilon": float(self.epsilon),
            "steps": int(self.steps),
            "normaliser": None if self.norm is None else self.norm.state(),
        }
        torch.save(document, path)
        return str(path)

    @classmethod
    def load(cls, path: str | Path, *, device: str | None = None) -> "LookaheadAgent":
        """Rebuild an agent of this class from the checkpoint at ``path``."""
        import torch

        document = torch.load(Path(path), map_location="cpu")
        if not isinstance(document, Mapping) or document.get("format") != FORMAT:
            raise ValueError(f"{path} is not a {FORMAT} checkpoint")
        if document.get("algorithm") != cls.name:
            raise ValueError(
                f"{path} holds a {document.get('algorithm')!r} agent, not a {cls.name!r} one"
            )
        return cls._from_document(document, device=device)

    @classmethod
    def _from_document(cls, document: Mapping[str, Any], *, device: str | None = None):
        config = dict(document["config"])
        if device is not None:
            config["device"] = str(device)
        agent = cls(**config)
        state = document["state_dict"]
        agent.online.load_state_dict(state)
        agent.target.load_state_dict(state)
        agent.online.eval()
        agent.target.eval()
        agent.epsilon = float(document.get("epsilon", agent.epsilon))
        agent.steps = int(document.get("steps", 0))
        if agent.norm is not None and document.get("normaliser"):
            agent.norm = _RunningNorm.from_state(document["normaliser"])
        return agent
