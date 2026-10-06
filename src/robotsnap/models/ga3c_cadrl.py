"""GA3C-CADRL: Everett, Chen & How (IROS 2018), arXiv:1805.01956.

The A3C/GA3C successor of CADRL replaces value learning with actor-critic
learning: one shared DNN carries a scalar value head and a policy head.
There is no action lookahead or Q(s, a) head.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from robotsnap.models.social import (
    EGO_SIZE, GOAL_SIZE, NEIGHBOUR_SIZE, FORMAT, SocialNavEnv,
    _RunningNorm, pack_observation,
)

__all__ = [
    "GA3C_CADRL_REWARD_PARAMETERS", "GA3C_CADRL_PAPER_COMMANDS",
    "Ga3cCadrlEnv", "Ga3cAgent", "ga3c_cadrl_class", "ga3c_cadrl_value",
    "ENVIRONMENT", "AGENT",
]

# Eq. (12) of Everett, Chen & How (IROS 2018), arXiv:1805.01956.
# There is no time penalty.
GA3C_CADRL_REWARD_PARAMETERS = {
    "goal_bonus": 1.0, "collision_penalty": -0.25,
    "near_miss_distance": 0.2, "near_miss_base": -0.1,
    "near_miss_slope": 0.05, "time_penalty": 0.0,
}

#: The prose says "six headings at v_pref within +-pi/6", but the authors'
#: reference implementation (mit-acl/rl_collision_avoidance /
#: gym-collision-avoidance network.Actions) has five evenly spaced headings at
#: v_pref, three at half speed and three at zero. The count 11 only holds with
#: this table. The project default instead uses the shared 80-command grid
#: (env.social_actions), keeping CADRL/SARL/GA3C-CADRL comparable. To switch,
#: multiply this normalised table's first column by the robot's preferred
#: speed, pass it as the agent's action_table, and build/mirror the environment
#: action set so indices line up (convert heading offsets to angular commands
#: using its control period where required).
GA3C_CADRL_PAPER_COMMANDS = (
    (1.0, -math.pi / 6), (1.0, -math.pi / 12), (1.0, 0.0),
    (1.0, math.pi / 12), (1.0, math.pi / 6),
    (0.5, -math.pi / 6), (0.5, 0.0), (0.5, math.pi / 6),
    (0.0, -math.pi / 6), (0.0, 0.0), (0.0, math.pi / 6),
)


class Ga3cCadrlEnv(SocialNavEnv):
    """Shared social task with Eq. (12) shaping in reward only.

    reward_parameters and self.social_reward are inherited unchanged; the
    inherited SocialReward object is unused by this reward implementation.
    Consequently the shared run description still reports reward=social.
    """

    def reward(self, previous, current, action, world) -> float:
        parameters = GA3C_CADRL_REWARD_PARAMETERS
        if current.goal_reached:
            return parameters["goal_bonus"]
        if current.collision or current.out_of_bounds:
            return parameters["collision_penalty"]
        distance = self.min_human_distance(world)
        if distance is not None and 0 < distance < parameters["near_miss_distance"]:
            return parameters["near_miss_base"] + parameters["near_miss_slope"] * distance
        return parameters["time_penalty"]


_VALUE_CLASS = None


def ga3c_cadrl_class() -> type:
    """Lazily construct the shared actor-critic network without import-time torch."""
    global _VALUE_CLASS
    if _VALUE_CLASS is None:
        import torch
        from torch import nn

        class Ga3cCadrlValue(nn.Module):
            def __init__(self, *, ego_features, neighbour_features, actions,
                         hidden=64, action_features=2):
                super().__init__()
                if int(ego_features) <= 0 or int(neighbour_features) <= 0:
                    raise ValueError("ego_features and neighbour_features must be positive")
                if int(actions) <= 0:
                    raise ValueError("an agent needs at least one action")
                self.ego_features = int(ego_features)
                self.neighbour_features = int(neighbour_features)
                self.actions = int(actions)
                self.hidden = int(hidden)
                self.action_features = int(action_features)
                self.encoder = nn.Sequential(
                    nn.Linear(self.neighbour_features, self.hidden), nn.ReLU(),
                    nn.Linear(self.hidden, self.hidden), nn.ReLU(),
                )
                self.lstm = nn.LSTM(self.hidden, self.hidden, num_layers=1, batch_first=True)
                self.trunk = nn.Sequential(
                    nn.Linear(self.ego_features + self.hidden, self.hidden), nn.ReLU(),
                    nn.Linear(self.hidden, self.hidden), nn.ReLU(),
                )
                self.value_head = nn.Linear(self.hidden, 1)
                self.policy_head = nn.Linear(self.hidden, self.actions)
                self.register_buffer("action_table", torch.zeros(0, self.action_features))

            def _tensor(self, value):
                reference = next(self.parameters())
                return torch.as_tensor(value, device=reference.device, dtype=reference.dtype)

            def set_action_table(self, table):
                values = self._tensor(table)
                if values.ndim != 2 or values.shape[1] != self.action_features:
                    raise ValueError(
                        f"an action table is (actions, {self.action_features}), "
                        f"got shape {tuple(values.shape)}"
                    )
                if values.shape[0] != self.actions:
                    raise ValueError(
                        f"the table holds {values.shape[0]} actions but the network was built "
                        f"for {self.actions}"
                    )
                self.action_table = values.detach().clone()
                return self

            def forward(self, ego, neighbours, mask):
                ego, neighbours = self._tensor(ego), self._tensor(neighbours)
                if ego.ndim != 2:
                    raise ValueError(f"ego is (batch, features), got shape {tuple(ego.shape)}")
                if neighbours.ndim != 3:
                    raise ValueError(
                        f"neighbours are (batch, count, features), got shape {tuple(neighbours.shape)}"
                    )
                batch, count, features = neighbours.shape
                if ego.shape != (batch, self.ego_features):
                    raise ValueError(f"ego is ({batch}, {self.ego_features}) to match neighbour rows")
                if features != self.neighbour_features:
                    raise ValueError(f"neighbour features are {self.neighbour_features}, got {features}")
                keep = self._tensor(mask).bool()
                if keep.shape != (batch, count):
                    raise ValueError(f"the mask is ({batch}, {count}), got shape {tuple(keep.shape)}")
                if count == 0:
                    pooled = ego.new_zeros(batch, self.hidden)
                else:
                    safe = torch.where(keep.unsqueeze(-1), neighbours, torch.zeros_like(neighbours))
                    encoded = self.encoder(safe)
                    lengths = keep.sum(dim=1)
                    order = keep.argsort(dim=1, descending=True, stable=True)
                    ordered = encoded.gather(1, order.unsqueeze(-1).expand_as(encoded))
                    packed = nn.utils.rnn.pack_padded_sequence(
                        ordered, lengths.clamp(min=1).cpu(), batch_first=True, enforce_sorted=False,
                    )
                    _, (hidden, _) = self.lstm(packed)
                    pooled = hidden[-1] * (lengths > 0).unsqueeze(-1).to(hidden.dtype)
                state = self.trunk(torch.cat([ego, pooled], dim=1))
                return self.value_head(state).squeeze(-1), self.policy_head(state)

        _VALUE_CLASS = Ga3cCadrlValue
    return _VALUE_CLASS


def ga3c_cadrl_value(**kwargs):
    """Build a shared value/policy network; forward accepts only state inputs."""
    return ga3c_cadrl_class()(**kwargs)


class Ga3cAgent:
    """Single-actor synchronous A2C limit of the GA3C update.

    Multiple asynchronous GA3C workers are not reproduced. To get closer,
    run several workers/environments and batch their rollouts into one update.
    The paper's supervised initialisation from a CADRL policy is not reproduced
    either: a user could warm-start compatible encoder weights from CADRL
    (the different heads require adaptation), or add an imitation loss.
    Unknown constructor extras are silently ignored for make_agent compatibility.
    batch_size is retained as paper metadata; this single actor updates on
    n_steps or a terminal flush, rather than aggregating asynchronous workers.
    When normalising, the stored states are read again under the current
    running statistics at update time (as _DQNAgent does), so a rollout is not
    scored against the cold statistics in force when its first action was taken.
    """

    name = "ga3c_cadrl"

    def __init__(self, *, ego_features=EGO_SIZE + GOAL_SIZE,
                 neighbour_features=NEIGHBOUR_SIZE, actions, action_table=None,
                 hidden=64, learning_rate=2e-5, gamma=0.97,
                 entropy_coefficient=1e-4, n_steps=5, batch_size=100,
                 normalise=True, seed=None, device="cpu", explore_epsilon=0.0,
                 grad_clip=10.0, **extras):
        import torch

        self.ego_features = int(ego_features)
        self.neighbour_features = int(neighbour_features)
        self.actions = int(actions)
        self.action_features = 2
        self.hidden = int(hidden)
        self.learning_rate = float(learning_rate)
        self.gamma = float(gamma)
        self.entropy_coefficient = float(entropy_coefficient)
        self.n_steps = int(n_steps)
        self.batch_size = int(batch_size)
        self.normalise = bool(normalise)
        self.seed = seed
        self.device = str(device)
        self.explore_epsilon = float(explore_epsilon)
        self.grad_clip = float(grad_clip)
        if self.actions <= 0:
            raise ValueError("an agent needs at least one action")
        if self.n_steps <= 0:
            raise ValueError("n_steps must be positive")
        if seed is not None:
            torch.manual_seed(seed)
        if action_table is None:
            action_table = np.zeros((self.actions, self.action_features), dtype=np.float32)
            action_table[:, 0] = np.arange(self.actions)
        self.online = ga3c_cadrl_value(
            ego_features=self.ego_features, neighbour_features=self.neighbour_features,
            actions=self.actions, hidden=self.hidden, action_features=self.action_features,
        ).to(self.device)
        self.online.set_action_table(action_table)
        self.optimiser = torch.optim.Adam(self.online.parameters(), lr=self.learning_rate)
        self.epsilon = float(explore_epsilon)
        self.steps = 0
        self.rng = np.random.default_rng(seed)
        self.norm = _RunningNorm(self.ego_features, self.neighbour_features) if normalise else None
        self._rollout = []
        self._pending = None

    def _pack(self, observation):
        return tuple(entry.copy() for entry in pack_observation(observation))

    def _state(self, observation):
        return self._normalised(self._pack(observation))

    def _normalised(self, packed):
        ego, neighbours, mask = packed
        if self.norm is not None:
            ego, neighbours = self.norm.look(ego, neighbours)
        return ego, neighbours, mask

    def _forward(self, packed):
        ego, neighbours, mask = self._normalised(packed)
        value, logits = self.online(ego[None], neighbours[None], mask[None])
        return value[0], logits[0]

    def act(self, observation, greedy=False) -> int:
        import torch

        packed = self._pack(observation)
        if greedy:
            self._pending = None
            with torch.no_grad():
                _, logits = self._forward(packed)
                return int(logits.argmax().item())
        value, logits = self._forward(packed)
        log_probs = torch.log_softmax(logits, dim=-1)
        if self.epsilon > 0 and self.rng.random() < self.epsilon:
            action = int(self.rng.integers(self.actions))
        else:
            action = int(torch.multinomial(log_probs.exp(), 1).item())
        self._pending = (packed, action, log_probs[action], value, logits)
        return action

    def observe(self, observation, action, reward, next_observation, done) -> None:
        import torch

        packed = self._pack(observation)
        pending = self._pending
        self._pending = None
        if (pending is not None and pending[1] == action
                and all(np.array_equal(a, b) for a, b in zip(pending[0], packed))):
            _, _, log_prob, value, logits = pending
        else:
            # Defensive: trainers may call act/observe out of step, or observe
            # a transition that this agent did not act on.
            value, logits = self._forward(packed)
            log_prob = torch.log_softmax(logits, dim=-1)[int(action)]
        self._rollout.append((packed, int(action), log_prob, value, logits,
                              float(reward), bool(done),
                              self._pack(next_observation)))
        if self.norm is not None:
            self.norm.observe(*packed)

    def learn(self) -> float | None:
        import torch
        from torch.nn import functional as F

        if not self._rollout or (len(self._rollout) < self.n_steps and not self._rollout[-1][6]):
            return None
        last = self._rollout[-1]
        with torch.no_grad():
            running = last[3].new_zeros(()) if last[6] else self._forward(last[7])[0].detach()
            returns = []
            for step in reversed(self._rollout):
                running = step[5] + self.gamma * (1 - step[6]) * running
                returns.append(running)
            returns = torch.stack(returns[::-1])
        if self.norm is None:
            # Nothing shifted under our feet, so the log-probability and value
            # read when the action was taken are exactly what the update needs.
            log_probs = torch.stack([step[2] for step in self._rollout])
            values = torch.stack([step[3] for step in self._rollout])
            logits = torch.stack([step[4] for step in self._rollout])
        else:
            # The running normaliser has moved since the actions were taken, so
            # the states are read again under the statistics the network now
            # sees - the same reason _DQNAgent normalises inside update().
            ego = np.stack([step[0][0] for step in self._rollout])
            neighbours = np.stack([step[0][1] for step in self._rollout])
            mask = np.stack([step[0][2] for step in self._rollout])
            ego, neighbours = self.norm.look(ego, neighbours)
            values, logits = self.online(ego, neighbours, mask)
            actions = torch.as_tensor(
                [step[1] for step in self._rollout], dtype=torch.int64, device=logits.device
            )
            log_probs = torch.log_softmax(logits, dim=-1).gather(
                1, actions.unsqueeze(1)
            ).squeeze(1)
        advantage = (returns - values).detach()
        policy_loss = -(log_probs * advantage).mean()
        value_loss = F.mse_loss(values, returns)
        log_distribution = F.log_softmax(logits, dim=-1)
        entropy = -(log_distribution.exp() * log_distribution).sum(dim=-1).mean()
        loss = value_loss + policy_loss - self.entropy_coefficient * entropy
        self.optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.grad_clip)
        self.optimiser.step()
        self.steps += 1
        self._rollout.clear()
        self._pending = None  # A pending graph would refer to pre-update weights.
        return float(loss.item())

    def config(self) -> dict:
        keys = ("ego_features", "neighbour_features", "actions", "action_features",
                "hidden", "learning_rate", "gamma", "entropy_coefficient", "n_steps",
                "batch_size", "normalise", "seed", "device", "explore_epsilon", "grad_clip")
        result = {key: getattr(self, key) for key in keys}
        result["action_table"] = self.online.action_table.detach().cpu().tolist()
        return result

    def save(self, path) -> str:
        import torch

        torch.save({"format": FORMAT, "algorithm": self.name, "config": self.config(),
                    "state_dict": self.online.state_dict(), "epsilon": float(self.epsilon),
                    "steps": int(self.steps),
                    "normaliser": None if self.norm is None else self.norm.state()}, path)
        return str(path)

    @classmethod
    def load(cls, path, *, device=None):
        import torch

        document = torch.load(path, map_location="cpu")
        if document.get("format") != FORMAT:
            raise ValueError(f"{path} is not a {FORMAT} checkpoint")
        if document.get("algorithm") != cls.name:
            raise ValueError(
                f"{path} holds a {document.get('algorithm')!r} agent, not a {cls.name!r} one"
            )
        return cls._from_document(document, device=device)

    @classmethod
    def _from_document(cls, document, *, device=None):
        config = dict(document["config"])
        if device is not None:
            config["device"] = device
        agent = cls(**config)
        agent.online.load_state_dict(document["state_dict"])
        agent.online.eval()
        agent.epsilon = float(document.get("epsilon", agent.epsilon))
        agent.steps = int(document.get("steps", 0))
        if agent.norm is not None and document.get("normaliser"):
            agent.norm = _RunningNorm.from_state(document["normaliser"])
        return agent


ENVIRONMENT = Ga3cCadrlEnv
AGENT = Ga3cAgent
