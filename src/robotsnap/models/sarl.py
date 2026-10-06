"""SARL, whole: its reward constants, its value network, its agent, its task.

Chen et al. (2019) answer the same question CADRL answers, with the same
learning rule. They train a *state* value ``V(s)`` - one scalar per state, by
temporal difference with an experience replay, a fixed target network and an
epsilon-greedy behaviour policy - and read the action off a one-step model
lookahead. Their Algorithm 1 line 10 is explicit about the target being
``y_i = r_i + gamma * V_hat(s_{i+1})``, with no maximum over the actions; only
the encoder changes, replacing CADRL's LSTM over the neighbour rows with
multi-head self-attention and a masked mean. What they reward, and what the
robot observes and commands, is CADRL's: the task is shared and the encoder is
not.

Reference: Chen, C., Liu, Y., Kreiss, S. and Alahi, A. (2019), "Crowd-Robot
Interaction: Crowd-Aware Robot Navigation with Attention-Based Deep
Reinforcement Learning", ICRA 2019 (arXiv:1809.08835).

SARL and CADRL share this task - the same structured observation, the same
eighty commands and the same family of social cost - so
:data:`SARL_REWARD_PARAMETERS` is identical to
:data:`~robotsnap.models.cadrl.CADRL_REWARD_PARAMETERS`. Naming the constants
here is what lets a study change SARL's shaping without touching the shared task
or its sibling.
"""

from __future__ import annotations

from typing import Any

from robotsnap.models.lookahead import LookaheadAgent, state_value_base
from robotsnap.models.social import SocialNavEnv

__all__ = [
    "AGENT",
    "ENVIRONMENT",
    "SARL_REWARD_PARAMETERS",
    "SarlAgent",
    "SarlEnv",
    "sarl_class",
    "sarl_value",
]

#: The social cost Chen et al. (2019) inherit from CADRL, as the numbers a run
#: starts from. The values are the shared defaults of
#: :class:`~robotsnap.models.social.SocialReward`.
SARL_REWARD_PARAMETERS: dict[str, float] = {
    "progress_weight": 1.0,
    "collision_penalty": 10.0,
    "goal_bonus": 10.0,
    "time_penalty": 0.1,
    "comfort_distance": 1.0,
    "personal_space_weight": 1.0,
    "too_close_distance": 0.5,
    "too_close_penalty": 5.0,
}

#: The number of attention heads the paper's encoder uses.
SARL_HEADS = 4


class SarlEnv(SocialNavEnv):
    """The SARL task, in the four things that make a reinforcement problem.

    - observation: the structured social state - the robot's features, the goal
      in the robot frame and one masked row per neighbour, nearest first;
    - actions: the same eighty discrete commands CADRL uses - five speeds
      crossed with sixteen headings, plus the do-nothing command;
    - reward: the shared social cost - progress towards the goal, a charge per
      second, a Gaussian charge for crowding a human, flat prices for a
      collision and a bonus for arriving;
    - agent: :class:`SarlAgent`, which trains the state value ``V(s)`` by
      temporal difference with replay, a target network and epsilon-greedy
      exploration, and picks its command by a one-step model lookahead. The
      target is ``y = r + gamma * V_hat(s')`` - SARL's Algorithm 1 line 10, with
      no maximum over the actions - and the action rule is Chen et al. (2017)'s
      Eq. (5)-(7), which SARL inherits.

    A study that wants SARL to train against a different shaping replaces
    :data:`SARL_REWARD_PARAMETERS` - or this class's :attr:`reward_parameters` -
    and leaves every other line of the catalogue alone.
    """

    reward_parameters = SARL_REWARD_PARAMETERS


_VALUE_CLASS: type | None = None


def sarl_class() -> type:
    """SARL's state-value network class, built once on the first call.

    Built here rather than at module import so that reading the catalogue does
    not pull PyTorch in.
    """
    global _VALUE_CLASS
    if _VALUE_CLASS is None:
        base = state_value_base()

        class SarlValue(base):
            """SARL's state value ``V(s)``: self-attention over the neighbours, then a mean.

            Attention lets every neighbour look at every other one before the
            summary is taken, which is what the relational variant is for;
            padded rows are masked out of the attention and out of the mean.
            There is no action input: this is ``V(s)``, not ``Q(s, a)``.
            """

            def __init__(self, *, heads: int = SARL_HEADS, **kwargs: Any):
                super().__init__(**kwargs)
                import torch.nn as nn

                if self.hidden % int(heads):
                    raise ValueError(
                        f"the hidden size {self.hidden} must be a multiple of the {heads} heads"
                    )
                self.attention = nn.MultiheadAttention(
                    self.hidden, int(heads), batch_first=True
                )

            def pool(self, encoded, keep):
                present = keep.any(dim=1)
                # MultiheadAttention rejects a row whose every key is masked, so
                # a state with no neighbour keeps one key alive and is zeroed
                # after the pooling, where the mean would otherwise divide by it.
                padding = ~keep
                padding = padding.clone()
                padding[~present, 0] = False
                attended, _ = self.attention(
                    encoded, encoded, encoded, key_padding_mask=padding
                )
                weights = keep.to(attended.dtype).unsqueeze(-1)
                total = weights.sum(dim=1).clamp(min=1.0)
                pooled = (attended * weights).sum(dim=1) / total
                return pooled * present.unsqueeze(-1).to(pooled.dtype)

        _VALUE_CLASS = SarlValue
    return _VALUE_CLASS


def __getattr__(name: str):
    """``SarlValue`` on request, so the class name exists without torch at import."""
    if name == "SarlValue":
        return sarl_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def sarl_value(**kwargs: Any):
    """Build a fresh SARL state-value network.

    The same arguments as :func:`~robotsnap.models.cadrl.cadrl_value` plus
    ``heads``, the number of attention heads.
    """
    return sarl_class()(**kwargs)


class SarlAgent(LookaheadAgent):
    """SARL's learner: ``V(s)`` by TD, and one-step lookahead for the action.

    The TD regression and the lookahead are the shared ones of
    :class:`~robotsnap.models.lookahead.LookaheadAgent`; the only thing named
    here is the encoder, SARL's self-attention over the neighbours.
    """

    name = "sarl"

    @property
    def value_class(self) -> type:
        """SARL's ``V(s)`` network class, built on the first use."""
        return sarl_class()


#: What the catalogue registers for ``sarl``: the task and the agent on it.
ENVIRONMENT = SarlEnv
AGENT = SarlAgent
