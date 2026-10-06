"""CADRL, whole: its reward constants, its value network, its agent, its task.

Chen et al. (2017) do not learn ``Q(s, a)``. They learn a *state* value
``V(s)`` - a scalar per state, trained by temporal difference with an experience
replay, a duplicated target network and an epsilon-greedy behaviour policy
(Algorithm 1, "Deep V-learning") - and turn it into a policy with a one-step
model lookahead: for every candidate command they propagate the state by a
constant-velocity model and take

    ``a* = argmax_a [ R_col(s, a) + gamma^(dt * v_pref) * V(s_hat'_{a}) ]``,

their Eq. (5)-(7). The task, the action set, the reward and the lookahead
machinery are the shared ones of :mod:`robotsnap.models.social` and
:mod:`robotsnap.models.lookahead`; what makes this *CADRL* is the encoder, an
LSTM over the per-neighbour rows, so the number of neighbours is a property of
the sequence rather than of the network.

Reference: Chen, Y. F., Everett, M., Liu, M. and How, J. P. (2017), "Socially
Aware Motion Planning with Deep Reinforcement Learning", IROS 2017
(arXiv:1703.08862).

CADRL and SARL share this task - the same structured observation, the same
eighty commands and the same family of social cost - so
:data:`CADRL_REWARD_PARAMETERS` is identical to
:data:`~robotsnap.models.sarl.SARL_REWARD_PARAMETERS`. Naming the constants here
is what lets a study change CADRL's shaping without touching the shared task or
its sibling, and what makes a reader find them without opening ``social.py``.
"""

from __future__ import annotations

from typing import Any

from robotsnap.models.lookahead import LookaheadAgent, state_value_base
from robotsnap.models.social import SocialNavEnv

__all__ = [
    "AGENT",
    "CADRL_REWARD_PARAMETERS",
    "ENVIRONMENT",
    "CadrlAgent",
    "CadrlEnv",
    "cadrl_class",
    "cadrl_value",
]

#: The social cost of Chen et al. (2017), as the numbers a run starts from.
#: The values are the shared defaults of
#: :class:`~robotsnap.models.social.SocialReward`.
CADRL_REWARD_PARAMETERS: dict[str, float] = {
    "progress_weight": 1.0,
    "collision_penalty": 10.0,
    "goal_bonus": 10.0,
    "time_penalty": 0.1,
    "comfort_distance": 1.0,
    "personal_space_weight": 1.0,
    "too_close_distance": 0.5,
    "too_close_penalty": 5.0,
}

#: The number of LSTM layers the paper's encoder stacks.
CADRL_LAYERS = 1


class CadrlEnv(SocialNavEnv):
    """The CADRL task, in the four things that make a reinforcement problem.

    - observation: the structured social state - the robot's features, the goal
      in the robot frame and one masked row per neighbour, nearest first;
    - actions: the eighty discrete commands of CADRL's table - five speeds
      crossed with sixteen headings, plus the do-nothing command;
    - reward: the shared social cost - progress towards the goal, a charge per
      second, a Gaussian charge for crowding a human, flat prices for a
      collision and a bonus for arriving;
    - agent: :class:`CadrlAgent`, which trains the state value ``V(s)`` by
      temporal difference with replay, a target network and epsilon-greedy
      exploration, and picks its command by a one-step model lookahead - Eq.
      (5)-(7) and Algorithm 1 "Deep V-learning" of Chen et al. (2017).

    A study that wants CADRL to train against a different shaping replaces
    :data:`CADRL_REWARD_PARAMETERS` - or this class's :attr:`reward_parameters`
    - and leaves every other line of the catalogue alone.
    """

    reward_parameters = CADRL_REWARD_PARAMETERS


_VALUE_CLASS: type | None = None


def cadrl_class() -> type:
    """CADRL's state-value network class, built once on the first call.

    Built here rather than at module import so that reading the catalogue - or
    importing this module to see what CADRL is - does not pull PyTorch in.
    """
    global _VALUE_CLASS
    if _VALUE_CLASS is None:
        base = state_value_base()

        class CadrlValue(base):
            """CADRL's state value ``V(s)``: an LSTM over the neighbours.

            The neighbours are a sequence and the last hidden state is its
            summary, which is what makes the same network work for one neighbour
            and for twenty. Padded rows never enter the recurrence, so the
            summary of three neighbours is the same whether the batch was built
            with room for three or for eight. There is no action input: this is
            ``V(s)``, not ``Q(s, a)``.
            """

            def __init__(self, *, layers: int = CADRL_LAYERS, **kwargs: Any):
                super().__init__(**kwargs)
                import torch.nn as nn

                self.lstm = nn.LSTM(
                    self.hidden, self.hidden, num_layers=int(layers), batch_first=True
                )

            def pool(self, encoded, keep):
                import torch
                from torch import nn

                lengths = keep.sum(dim=1)
                # pack_padded_sequence reads the first `length` rows of each
                # sequence, so the valid rows are moved to the front first - a
                # caller may hand the mask in any order and still be right.
                order = keep.to(torch.int8).argsort(dim=1, descending=True, stable=True)
                ordered = encoded.gather(1, order.unsqueeze(-1).expand_as(encoded))
                packed = nn.utils.rnn.pack_padded_sequence(
                    ordered,
                    lengths.clamp(min=1).cpu(),
                    batch_first=True,
                    enforce_sorted=False,
                )
                _, (hidden, _) = self.lstm(packed)
                # A state with no neighbour at all has an empty sequence, which
                # packs as one padded row; zeroing it here is what keeps the
                # empty case equal to "no neighbour contributed".
                return hidden[-1] * (lengths > 0).unsqueeze(-1).to(hidden.dtype)

        _VALUE_CLASS = CadrlValue
    return _VALUE_CLASS


def __getattr__(name: str):
    """``CadrlValue`` on request, so the class name exists without torch at import."""
    if name == "CadrlValue":
        return cadrl_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def cadrl_value(**kwargs: Any):
    """Build a fresh CADRL state-value network.

    Keyword arguments are the constructor's: ``ego_features``,
    ``neighbour_features``, ``hidden`` and ``layers``. Calling this is what pulls
    PyTorch in; importing this module is not.
    """
    return cadrl_class()(**kwargs)


class CadrlAgent(LookaheadAgent):
    """CADRL's learner: ``V(s)`` by TD, and one-step lookahead for the action.

    The TD regression and the lookahead are the shared ones of
    :class:`~robotsnap.models.lookahead.LookaheadAgent`; the only thing named
    here is the encoder, CADRL's LSTM over the neighbours.
    """

    name = "cadrl"

    @property
    def value_class(self) -> type:
        """CADRL's ``V(s)`` network class, built on the first use."""
        return cadrl_class()


#: What the catalogue registers for ``cadrl``: the task and the agent on it.
ENVIRONMENT = CadrlEnv
AGENT = CadrlAgent
