"""The learning half: the network, its checkpoint, and the algorithms over it.

Two modules, one per thing a training run needs:

- :mod:`robotsnap.rl.policy` - the learned policy, the running normaliser of its
  observations, and the checkpoint the two travel in. The one place that knows
  the shape of the network of this package's own REINFORCE loop.
- :mod:`robotsnap.rl.sb3` - Stable-Baselines3's PPO, A2C, SAC and DQN on the
  RobotSNAP task, with the discrete action set a value-based method needs.
- :mod:`robotsnap.rl.curriculum` - the ordered ladder of difficulty stages a run
  climbs, and :mod:`robotsnap.rl.curriculum_env` - the Gymnasium wrapper that
  applies a stage's scenario on every reset.

Both keep their frameworks behind factories, so importing this package imports
neither ``torch``, ``stable_baselines3`` nor ``numpy``: a run that only loads a
checkpoint - or the command line, ``--help`` included - stays usable in an
interpreter that has only the base package. The names below are re-exported for
convenience; the two modules remain the reference, at
``robotsnap.rl.policy`` and ``robotsnap.rl.sb3``.

The curriculum names are re-exported lazily, through the module-level
``__getattr__`` below: ``robotsnap.rl.curriculum`` is importable on its own, and
``CurriculumEnv`` - the one name that needs ``gymnasium`` - is imported only
when it is actually asked for.
"""

from importlib import import_module

from robotsnap.rl import policy, sb3
from robotsnap.rl.policy import (
    FORMAT,
    load_policy,
    policy_class,
    running_norm_class,
    save_policy,
)
from robotsnap.rl.sb3 import (
    ALGORITHMS,
    DISCRETE_COMMANDS,
    DiscreteCommands,
    algorithm_class,
    check_environment,
    load_model,
    make_env,
    make_vec_env,
    play,
    train,
)

__all__ = [
    "policy",
    "sb3",
    "FORMAT",
    "policy_class",
    "running_norm_class",
    "save_policy",
    "load_policy",
    "ALGORITHMS",
    "DISCRETE_COMMANDS",
    "DiscreteCommands",
    "algorithm_class",
    "check_environment",
    "load_model",
    "make_env",
    "make_vec_env",
    "play",
    "train",
    "Curriculum",
    "Stage",
    "CurriculumError",
    "CurriculumEnv",
]

#: The curriculum names this package re-exports, and the module each lives in.
#: Kept as strings so that naming a name here does not import its module.
_LAZY_NAMES = {
    "Curriculum": "robotsnap.rl.curriculum",
    "Stage": "robotsnap.rl.curriculum",
    "CurriculumError": "robotsnap.rl.curriculum",
    "CurriculumEnv": "robotsnap.rl.curriculum_env",
}


def __getattr__(name: str):
    """Import a curriculum name on first use, keeping gymnasium out of import.

    ``robotsnap.rl.curriculum_env`` imports ``gymnasium``; a caller that only
    wants the ladder, or the command line that wants nothing at all, must not
    pay for it. ``from robotsnap.rl import CurriculumEnv`` still works - Python
    falls through to this hook - but ``import robotsnap.rl`` does not.
    """
    module_name = _LAZY_NAMES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(module_name), name)


def __dir__():
    return sorted(set(__all__) | set(globals()))
