"""Deprecated home of the navigation methods; they live in :mod:`robotsnap.models`.

This module is a compatibility shim and nothing else. The methods used to be
split across this package by *responsibility* - one module for the action set,
one for the reward, one for each network - which meant that reading what CADRL
is took four files. They are now split by *method*: a method and all of its
parts live in one module of :mod:`robotsnap.models`, beside the primitives it
shares with the others in :mod:`robotsnap.models.social`.

Importing this package still works, because scripts and notebooks were written
against it, and the old submodule paths resolve too - each onto the module that
now holds what it used to. It warns, so the move is visible rather than silent.
New code should import from :mod:`robotsnap.models`.
"""

from __future__ import annotations

import importlib
import sys
import types
import warnings

warnings.warn(
    "robotsnap.social has moved to robotsnap.models: import the method you want "
    "from robotsnap.models (for example 'from robotsnap.models import "
    "ALGORITHMS') and read it in robotsnap/models/<method>.py",
    DeprecationWarning,
    stacklevel=2,
)

from robotsnap.models import (  # noqa: E402 - the warning has to come first
    ALGORITHM_CLASSES,
    ALGORITHMS,
    METHODS,
    METHOD_LEARNER,
    MODELS,
    load_agent,
    make_agent,
)
from robotsnap.models.cadrl import (  # noqa: E402
    CADRL_REWARD_PARAMETERS,
    CadrlAgent,
    CadrlEnv,
)
from robotsnap.models.sarl import (  # noqa: E402
    SARL_REWARD_PARAMETERS,
    SarlAgent,
    SarlEnv,
)
from robotsnap.models.social import (  # noqa: E402
    DEFAULT_HEADINGS,
    DEFAULT_MAX_NEIGHBOURS,
    DEFAULT_SOCIAL_PARAMETERS,
    DEFAULT_SPEEDS,
    EGO_KEY,
    EGO_SIZE,
    FORMAT,
    GOAL_KEY,
    GOAL_SIZE,
    HISTORY_KEYS,
    MASK_KEY,
    NEIGHBOURS_KEY,
    NEIGHBOUR_SIZE,
    SocialActionSpace,
    SocialNavEnv,
    SocialReward,
    as_social_reward,
    build_structured_observation,
    heading_offset_to_angular,
    pack_observation,
    play,
    train,
)

__all__ = [
    "ALGORITHMS",
    "ALGORITHM_CLASSES",
    "CADRL_REWARD_PARAMETERS",
    "CadrlAgent",
    "CadrlEnv",
    "SARL_REWARD_PARAMETERS",
    "SarlAgent",
    "SarlEnv",
    "SocialActionSpace",
    "SocialNavEnv",
    "SocialReward",
    "as_social_reward",
    "build_structured_observation",
    "heading_offset_to_angular",
    "load_agent",
    "make_agent",
    "pack_observation",
    "play",
    "train",
    "DEFAULT_HEADINGS",
    "DEFAULT_MAX_NEIGHBOURS",
    "DEFAULT_SPEEDS",
    "EGO_KEY",
    "EGO_SIZE",
    "GOAL_KEY",
    "GOAL_SIZE",
    "MASK_KEY",
    "NEIGHBOUR_SIZE",
    "NEIGHBOURS_KEY",
]


def _merged(name: str, *sources: str) -> types.ModuleType:
    """A stand-in for an old submodule, rebuilt from the modules that hold it."""
    module = types.ModuleType(name)
    for source in sources:
        source_module = importlib.import_module(source)
        for attribute in getattr(source_module, "__all__", ()):
            value = getattr(source_module, attribute, None)
            if value is not None:
                setattr(module, attribute, value)
    return module


#: The old submodule paths, each pointed at whatever holds its names now. The
#: shared primitives all live in one module, and the two names that used to be
#: split between ``actions``/``reward``/``models`` are merged where a reader of
#: the old path would expect to find them.
for _old_name, _sources in (
    ("actions", ("robotsnap.models.social",)),
    ("observation", ("robotsnap.models.social",)),
    ("reward", ("robotsnap.models.social",)),
    ("env", ("robotsnap.models.social",)),
    ("train", ("robotsnap.models.social",)),
    ("models", ("robotsnap.models.social", "robotsnap.models.cadrl", "robotsnap.models.sarl")),
    (
        "agents",
        (
            "robotsnap.models.social",
            "robotsnap.models.cadrl",
            "robotsnap.models.sarl",
            "robotsnap.models",
        ),
    ),
    ("cadrl", ("robotsnap.models.cadrl",)),
    ("sarl", ("robotsnap.models.sarl",)),
):
    sys.modules.setdefault(
        f"{__name__}.{_old_name}", _merged(f"{__name__}.{_old_name}", *_sources)
    )
