"""The catalogue of named methods: one module per method, one registry over them.

This is where a name a command line can spell becomes the pair of classes that
drive it. Each named method lives in its own module of this package and holds
*all* of its parts - its environment, its reward constants, its policy and its
agent together - so a reader who wants to know what CADRL is reads one file, and
a study that adds a method adds one file and one line here.

Only what is genuinely shared lives beside them, in
:mod:`robotsnap.models.social`: the discrete command grid, the layout of the
structured observation, the social cost, the task base, the state-value base,
the lookahead learner and the plain DQN. A method inherits them and names only
what makes it that method.

The split with the core is deliberate and holds in both directions. Nothing
under :mod:`robotsnap` outside this package names CADRL or SARL, and importing
this package does not import the method modules: the tables below are plain
data, and the classes they point at are loaded when a name is asked for. A
command line that only lists what exists therefore pays for nothing, and a
machine that installs the base package can still run every command that does
not name a method.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

__all__ = [
    "ALGORITHMS",
    "ALGORITHM_CLASSES",
    "METHODS",
    "METHOD_LEARNER",
    "MODELS",
    "load_agent",
    "make_agent",
]

#: The named methods, and the one line a command line shows for each. The
#: wording is what ``--help`` prints, so it has to say what a method *is* rather
#: than how it is spelled.
METHODS: dict[str, str] = {
    "cadrl": (
        "CADRL: an LSTM over the neighbours, a state value trained by TD and "
        "played by a one-step lookahead"
    ),
    "sarl": (
        "SARL: self-attention over the neighbours, the same value learner and "
        "the same lookahead"
    ),
    "ga3c_cadrl": (
        "GA3C-CADRL: the same crowd encoder as CADRL, an actor-critic head, "
        "no lookahead"
    ),
    "rgl": (
        "RGL: a relational graph over the crowd, a learned multi-step "
        "lookahead over the same discrete commands"
    ),
    "template": "a minimal worked example to copy when writing a method of your own",
}

#: The learner each method ships with. A method is not a learner - it brings its
#: own observation, action set and reward - so a run narrates the two on
#: separate axes, and this is the second one. The labels are the papers' own
#: rules: CADRL, SARL and RGL learn a state value ``V(s)`` and act through a
#: lookahead ("deep V-learning"), GA3C-CADRL is an actor-critic (A3C), and only
#: the worked template still ships the plain DQN.
METHOD_LEARNER: dict[str, str] = {
    "cadrl": "v-learning",
    "sarl": "v-learning",
    "ga3c_cadrl": "a3c",
    "rgl": "v-learning",
    "template": "dqn",
}

#: Where each method's parts are, as a module of this package. Adding a method is
#: adding one module and one line here.
_METHOD_MODULES: dict[str, str] = {
    "cadrl": "robotsnap.models.cadrl",
    "sarl": "robotsnap.models.sarl",
    "ga3c_cadrl": "robotsnap.models.ga3c_cadrl",
    "rgl": "robotsnap.models.rgl",
    "template": "robotsnap.models.template",
}

_REGISTRY: dict[str, tuple[type, type]] | None = None


def _load(name: str) -> tuple[type, type]:
    """The ``(environment, agent)`` pair of one method, from its own module."""
    module = importlib.import_module(_METHOD_MODULES[name])
    return module.ENVIRONMENT, module.AGENT


def _registry() -> dict[str, tuple[type, type]]:
    """The name -> ``(environment, agent)`` table, built on the first call."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = {name: _load(name) for name in METHODS}
    return _REGISTRY


def __getattr__(name: str):
    """The registry, loaded when one of its names is read rather than imported.

    Keeping the classes out of the module's own imports is what lets a plain
    ``import robotsnap.models`` stay free of numpy, gymnasium and torch: the
    tables above are data, and a caller that only wants the list of names never
    touches a class.
    """
    if name == "ALGORITHMS":
        return _registry()
    if name == "MODELS":
        return _registry()
    if name == "ALGORITHM_CLASSES":
        return {method: pair[1] for method, pair in _registry().items()}
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def make_agent(name: str, env: Any, **options: Any) -> Any:
    """The agent ``name`` for ``env``, sized from its observation and action spaces.

    Sizing it here rather than at the call site is what keeps a method's policy
    and the environment it is trained on from disagreeing about how many actions
    there are or how wide a feature row is: both numbers are read off the
    environment. A method whose observation is not the structured one supplies a
    ``size_from`` on its agent, and this reads whichever of the two applies.
    """
    from robotsnap.models.social import EGO_SIZE, GOAL_SIZE, NEIGHBOUR_SIZE

    registry = _registry()
    if name not in registry:
        known = ", ".join(sorted(registry))
        raise ValueError(f"unknown algorithm {name!r}; known algorithms are {known}")
    agent_class = registry[name][1]

    size_from = getattr(agent_class, "size_from", None)
    if size_from is not None:
        defaults: dict[str, Any] = dict(size_from(env))
    else:
        defaults = {
            "ego_features": EGO_SIZE + GOAL_SIZE,
            "neighbour_features": NEIGHBOUR_SIZE,
        }
    space = getattr(env, "social_actions", None)
    if space is not None:
        defaults["actions"] = len(space)
        defaults["action_table"] = space.commands
    elif hasattr(env, "action_space"):
        defaults["actions"] = int(env.action_space.n)
    defaults.update(options)
    return agent_class(**defaults)


def load_agent(path: str | Path, *, device: str | None = None) -> Any:
    """Load whichever agent the checkpoint at ``path`` holds.

    Which method wrote a file is a property of the file: an agent writes its own
    name beside its weights, and reading it is truer than asking a caller to
    repeat on the command line what the checkpoint already says.
    """
    import torch

    from robotsnap.models.social import FORMAT

    document = torch.load(Path(path), map_location="cpu")
    if not isinstance(document, Mapping) or document.get("format") != FORMAT:
        raise ValueError(f"{path} is not a {FORMAT} checkpoint")
    name = document.get("algorithm")
    registry = _registry()
    if name not in registry:
        known = ", ".join(sorted(registry))
        raise ValueError(f"{path} names an unknown algorithm {name!r}; known ones are {known}")
    return registry[name][1]._from_document(document, device=device)
