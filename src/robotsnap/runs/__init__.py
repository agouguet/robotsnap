"""The routines behind the one entry point, ``python -m robotsnap``.

This used to be one long module; it is now a package of short ones, one per
responsibility. The map:

- :mod:`robotsnap.runs.presets` - the default world of each command.
- :mod:`robotsnap.runs.scenario_document` - the demo scenario document.
- :mod:`robotsnap.runs.controllers` - the scripted go-to-goal controller and its polar encoding.
- :mod:`robotsnap.runs.session` - how a command opens and closes a session with Unity.
- :mod:`robotsnap.runs.episodes` - the random and scripted episodes, and their formatters.
- :mod:`robotsnap.runs.training` - index of :mod:`robotsnap.runs.training_reinforce` (REINFORCE),
  :mod:`robotsnap.runs.training_sb3` (Stable-Baselines3) and :mod:`robotsnap.runs.training_social`
  (the social methods).
- :mod:`robotsnap.runs.inference` - index of :mod:`robotsnap.runs.inference_sb3`,
  :mod:`robotsnap.runs.inference_social` and :mod:`robotsnap.runs.inference_torch`: playing a saved
  checkpoint, without learning.
- :mod:`robotsnap.runs.scenario_control` - writing, launching and driving a scenario.
- :mod:`robotsnap.runs.benchmark` - the timed run: simulated seconds against wall seconds.
- :mod:`robotsnap.runs.campaign` - the scoring run: a suite of scenarios, a policy, and the
  campaign it left behind; ``benchmark`` binds them, :mod:`robotsnap.analysis.campaign` runs them.

Pour comprendre le projet, commence par :mod:`robotsnap.runs.session`, qui ouvre
la session, puis :mod:`robotsnap.runs.episodes`, qui la fait tourner.

Nothing here imports ``gymnasium``, ``numpy`` or ``torch`` at module import: the
environment extra is pulled in by the runs that need it, inside the calls, so
the command line stays usable - ``--help`` included - in an interpreter that has
only the base package.

Every name the old single module defined is re-exported here - public and
private - so ``robotsnap.runs.<name>`` keeps working, and so do the places that
replace one of them on the module at runtime.
"""

from __future__ import annotations

import math  # noqa: F401
import sys  # noqa: F401
import time  # noqa: F401
from collections.abc import Mapping  # noqa: F401
from pathlib import Path  # noqa: F401
from typing import Any  # noqa: F401

from robotsnap import scenario  # noqa: F401
from robotsnap.rl import policy  # noqa: F401
from robotsnap.runs.spec import describe_run  # noqa: F401

from robotsnap.runs.benchmark import run_benchmark
from robotsnap.runs.campaign import compare, run_scenarios
from robotsnap.runs.controllers import _POLAR, _polar, go_to_goal
from robotsnap.runs.episodes import (
    _closest,
    _metres,
    _observation_summary,
    _outcome,
    _why,
    run_goal_episode,
    run_random_episode,
)
from robotsnap.runs.inference import (
    _action_count,
    _checkpoint_kind,
    run_policy_episodes,
    run_sb3_policy,
    run_social_policy,
    social_checkpoint_algorithm,
)
from robotsnap.runs.presets import (
    bench_fields,
    random_episode_fields,
    social_fields,
    training_fields,
)
from robotsnap.runs.scenario_control import run_scenario
from robotsnap.runs.scenario_document import scenario_document
from robotsnap.runs.session import (
    _announce,
    _announce_session,
    _build_environment,
    _environment_options,
    _scenario_directory,
    _stop_session,
)
from robotsnap.runs.training import (
    _reinforce_components,
    _social_nav_class,
    run_sb3_training,
    run_social_training,
    run_training,
)

__all__ = [
    "go_to_goal",
    "random_episode_fields",
    "social_fields",
    "training_fields",
    "bench_fields",
    "scenario_document",
    "run_random_episode",
    "run_goal_episode",
    "run_training",
    "run_policy_episodes",
    "run_scenario",
    "run_benchmark",
    "run_scenarios",
    "compare",
]
