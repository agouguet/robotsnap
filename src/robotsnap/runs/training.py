"""Index of the learning runs: which method of learning lives where.

- :mod:`robotsnap.runs.training_reinforce` - ``run_training`` and ``_reinforce_components``.
- :mod:`robotsnap.runs.training_sb3` - ``run_sb3_training`` (Stable-Baselines3).
- :mod:`robotsnap.runs.training_social` - ``run_social_training`` and ``_social_nav_class``.

This module holds no logic: it only re-exports, so ``robotsnap.runs.training``
keeps naming every learning run the package used to gather here.
"""

from __future__ import annotations

from robotsnap.runs.training_reinforce import _reinforce_components, run_training
from robotsnap.runs.training_sb3 import run_sb3_training
from robotsnap.runs.training_social import _social_nav_class, run_social_training
