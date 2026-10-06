"""Index of the inference runs: which kind of checkpoint lives where.

- :mod:`robotsnap.runs.inference_sb3` - ``run_sb3_policy`` (Stable-Baselines3).
- :mod:`robotsnap.runs.inference_social` - ``run_social_policy`` and ``social_checkpoint_algorithm``.
- :mod:`robotsnap.runs.inference_torch` - ``run_policy_episodes``, ``_checkpoint_kind`` and ``_action_count``.

This module holds no logic: it only re-exports, so ``robotsnap.runs.inference``
keeps naming every inference run the package used to gather here.
"""

from __future__ import annotations

from robotsnap.runs.inference_sb3 import run_sb3_policy
from robotsnap.runs.inference_social import run_social_policy, social_checkpoint_algorithm
from robotsnap.runs.inference_torch import _action_count, _checkpoint_kind, run_policy_episodes
