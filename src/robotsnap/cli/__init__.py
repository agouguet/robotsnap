"""The one entry point: ``python -m robotsnap <command>``.

This package replaces the single ``cli`` module; the command line is the same,
split so every file does one job:

- :mod:`~robotsnap.cli.main` - ``build_parser`` and ``main``, the two names this
  package re-exports.
- :mod:`~robotsnap.cli.options` - the options every command shares (the bridge,
  the pacing, ``--method``, ``--stop``, the argument types) and the one place a
  command imports them from.
- :mod:`~robotsnap.cli.environment_options` - the environment surface: the
  observation, its parameters and the controller and episode limits.
- :mod:`~robotsnap.cli.commands` - the list of sub-commands, in display order.
- :mod:`~robotsnap.cli.commands.watch` - the ``watch`` command (alias
  ``viewer``): draw a live 2D view of a session.
- :mod:`~robotsnap.cli.commands.episode` - run one random-action episode.
- :mod:`~robotsnap.cli.commands.goal` - run the social task with the scripted
  controller.
- :mod:`~robotsnap.cli.commands.train` - train a policy.
- :mod:`~robotsnap.cli.commands.play` - play a saved policy back (alias
  ``infer``).
- :mod:`~robotsnap.cli.commands.scenario` - write, launch, drive and freeze a
  scenario.
- :mod:`~robotsnap.cli.commands.bench` - time N control steps.
- :mod:`~robotsnap.cli.commands.benchmark` - run a suite of scenarios and
  compare the campaigns (alias ``campaign``).
- :mod:`~robotsnap.cli.commands.bridge` - serve Unity, optionally mirrored on
  ROS2.

``train``, ``play``, ``bench`` and ``benchmark`` also take ``--config``: a file
of hyperparameters under ``configs/`` supplies the defaults the final parse
starts from, and an option typed on the command line still wins. ``train``
takes ``--curriculum``, the ladder of scenarios :mod:`robotsnap.rl.curriculum`
climbs.

Every command module exposes the same four names - ``NAME``, ``ALIASES``,
``HELP`` and ``DESCRIPTION`` - plus the pair ``add_arguments(parser)`` and
``run(args)``; :mod:`robotsnap.cli.main` walks them.
"""

from __future__ import annotations

from robotsnap.cli.main import build_parser, main

__all__ = ["build_parser", "main"]
