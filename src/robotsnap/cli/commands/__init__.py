"""Every sub-command of ``python -m robotsnap``, in the order the help lists them.

One module per command, each exposing the same four names - ``NAME``,
``ALIASES``, ``HELP`` and ``DESCRIPTION`` - plus ``add_arguments`` and ``run``.
:mod:`robotsnap.cli.main` walks this tuple, so a command is added to the entry
point by being imported here and nowhere else.
"""

from __future__ import annotations

from robotsnap.cli.commands import (
    bench,
    benchmark,
    bridge,
    episode,
    goal,
    play,
    scenario,
    train,
    watch,
)

#: The sub-commands, in the order ``--help`` shows them.
COMMANDS = (watch, episode, goal, train, play, scenario, bench, benchmark, bridge)

__all__ = ["COMMANDS"]
