"""The ``--launch`` option: start the RobotSNAP application around a run.

Every sub-command shares this. ``--launch`` opens the player that
:mod:`robotsnap.unity_app` finds, so one command starts the simulation, the
bridge and the run together and stops them again when the run ends - including
when it ends because the user pressed Ctrl-C. :func:`launched_unity` is the
context manager :func:`robotsnap.cli.main.main` wraps the handler in.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import signal
import tempfile
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from robotsnap import scenario as scenarios
from robotsnap.unity_app import (
    APP_ENV,
    MISSING_MESSAGE,
    UnityApplication,
    find_player,
    player_data_folder,
)

__all__ = ["add_launch_options", "launched_unity"]


def add_launch_options(parser: argparse.ArgumentParser) -> None:
    """Add the launch options every sub-command shares."""
    group = parser.add_argument_group("launch options")
    group.add_argument(
        "--launch",
        action="store_true",
        help="start the RobotSNAP application for this run and stop it when the run ends",
    )
    group.add_argument(
        "--unity-app",
        metavar="PATH",
        default=None,
        help="the player to start (default: discovered, see --launch)",
    )
    group.add_argument(
        "--headless",
        action="store_true",
        help=(
            "start the player with -batchmode -nographics : no window, much "
            "faster, works over ssh - pair it with --viewer to still watch"
        ),
    )
    group.add_argument(
        "--unity-log",
        metavar="PATH",
        default=None,
        help="where the player writes its log (default: a temporary file when --headless)",
    )


@contextlib.contextmanager
def launched_unity(args: argparse.Namespace) -> Iterator[UnityApplication | None]:
    """Start the application for ``args`` and stop it when the block ends.

    A run with ``--launch`` should not have to care where its player lives:
    this resolves it, starts it, and stops it in a ``finally``, so a run that
    ends early - an exception, a refused configuration, Ctrl-C - still closes
    the simulation. Ctrl-C reaches a Python process as ``KeyboardInterrupt``,
    but a plain ``kill`` sends SIGTERM; the handler installed below turns the
    two into the same unwinding, which is what lets the run's own stop reach
    Unity before the application is taken away.
    """
    if not getattr(args, "launch", False):
        yield None
        return

    log_path = _log_path(args)
    declared = getattr(args, "unity_app", None) or os.environ.get(APP_ENV)
    player = find_player(getattr(args, "unity_app", None))
    if player is None:
        # A path the run named is what it must get; anything else is a typo the
        # reader can fix, so it is named back rather than searched around.
        message = (
            f"no player at {declared}: {MISSING_MESSAGE}" if declared else MISSING_MESSAGE
        )
        raise SystemExit(message)

    _adopt_the_players_scenarios(args, player)

    application = UnityApplication(
        player,
        headless=getattr(args, "headless", False),
        log_path=log_path,
    )
    application.start()
    pid = application.pid
    print(f"unity started : {player} (pid {pid})", flush=True)
    if log_path is not None:
        print(f"unity log     : {log_path}", flush=True)
    if str(getattr(args, "transport", "tcp")) == "ros2":
        print(
            "unity note    : --transport ros2 has the player dial the ROS2 "
            "endpoint, which has to be running",
            flush=True,
        )

    previous, installed = _install_sigterm_handler()
    try:
        # A player that died before it dialled is a start failure, said before
        # the run waits out its whole --wait-timeout for a peer that is gone.
        _refuse_a_player_that_died(application, log_path)
        yield application
    finally:
        stopped = application.stop()
        if installed:
            signal.signal(signal.SIGTERM, previous)
        print(f"unity stopped : {stopped} (pid {pid})", flush=True)


def _adopt_the_players_scenarios(args: argparse.Namespace, player: Path) -> None:
    """Point the run at the scenarios the player it starts will really read.

    An installed application reads its scenarios from the data folder beside
    its executable, not from the source checkout a run would otherwise write
    into. A scenario written into the checkout is one the player never sees,
    and the run is refused by a simulator that does not know the name - so when
    the caller named no project of its own, the player's own folder is the one
    the run gets. An explicit ``--unity-project`` stands: a caller driving an
    editor session knows which folder that session reads.
    """
    if not hasattr(args, "unity_project") or getattr(args, "unity_project", None):
        return

    data = player_data_folder(player)
    if data is None:
        return

    _report_an_edited_copy(args, data)
    args.unity_project = str(data)
    print(f"unity project : {data}", flush=True)
    _refuse_an_unreachable_scenario(args, data)


def _report_an_edited_copy(args: argparse.Namespace, data: Path) -> None:
    """Say when the project's copy of a scenario is not the player's copy.

    An editor session reads the project's scenarios and a packaged player reads
    its own baked ones, so an edit made in one is invisible in the other - which
    is what "I changed the scenario and nothing happened" almost always is. One
    line naming both files, before the run starts, is worth more than playing
    the copy the caller was not looking at.
    """
    name = getattr(args, "scenario", None)
    if not name:
        return

    try:
        project = scenarios.unity_project()
        played = scenarios.find(str(name), scenarios.scenarios_dir(str(data)))
        edited = scenarios.find(str(name), scenarios.scenarios_dir(project))
    except FileNotFoundError:
        return

    if played is None or edited is None or _same_contents(played, edited):
        return
    print(
        f"scenario note : the player reads {played}, not the project's {edited}",
        flush=True,
    )


def _same_contents(one: Path, other: Path) -> bool:
    """Whether two scenario files are the same one, or hold the same bytes."""
    try:
        if one.samefile(other):
            return True
        return one.read_bytes() == other.read_bytes()
    except OSError:
        return False


def _refuse_an_unreachable_scenario(args: argparse.Namespace, data: Path) -> None:
    """Stop before the run when the scenario it asks for cannot reach the player.

    An application installed system-wide keeps its own folder read-only and can
    only play a scenario that folder already ships. A run that asks for another
    name, and cannot write one there, would otherwise start the session, build
    the world and be refused at its first reset; the folder and the two ways out
    are what belong in the message instead.
    """
    name = getattr(args, "scenario", None)
    if not name:
        return

    folder = scenarios.scenarios_dir(str(data))
    if scenarios.find(str(name), folder) is not None:
        return
    if os.access(folder, os.W_OK):
        return

    raise SystemExit(
        f"the application reads its scenarios from {folder}, which is not "
        f"writable, and does not ship {name!r}: install the application under "
        "your home (the .run installer's default prefix), where that folder "
        "belongs to you, or run with a scenario it ships - --scenario default, "
        "for example"
    )


def _log_path(args: argparse.Namespace) -> Path | None:
    """Where the player writes its log, a temporary file when headless.

    Headless is where a log matters most - there is no window to say what
    happened - so a headless run that named no file still gets one, and the
    path is printed beside the pid so it can be read once the run is over.
    """
    declared = getattr(args, "unity_log", None)
    if declared is not None:
        return Path(declared)
    if getattr(args, "headless", False):
        handle, name = tempfile.mkstemp(prefix="robotsnap-unity-", suffix=".log")
        os.close(handle)
        return Path(name)
    return None


def _refuse_a_player_that_died(
    application: UnityApplication, log_path: Path | None, *, settle: float = 0.5
) -> None:
    """Fail at once, with the log, when the player is gone before it dials.

    A player that cannot start - a missing library, a build for another machine,
    no display for a windowed run - exits within a moment. Left alone the run
    waits out its whole ``--wait-timeout`` and reports only that Unity never
    connected, so the moment is spent here instead, where the last lines of the
    player's own log are still the answer.
    """
    time.sleep(settle)
    if application.running:
        return
    tail = _log_tail(log_path)
    hint = (
        f"; its log is {log_path}"
        if log_path is not None
        else "; pass --unity-log to keep its log"
    )
    raise SystemExit(
        f"the RobotSNAP application exited while starting{hint}"
        + (f"\n{tail}" if tail else "")
    )


def _log_tail(log_path: Path | None, *, lines: int = 8) -> str:
    """The last non-empty lines of ``log_path``, or ``''`` when there is none.

    Only the tail of the file is read: a player's log runs to megabytes, and
    the reason a start failed is at the end of it.
    """
    if log_path is None:
        return ""
    try:
        with open(log_path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - 64 * 1024))
            text = handle.read().decode("utf-8", "replace")
    except OSError:
        return ""
    tail = [line for line in text.splitlines() if line.strip()][-lines:]
    return "\n".join(tail)


def _install_sigterm_handler() -> tuple[Any, bool]:
    """Make SIGTERM unwind like Ctrl-C, reporting whether it was installed.

    Signals are only handled in the main thread, and a library that installs
    one owes the caller the previous handler back: the pair returned is the
    handler that was there and whether the install happened at all, so the
    caller can put it back once the run is done.
    """
    if threading.current_thread() is not threading.main_thread():
        return None, False
    previous = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, _raise_interrupt)
    return previous, True


def _raise_interrupt(signum: int, frame: Any) -> None:
    """A SIGTERM as the same unwinding Ctrl-C already gives a run."""
    raise KeyboardInterrupt
