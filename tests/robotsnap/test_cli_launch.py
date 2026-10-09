"""Tests for :mod:`robotsnap.cli.launch_unity`, the one-command launcher.

No real Unity is started: the player is a throwaway shell script, so what is
checked is that ``--launch`` starts it, that the run happens while it is up,
and that it is gone once the block ends - including when the block ends in the
``KeyboardInterrupt`` a Ctrl-C raises.
"""

import argparse
import os
import sys
from pathlib import Path

import pytest

from robotsnap import cli
from robotsnap.cli import commands, launch_unity
from robotsnap.cli.commands import COMMANDS


def _make_player(directory: Path) -> Path:
    """An executable stand-in for the player, in ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "robotsnap-unity.x86_64"
    path.write_text("#!/bin/sh\nwhile true; do sleep 1; done\n")
    path.chmod(0o755)
    return path


def _launch_args(player: Path | None) -> argparse.Namespace:
    return argparse.Namespace(
        launch=True,
        unity_app=None if player is None else str(player),
        headless=False,
        unity_log=None,
        unity_project=None,
        scenario=None,
        transport="tcp",
    )


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


# -- the no-op path ---------------------------------------------------------


def test_launched_unity_does_nothing_without_the_flag(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("find_player must not run without --launch")

    monkeypatch.setattr(launch_unity, "find_player", refuse)
    args = argparse.Namespace(launch=False, unity_app=None, headless=False, unity_log=None)
    with launch_unity.launched_unity(args) as application:
        assert application is None


def test_launched_unity_refuses_when_no_player_is_found(monkeypatch, tmp_path):
    monkeypatch.setattr(launch_unity, "find_player", lambda explicit: None)
    args = _launch_args(tmp_path / "missing-player")
    with pytest.raises(SystemExit) as exit_info:
        with launch_unity.launched_unity(args):
            raise AssertionError("the block must not be entered")
    message = str(exit_info.value)
    assert "--unity-app" in message
    assert "setup-workspace.sh" in message


# -- the lifecycle ----------------------------------------------------------


@pytest.mark.skipif(
    sys.platform.startswith("win"), reason="process groups are POSIX"
)
def test_launched_unity_runs_the_player_around_the_block(tmp_path, capsys):
    player = _make_player(tmp_path)
    pid = None
    with launch_unity.launched_unity(_launch_args(player)) as application:
        pid = application.pid
        assert pid is not None
        assert application.running is True
        assert _alive(pid) is True

    assert _alive(pid) is False
    out = capsys.readouterr().out
    assert "unity started :" in out
    assert "unity stopped : True" in out


@pytest.mark.skipif(
    sys.platform.startswith("win"), reason="process groups are POSIX"
)
def test_launched_unity_stops_the_player_on_a_keyboard_interrupt(tmp_path):
    player = _make_player(tmp_path)
    pid = None
    with pytest.raises(KeyboardInterrupt):
        with launch_unity.launched_unity(_launch_args(player)) as application:
            pid = application.pid
            assert pid is not None
            raise KeyboardInterrupt

    assert _alive(pid) is False


@pytest.mark.skipif(
    sys.platform.startswith("win"), reason="process groups are POSIX"
)
def test_launched_unity_refuses_a_player_that_dies_at_once(tmp_path, capsys):
    """A player that cannot start is reported now, not after the wait timeout."""
    player = tmp_path / "robotsnap-unity.x86_64"
    player.write_text("#!/bin/sh\nexit 7\n")
    player.chmod(0o755)

    args = _launch_args(player)
    args.headless = True
    args.unity_log = str(tmp_path / "player.log")
    with pytest.raises(SystemExit) as exit_info:
        with launch_unity.launched_unity(args):
            raise AssertionError("the block must not be entered")

    assert "exited while starting" in str(exit_info.value)
    out = capsys.readouterr().out
    assert "unity started :" in out
    assert "unity stopped :" in out


# -- the folder the player really reads -------------------------------------


def _installed_player(tmp_path: Path) -> tuple[Path, Path]:
    """A player with an installed layout: the executable and its data folder."""
    install = tmp_path / "install"
    player = _make_player(install)
    data = install / "robotsnap-unity_Data"
    (data / "StreamingAssets" / "Scenarios").mkdir(parents=True)
    return player, data


@pytest.mark.skipif(
    sys.platform.startswith("win"), reason="process groups are POSIX"
)
def test_launched_unity_points_the_run_at_the_players_own_scenarios(tmp_path):
    """An installed player reads its own folder, not the source checkout."""
    player, data = _installed_player(tmp_path)
    (data / "StreamingAssets" / "Scenarios" / "python_bench_demo.yaml").write_text(
        "map: basic/crowd\n"
    )

    args = _launch_args(player)
    args.scenario = "python_bench_demo"
    with launch_unity.launched_unity(args):
        assert args.unity_project == str(data)


@pytest.mark.skipif(
    sys.platform.startswith("win"), reason="process groups are POSIX"
)
def test_launched_unity_keeps_the_project_the_run_named(tmp_path):
    """``--unity-project`` is what an editor session reads: it stands."""
    player, _ = _installed_player(tmp_path)
    project = tmp_path / "Project"
    (project / "Assets" / "StreamingAssets").mkdir(parents=True)

    args = _launch_args(player)
    args.unity_project = str(project)
    with launch_unity.launched_unity(args):
        assert args.unity_project == str(project)


@pytest.mark.skipif(
    sys.platform.startswith("win") or os.geteuid() == 0,
    reason="a read-only folder needs a non-root user",
)
def test_launched_unity_refuses_a_scenario_the_player_cannot_read(tmp_path, capsys):
    """A read-only install that misses the scenario is said before it starts."""
    player, data = _installed_player(tmp_path)
    (data / "StreamingAssets" / "Scenarios").chmod(0o500)

    args = _launch_args(player)
    args.scenario = "python_bench_demo"
    with pytest.raises(SystemExit) as exit_info:
        with launch_unity.launched_unity(args):
            raise AssertionError("the block must not be entered")

    message = str(exit_info.value)
    assert "python_bench_demo" in message
    assert "not writable" in message
    assert "--scenario" in message
    assert "unity started" not in capsys.readouterr().out


@pytest.mark.skipif(
    sys.platform.startswith("win"), reason="process groups are POSIX"
)
def test_launch_says_when_the_project_holds_another_copy(tmp_path, monkeypatch, capsys):
    """The edit that "does nothing" is a project copy the player never reads."""
    player, data = _installed_player(tmp_path)
    (data / "StreamingAssets" / "Scenarios" / "default.yaml").write_text("map: old\n")
    project = tmp_path / "Project"
    edited = project / "Assets" / "StreamingAssets" / "Scenarios"
    edited.mkdir(parents=True)
    (edited / "default.yaml").write_text("map: new\n")
    monkeypatch.setattr(launch_unity.scenarios, "_FALLBACK_PROJECTS", (project,))

    args = _launch_args(player)
    args.scenario = "default"
    with launch_unity.launched_unity(args):
        pass

    out = capsys.readouterr().out
    assert "scenario note" in out
    assert str(edited / "default.yaml") in out


@pytest.mark.skipif(
    sys.platform.startswith("win"), reason="process groups are POSIX"
)
def test_launch_is_quiet_when_both_copies_agree(tmp_path, monkeypatch, capsys):
    """A build that was refreshed holds the same scenario, and says nothing."""
    player, data = _installed_player(tmp_path)
    (data / "StreamingAssets" / "Scenarios" / "default.yaml").write_text("map: same\n")
    project = tmp_path / "Project"
    edited = project / "Assets" / "StreamingAssets" / "Scenarios"
    edited.mkdir(parents=True)
    (edited / "default.yaml").write_text("map: same\n")
    monkeypatch.setattr(launch_unity.scenarios, "_FALLBACK_PROJECTS", (project,))

    args = _launch_args(player)
    args.scenario = "default"
    with launch_unity.launched_unity(args):
        pass

    assert "scenario note" not in capsys.readouterr().out


# -- the flag is everywhere -------------------------------------------------


@pytest.mark.parametrize("command", [module.NAME for module in COMMANDS])
def test_every_subcommand_offers_the_launch_options(command, capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main([command, "--help"])
    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    assert "--launch" in out
    assert "--unity-app" in out
    assert "--headless" in out
    assert "--unity-log" in out
    assert "launch options" in out


def test_a_ctrl_c_ends_the_command_line_quietly(monkeypatch, capsys):
    """Ctrl-C is a stop, not a crash: 130 and one word, no traceback."""

    def interrupt(args):
        raise KeyboardInterrupt

    monkeypatch.setattr(commands.episode, "run", interrupt)
    assert cli.main(["episode"]) == 130
    assert "interrupted" in capsys.readouterr().err
