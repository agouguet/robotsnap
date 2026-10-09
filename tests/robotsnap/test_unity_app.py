"""Tests for :mod:`robotsnap.unity_app`, the Unity player as a process.

Nothing here starts Unity: discovery is checked against throwaway layouts, and
the process lifecycle against a small shell script that stands in for the
player by looping forever.
"""

import os
import sys
from pathlib import Path

import pytest

from robotsnap import unity_app


@pytest.fixture
def clean_environment(monkeypatch):
    """No launch environment of the machine leaks into a discovery test."""
    monkeypatch.delenv(unity_app.APP_ENV, raising=False)
    monkeypatch.delenv(unity_app.HOME_ENV, raising=False)
    monkeypatch.delenv(unity_app.SCENARIOS_ENV, raising=False)


def _make_player(directory: Path, name: str | None = None) -> Path:
    """An executable stand-in for the player, in ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (unity_app.player_name() if name is None else name)
    path.write_text("#!/bin/sh\nwhile true; do sleep 1; done\n")
    path.chmod(0o755)
    return path


# -- discovery --------------------------------------------------------------


def test_find_player_takes_an_explicit_executable(tmp_path, clean_environment):
    player = _make_player(tmp_path / "install")
    assert unity_app.find_player(player) == player


def test_find_player_takes_an_explicit_directory(tmp_path, clean_environment):
    player = _make_player(tmp_path / "install")
    assert unity_app.find_player(tmp_path / "install") == player


def test_find_player_reads_the_environment(monkeypatch, tmp_path, clean_environment):
    player = _make_player(tmp_path / "env-install")
    monkeypatch.setenv(unity_app.APP_ENV, str(tmp_path / "env-install"))
    assert unity_app.find_player() == player


def test_find_player_reads_the_workspace_layout(tmp_path, clean_environment):
    """``root/app/robotsnap-unity/<version>/`` is where the installer writes."""
    player = _make_player(tmp_path / "app" / "robotsnap-unity" / "0.1.0")
    assert unity_app.find_player(root=tmp_path) == player


def test_find_player_reads_the_user_prefix(monkeypatch, tmp_path, clean_environment):
    """``~/.local/share/robotsnap-unity/<version>/`` is the installer default."""
    player = _make_player(
        tmp_path / ".local" / "share" / "robotsnap-unity" / "1.2.3"
    )
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    assert unity_app.find_player() == player


def test_find_player_reads_the_workspace_clone(monkeypatch, tmp_path, clean_environment):
    """``~/robotsnap-workspace/app/...`` is where setup-workspace.sh installs.

    That root is not ``$ROBOTSNAP_HOME`` by default, so the clone beside the
    home directory has to be a candidate of its own.
    """
    player = _make_player(
        tmp_path / "robotsnap-workspace" / "app" / "robotsnap-unity" / "0.1.0"
    )
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(unity_app, "_USER_PREFIXES", ())
    monkeypatch.setattr(unity_app, "_SYSTEM_PREFIXES", ())
    assert unity_app.find_player() == player


def test_find_player_does_not_swap_a_path_that_does_not_resolve(
    tmp_path, clean_environment
):
    """An explicit path is an instruction: no player there means no player."""
    assert unity_app.find_player(tmp_path / "install") is None
    _make_player(tmp_path / "install")
    assert unity_app.find_player(tmp_path / "install") is not None
    assert unity_app.find_player(tmp_path / "typo") is None


def test_find_player_is_none_when_nothing_is_installed(
    monkeypatch, tmp_path, clean_environment
):
    """No candidate anywhere answers ``None`` rather than raising."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(unity_app, "_SYSTEM_PREFIXES", ())
    monkeypatch.setattr(unity_app.shutil, "which", lambda name: None)
    monkeypatch.setattr(unity_app, "_source_build_player", lambda: None)
    assert unity_app.find_player(root=tmp_path) is None


# -- the command line of a player -------------------------------------------


def test_player_command_windowed():
    assert unity_app.player_command("/opt/robotsnap-unity/player") == [
        "/opt/robotsnap-unity/player"
    ]


def test_player_command_headless():
    assert unity_app.player_command("player", headless=True) == [
        "player",
        "-batchmode",
        "-nographics",
    ]


def test_player_command_headless_with_a_log():
    assert unity_app.player_command("player", headless=True, log_path="/tmp/u.log") == [
        "player",
        "-batchmode",
        "-nographics",
        "-logFile",
        "/tmp/u.log",
    ]


def test_player_command_windowed_with_a_log():
    assert unity_app.player_command("player", log_path="/tmp/u.log") == [
        "player",
        "-logFile",
        "/tmp/u.log",
    ]


def test_player_data_folder_reads_an_installed_layout(tmp_path):
    """A player keeps its assets in the folder named after it."""
    install = tmp_path / "install"
    player = _make_player(install)
    data = install / "robotsnap-unity_Data"
    data.mkdir()

    assert unity_app.player_data_folder(player) == data
    assert unity_app.player_data_folder(tmp_path / "absent" / player.name) is None


def test_player_data_folder_reads_a_macos_bundle(tmp_path):
    """A ``.app`` hides the same assets under ``Contents/Resources/Data``."""
    player = tmp_path / "robotsnap-unity.app" / "Contents" / "MacOS" / "robotsnap-unity"
    player.parent.mkdir(parents=True)
    player.write_text("#!/bin/sh\n")
    player.chmod(0o755)
    data = player.parents[1] / "Resources" / "Data"
    data.mkdir(parents=True)

    assert unity_app.player_data_folder(player) == data


# -- the scenarios a launch hands the player --------------------------------


def _capture_start(monkeypatch) -> dict:
    """Stand in for ``Popen`` and record what :meth:`start` passes it."""
    captured: dict = {}

    class FakeProcess:
        pid = 4242

        def poll(self):
            return None

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs.get("env")
        return FakeProcess()

    monkeypatch.setattr(unity_app.subprocess, "Popen", fake_popen)
    return captured


def test_start_exports_the_projects_scenarios(monkeypatch, tmp_path, clean_environment):
    """A workspace clone names its live scenarios to the player it starts."""
    scenarios = (
        tmp_path / "robotsnap-unity" / "Assets" / "StreamingAssets" / "Scenarios"
    )
    scenarios.mkdir(parents=True)
    monkeypatch.setenv(unity_app.HOME_ENV, str(tmp_path))
    monkeypatch.setenv("ROBOTSNAP_TEST_MARKER", "kept")

    captured = _capture_start(monkeypatch)
    unity_app.UnityApplication("player").start()

    assert captured["env"][unity_app.SCENARIOS_ENV] == str(scenarios.resolve())
    # Only the one variable is added: the rest of the environment still reaches
    # the player, so a PATH or a HOME the run needs is not dropped.
    assert captured["env"]["ROBOTSNAP_TEST_MARKER"] == "kept"


def test_start_leaves_the_environment_alone_without_a_project(
    monkeypatch, tmp_path, clean_environment
):
    """A package install with no checkout beside it exports nothing."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    captured = _capture_start(monkeypatch)
    unity_app.UnityApplication("player").start()

    assert captured["env"] is None


# -- the process lifecycle --------------------------------------------------


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@pytest.mark.skipif(
    sys.platform.startswith("win"), reason="process groups are POSIX"
)
def test_unity_application_starts_and_stops(tmp_path):
    player = _make_player(tmp_path)
    application = unity_app.UnityApplication(player)

    assert application.pid is None
    assert application.running is False

    application.start()
    pid = application.pid
    assert pid is not None
    assert application.running is True
    assert _alive(pid) is True

    assert application.stop() is True
    assert application.pid is None
    assert application.running is False
    assert _alive(pid) is False

    # Stopping twice is stopped, not an error.
    assert application.stop() is True
