"""The RobotSNAP Unity player as a process: find it, start it, stop it.

A run does not need Unity to be opened by hand: the player ships as a package
of its own (``robotsnap-unity``) and :func:`find_player` locates it on this
machine, so one command can open the application, drive it and close it again::

    from robotsnap.unity_app import UnityApplication, find_player

    app = UnityApplication(find_player(), headless=True, log_path="/tmp/unity.log")
    app.start()
    ...
    app.stop()

Nothing here prints: the command line (:mod:`robotsnap.cli.launch_unity`) is
what reports the player it started, and a library that spoke on every start
would be noise inside a run that owns its own output. The discovery order is
what saves a machine with a single install from naming it: the explicit path
wins, then the environment, then the workspace installer's layout and the
usual install prefixes, and the source project's build is the last resort.

Scenarios follow the same split: a packaged player reads the copy baked into
its own data folder, so :meth:`UnityApplication.start` names the project's
``Assets/StreamingAssets/Scenarios`` to it through :data:`SCENARIOS_ENV` when a
workspace clone holds one, and leaves the environment untouched when none does.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

__all__ = [
    "APP_ENV",
    "HOME_ENV",
    "MISSING_MESSAGE",
    "PLAYER_NAMES",
    "SCENARIOS_ENV",
    "UnityAppMissing",
    "UnityApplication",
    "find_player",
    "player_data_folder",
    "player_command",
    "player_name",
]

#: Environment variable that names the player, or a directory holding it.
APP_ENV = "ROBOTSNAP_UNITY_APP"
#: Environment variable that names the workspace root the installer wrote to.
HOME_ENV = "ROBOTSNAP_HOME"
#: Environment variable that names a folder of scenarios the player reads first.
SCENARIOS_ENV = "ROBOTSNAP_SCENARIOS_DIR"

#: The workspace clone's own name, beside the home directory.
WORKSPACE_FOLDER = "robotsnap-workspace"
#: The Unity project's folder inside a workspace clone.
PROJECT_FOLDER = "robotsnap-unity"

#: The player's file name, keyed by the family of :data:`sys.platform`.
PLAYER_NAMES: dict[str, str] = {
    "linux": "robotsnap-unity.x86_64",
    "darwin": "robotsnap-unity",
    "win32": "robotsnap-unity.exe",
}

#: The ``Builds/`` folder of the Unity project, keyed by the same families.
_BUILD_FOLDERS: dict[str, str] = {
    "linux": "Linux",
    "darwin": "macOS",
    "win32": "Windows",
}

#: Per-user install prefixes, relative to the home directory, in probe order.
#: The first is the versioned prefix the ``.run`` installer writes; the second
#: is the launcher it drops on ``$PATH``; the last two are plain checkouts.
_USER_PREFIXES: tuple[str, ...] = (
    os.path.join(".local", "share", "robotsnap-unity"),
    os.path.join(".local", "bin", "robotsnap-unity"),
    os.path.join(".local", "opt", "robotsnap-unity"),
    "robotsnap-unity",
)

#: System-wide install prefixes, probed after the per-user ones.
_SYSTEM_PREFIXES: tuple[str, ...] = (
    os.path.join(os.sep, "opt", "robotsnap-unity"),
    os.path.join(os.sep, "usr", "local", "lib", "robotsnap-unity"),
)

#: What to tell a caller who asked for a player and none was found.
MISSING_MESSAGE = (
    "no RobotSNAP player found: looked in --unity-app, $ROBOTSNAP_UNITY_APP, "
    "$ROBOTSNAP_HOME/app/robotsnap-unity, "
    "~/robotsnap-workspace/app/robotsnap-unity, ~/.local/share/robotsnap-unity, "
    "~/.local/bin/robotsnap-unity, ~/.local/opt/robotsnap-unity, "
    "~/robotsnap-unity, /opt/robotsnap-unity, /usr/local/lib/robotsnap-unity, "
    "$PATH and the Unity project's Builds. Point at one with --unity-app "
    "/path/to/robotsnap-unity.x86_64, or install one with setup-workspace.sh."
)


class UnityAppMissing(RuntimeError):
    """No RobotSNAP player was found, and a caller asked for one anyway.

    :func:`find_player` answers ``None`` rather than raising, so a caller can
    decide what to do; this is the error a caller raises when it cannot go on
    without a player. The default message names the places that were searched
    and both ways to point at one.
    """

    def __init__(self, message: str | None = None) -> None:
        super().__init__(MISSING_MESSAGE if message is None else message)


def player_name(platform: str | None = None) -> str:
    """The player's file name on ``platform``, this machine's by default."""
    return PLAYER_NAMES[_platform_key(platform)]


def find_player(
    explicit: str | os.PathLike[str] | None = None,
    *,
    root: str | os.PathLike[str] | None = None,
) -> Path | None:
    """The player to run, or ``None`` when this machine has none to run.

    Candidates are tried in order and the first one that is a file this
    process may execute wins, so a machine with several installs gets the most
    specific one: the installer's prefixes, then ``$PATH``, and finally the
    Unity project's own ``Builds/``.

    A path - explicit or from the environment - may name the player itself or
    a directory that holds it, including the ``app/robotsnap-unity/<version>``
    folders the workspace installer writes. Such a path is an instruction and
    not a hint: when it names no player the search stops there, so a typo on
    ``--unity-app`` cannot quietly run another install.

    Without one, the workspace layouts are read first - ``root``, or
    ``$ROBOTSNAP_HOME`` when the caller gave none, then ``~/robotsnap-workspace``
    and the home directory. Nothing here raises for a missing project: a build
    that was never made simply has no candidate.
    """
    for value in (explicit, os.environ.get(APP_ENV)):
        if value:
            return _from_value(value)

    for directory in _workspace_roots(root):
        found = _player_in(directory)
        if found is not None:
            return found

    home = Path.home()
    for prefix in _USER_PREFIXES:
        found = _from_value(home / prefix)
        if found is not None:
            return found
    for prefix in _SYSTEM_PREFIXES:
        found = _from_value(prefix)
        if found is not None:
            return found

    on_path = shutil.which("robotsnap-unity")
    if on_path:
        candidate = Path(on_path)
        if _runnable(candidate):
            return candidate

    return _source_build_player()


def player_command(
    player: str | os.PathLike[str],
    *,
    headless: bool = False,
    log_path: str | os.PathLike[str] | None = None,
) -> list[str]:
    """The argv that runs ``player``, windowed or headless.

    ``headless`` asks Unity for ``-batchmode -nographics``: no window and no
    swap of a back buffer, which is where a long run's speed comes from.
    ``log_path`` makes the player write its log there instead of Unity's own
    default location, and stands on its own so a windowed run can keep one too.
    """
    command = [str(player)]
    if headless:
        command += ["-batchmode", "-nographics"]
    if log_path is not None:
        command += ["-logFile", str(log_path)]
    return command


def player_data_folder(player: str | os.PathLike[str]) -> Path | None:
    """The data folder of a packaged player, or ``None`` when it has none.

    A built player keeps its assets beside the executable, in a folder named
    after it: ``robotsnap-unity_Data`` on Linux and Windows, and
    ``Contents/Resources/Data`` inside the ``.app`` on macOS. That folder holds
    the ``StreamingAssets`` the application reads its scenarios and maps from,
    which is what a run has to be pointed at when the application is installed
    rather than opened as a project.
    """
    path = _as_path(player)
    directory = path.parent
    candidates = [directory / f"{path.stem}_Data"]
    contents = directory.parent
    if contents.name == "Contents":
        candidates.append(contents / "Resources" / "Data")

    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


class UnityApplication:
    """A player process, started on its own so its whole group can be stopped."""

    def __init__(
        self,
        player: str | os.PathLike[str],
        *,
        headless: bool = False,
        log_path: str | os.PathLike[str] | None = None,
    ) -> None:
        self._player = _as_path(player)
        self._headless = headless
        self._log_path = None if log_path is None else _as_path(log_path)
        self._process: subprocess.Popen | None = None

    @property
    def player(self) -> Path:
        """The player file this application runs."""
        return self._player

    @property
    def pid(self) -> int | None:
        """The player's process id, before a start and after a stop ``None``."""
        return None if self._process is None else self._process.pid

    @property
    def log_path(self) -> Path | None:
        """Where the player writes its log, or ``None`` for Unity's own default."""
        return self._log_path

    @property
    def running(self) -> bool:
        """Whether the player process is still up."""
        return self._process is not None and self._process.poll() is None

    def start(self) -> None:
        """Spawn the player in a session of its own, mute on out and err.

        The session is what lets :meth:`stop` end the whole tree Unity starts -
        the render threads and helpers - rather than only the wrapper, and
        ``DEVNULL`` keeps the player's chatter out of the run's own output; a
        caller that wants the log passes ``log_path``. Already running, this is
        a no-op, so a second ``start`` cannot leak a second player.

        The environment is this process's own, plus :data:`SCENARIOS_ENV` when a
        workspace clone holds the project's scenarios, so a scenario edited in
        the project is the copy this player reads rather than the one baked into
        its data folder. Without the project, the player inherits the
        environment untouched.
        """
        if self.running:
            return
        self._process = subprocess.Popen(
            player_command(
                self._player, headless=self._headless, log_path=self._log_path
            ),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            env=_launch_environment(),
        )

    def stop(self, timeout: float = 10.0) -> bool:
        """End the player, gently then not, answering whether it is gone.

        SIGTERM goes to the process group first, so Unity shuts its scene down
        and leaves the simulation in standby; a player that ignores it for
        ``timeout`` seconds then gets SIGKILL. Idempotent: a player that never
        started, or is already gone, is already stopped, and nothing is raised
        for either.
        """
        process = self._process
        if process is None:
            return True
        if process.poll() is not None:
            self._process = None
            return True

        _kill_group(process.pid, signal.SIGTERM)
        if _wait(process, timeout):
            self._process = None
            return True

        _kill_group(process.pid, signal.SIGKILL)
        if _wait(process, timeout):
            self._process = None
            return True
        return False


# -- the discovery machinery ------------------------------------------------


def _platform_key(platform: str | None = None) -> str:
    """The :data:`PLAYER_NAMES` key for ``platform`` (``sys.platform`` here)."""
    platform = sys.platform if platform is None else platform
    if platform.startswith("win"):
        return "win32"
    if platform == "darwin":
        return "darwin"
    return "linux"


def _as_path(value: str | os.PathLike[str]) -> Path:
    """``value`` as a path, both separators accepted and ``~`` expanded.

    A Windows caller writes ``C:\\...`` as often as ``C:/...``; ``normpath``
    settles the two into the spelling the platform probes, and the home
    shorthand is expanded at the same time.
    """
    return Path(os.path.normpath(os.fspath(value))).expanduser()


def _runnable(path: Path) -> bool:
    """Whether ``path`` is a file this process may execute."""
    return path.is_file() and os.access(path, os.X_OK)


def _from_value(value: str | os.PathLike[str]) -> Path | None:
    """The player ``value`` names, whether it is the player or a folder of it."""
    path = _as_path(value)
    if _runnable(path):
        return path
    if path.is_dir():
        return _player_in(path)
    return None


def _workspace_roots(root: str | os.PathLike[str] | None) -> Iterator[Path]:
    """Where an installed workspace may be, the most specific one first.

    ``root`` - or ``$ROBOTSNAP_HOME`` when the caller gave none - is the layout
    an explicit install asked for. The clone the project's own setup script
    writes into is beside the home directory
    (``~/robotsnap-workspace/app/robotsnap-unity/<version>``), and the home
    directory itself is the last place such a tree can hide.
    """
    explicit = root if root is not None else os.environ.get(HOME_ENV)
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(_as_path(explicit))
    candidates.append(Path.home() / WORKSPACE_FOLDER)
    candidates.append(Path.home())

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            yield candidate


def _project_scenarios_dir() -> Path | None:
    """The Unity project's scenarios in a workspace clone, or ``None``.

    The target is ``<workspace>/robotsnap-unity/Assets/StreamingAssets/
    Scenarios``: the folder a run has to name to the player so an edit there is
    the copy it plays. The workspace roots are the ones :func:`find_player`
    already probes, and a machine without the project - a package install with
    no checkout beside it - answers ``None``.
    """
    for directory in _workspace_roots(None):
        candidate = directory.joinpath(
            PROJECT_FOLDER, "Assets", "StreamingAssets", "Scenarios"
        )
        if candidate.is_dir():
            return candidate.resolve()
    return None


def _launch_environment() -> dict[str, str] | None:
    """The environment to start a player with, or ``None`` to inherit ours.

    ``None`` means nothing to add: the child keeps this process's environment,
    which is what a package install without a project beside it has always
    done. Otherwise a copy of it carries :data:`SCENARIOS_ENV`, an absolute
    path, which the player reads before the copy baked into its data folder.
    """
    scenarios = _project_scenarios_dir()
    if scenarios is None:
        return None
    environment = dict(os.environ)
    environment[SCENARIOS_ENV] = str(scenarios)
    return environment


def _bundle_candidates(directory: Path) -> Iterator[Path]:
    """Where the player sits inside ``directory``, app bundle included.

    Linux and Windows put the player straight in the directory; macOS wraps it
    in ``robotsnap-unity.app``. Both spellings are offered and the one that
    exists and is executable wins.
    """
    name = player_name()
    yield directory / name
    if _platform_key() == "darwin":
        yield directory / "robotsnap-unity.app" / "Contents" / "MacOS" / name


def _version_folders(directory: Path) -> Iterator[Path]:
    """The sub-directories of ``directory`` a versioned install may use.

    A versioned install is one directory per release, either straight under the
    prefix (``~/.local/share/robotsnap-unity/<version>/``) or under the
    workspace layout's ``app/robotsnap-unity/<version>/``.
    """
    seen: set[Path] = set()
    for pattern in ("*", os.path.join("app", "robotsnap-unity", "*")):
        for entry in sorted(directory.glob(pattern)):
            if entry.is_dir() and entry not in seen:
                seen.add(entry)
                yield entry


def _player_in(directory: Path) -> Path | None:
    """The player inside ``directory``, or one version folder down."""
    for candidate in _bundle_candidates(directory):
        if _runnable(candidate):
            return candidate
    for folder in _version_folders(directory):
        for candidate in _bundle_candidates(folder):
            if _runnable(candidate):
                return candidate
    return None


def _source_build_player() -> Path | None:
    """The player a local Unity build would have made, or ``None``.

    The last resort: a checkout that was built but never packaged. The project
    is wherever :func:`robotsnap.scenario.unity_project` says it is, and a
    machine without the project - or without a build of it - has no candidate.
    """
    from robotsnap import scenario

    try:
        project = scenario.unity_project()
    except OSError:
        return None
    folder = project / "Builds" / _BUILD_FOLDERS[_platform_key()]
    return _from_value(folder)


def _kill_group(pid: int, sig: int) -> None:
    """Send ``sig`` to the process group ``pid`` leads, ignoring a dead group."""
    try:
        os.killpg(os.getpgid(pid), sig)
    except OSError:
        # The group is already gone, or is not ours to signal: nothing to do.
        pass


def _wait(process: subprocess.Popen, timeout: float) -> bool:
    """Wait ``timeout`` seconds for ``process``, answering whether it exited."""
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        return False
    return True
