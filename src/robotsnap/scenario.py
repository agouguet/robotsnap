"""Author the scenario files the Unity project loads, and find where they live.

Unity reads a scenario from a YAML file of ``StreamingAssets/Scenarios`` and
names it after that file, so a scenario is created by writing one of those files
and then asking the session to load it::

    from robotsnap import scenario

    document = scenario.build(
        "python_demo",
        map_name="basic/crowd",
        robots=[scenario.robot("robot_1", "jackal", (2.0, -6.0, 90), (20.0, -6.0))],
        humans=[scenario.crowd("pedestrians", 6,
                               spawn=scenario.spawn_zone(11.0, 0.0, 6.0, 6.0),
                               goal=scenario.spawn_zone(11.0, 0.0, 6.0, 6.0))],
    )
    path = scenario.write(document)          # <project>/Assets/StreamingAssets/Scenarios/python_demo.yaml

Nothing here talks to a running session. :class:`robotsnap.client.RobotSNAPClient`
is what sends the ``load_scenario`` command once the file is on disk, and it
delegates its own ``create_scenario`` to this module.

Frames: the points and zones below are in the **Unity world** axes the scenario
editor writes, ``x`` and ``z`` on the ground plane and ``y`` up, which is *not*
the ROS frame of the published streams. A yaw is in degrees here, as the file
stores it, and Unity turns it into a heading around its own up axis.

The YAML is emitted by this module rather than by a parser: the package keeps
``rosbags`` as its only install requirement, and the documents a scenario needs
are flat enough to write directly. The test suite parses what this writes with a
real YAML reader, so the emitter is checked against a parser and not against
itself.
"""

from __future__ import annotations

import datetime as _datetime
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

__all__ = [
    "UNITY_PROJECT_ENV",
    "SCENARIOS_FOLDER",
    "ROBOT_TYPES",
    "unity_project",
    "scenarios_dir",
    "point",
    "zone",
    "spawn_point",
    "spawn_zone",
    "goal_point",
    "goal_ref",
    "value_range",
    "read_range",
    "robot",
    "crowd",
    "build",
    "to_yaml",
    "write",
    "find",
    "list_names",
    "delete",
]

#: Environment variable that names the Unity project when ``path`` is not given.
UNITY_PROJECT_ENV = "ROBOTSNAP_UNITY_PROJECT"
#: Folder under ``StreamingAssets``, the default of ``SimulationConfig.ScenariosFolder``.
SCENARIOS_FOLDER = "Scenarios"
#: The type ids of ``Assets/Resources/RobotCatalog.asset``, for the ``type`` of a robot.
ROBOT_TYPES = ("bibus", "freight", "ginger", "jackal", "kuri")

#: Last candidate tried when nothing else is given: the checkout beside the home.
#: Where this machine's checkout usually is: the workspace layout first, then a
#: checkout directly under the home directory. A candidate that does not hold
#: ``Assets/StreamingAssets`` is skipped, so a stale one costs nothing.
_FALLBACK_PROJECTS: tuple[Path, ...] = (
    Path.home() / "robotsnap-workspace" / "robotsnap-unity",
    Path.home() / "robotsnap-unity",
)

_SCENARIO_EXTENSIONS = (".yaml", ".yml")


# -- locating the Unity project ---------------------------------------------


def unity_project(path: str | os.PathLike[str] | None = None) -> Path:
    """Return the directory holding the ``StreamingAssets`` a session reads.

    The order is: an explicit ``path``, then ``$ROBOTSNAP_UNITY_PROJECT``, then
    the current working directory and this machine's checkout. A candidate
    counts when it carries ``StreamingAssets``, which is what says a directory
    is one the application reads rather than a directory named like one, and
    there are two layouts to carry it: the source project's
    ``Assets/StreamingAssets``, and the data folder of an installed application,
    which holds ``StreamingAssets`` itself.
    """
    candidates: list[Path] = []
    if path is not None:
        candidates.append(Path(path))
    if os.environ.get(UNITY_PROJECT_ENV):
        candidates.append(Path(os.environ[UNITY_PROJECT_ENV]))
    candidates.append(Path.cwd())
    candidates.extend(_FALLBACK_PROJECTS)

    for candidate in candidates:
        project = Path(candidate).expanduser()
        if _streaming_assets(project) is not None:
            return project.resolve()

    raise FileNotFoundError(
        "no Unity project found: pass one, or set "
        f"{UNITY_PROJECT_ENV} to a directory holding StreamingAssets"
    )


def scenarios_dir(
    project: str | os.PathLike[str] | None = None,
    folder: str = SCENARIOS_FOLDER,
) -> Path:
    """Return the directory a scenario file has to be written to."""
    streaming = _streaming_assets(unity_project(project))
    # ``unity_project`` only answers a directory that carries one.
    assert streaming is not None
    return streaming / folder


def _streaming_assets(project: Path) -> Path | None:
    """The ``StreamingAssets`` folder of ``project``, in either layout.

    A source project keeps it under ``Assets/``; the data folder of a packaged
    player - what an installed RobotSNAP application ships beside its own
    executable - holds it directly. The application reads its scenarios and
    maps from that folder whichever it is, so both count as a project here.
    """
    for relative in (("Assets", "StreamingAssets"), ("StreamingAssets",)):
        folder = project.joinpath(*relative)
        if folder.is_dir():
            return folder
    return None


# -- authoring one document --------------------------------------------------


def point(x: float, z: float, y: float = 0.0, yaw: float | None = None) -> dict[str, Any]:
    """A named point of the ``points`` section, in Unity world axes.

    ``x`` and ``z`` are the ground-plane coordinates and ``y`` the height, the
    three the editor writes. ``yaw`` is in degrees, or ``None`` for a point the
    scenario does not orient.
    """
    return {
        "x": float(x),
        "y": float(y),
        "z": float(z),
        "yaw": None if yaw is None else float(yaw),
        "center": None,
        "size": None,
    }


def zone(
    center_x: float,
    center_z: float,
    size_x: float,
    size_z: float,
    y: float = 0.0,
    yaw: float | None = None,
) -> dict[str, Any]:
    """A named area, given by its centre and its extent, in Unity world axes.

    A zone is the other shape a named point can take, and it is what a random
    spawn or a random goal draws from. ``size_x`` and ``size_z`` are full
    extents, not half extents, and ``y`` is the height of the centre.
    """
    return {
        "x": None,
        "y": None,
        "z": None,
        "yaw": None if yaw is None else float(yaw),
        "center": {"x": float(center_x), "y": float(y), "z": float(center_z)},
        "size": {"x": float(size_x), "y": 0.0, "z": float(size_z)},
    }


def spawn_point(
    x: float,
    z: float,
    *,
    y: float = 0.0,
    formation: str | None = None,
    spacing: float = 0.8,
    spacing_range: Any = None,
    relative_to: str | None = None,
) -> dict[str, Any]:
    """A spawn at one place: every agent of the route starts there.

    ``formation`` lays the members out around that place rather than on it, one
    of the names ``SpawnPlanner`` knows (``pair``, ``row``, ``wedge``, ...);
    ``None`` leaves the default. ``spacing`` is the metres between two members,
    and ``spacing_range`` lets the run draw it instead.
    """
    spawn = {
        "type": "point",
        "ref": None,
        "position": {"x": float(x), "y": float(y), "z": float(z)},
        "zone": None,
        "formation": formation,
        "formation_parameter": 0.0,
        "spacing": float(spacing),
        "relative_to": relative_to,
    }
    if spacing_range is not None:
        spawn["spacing_range"] = _ranged(spacing_range)
    return spawn


def spawn_zone(
    center_x: float,
    center_z: float,
    size_x: float,
    size_z: float,
    *,
    y: float = 0.0,
    formation: str = "scatter",
    spacing: float = 1.5,
    spacing_range: Any = None,
    relative_to: str | None = None,
) -> dict[str, Any]:
    """A spawn drawn at random inside an area, which is what a crowd wants.

    ``spacing_range`` lets the run draw the distance kept between the agents
    instead of using ``spacing``.
    """
    spawn = {
        "type": "random",
        "ref": None,
        "position": None,
        "zone": zone(center_x, center_z, size_x, size_z, y),
        "formation": formation,
        "formation_parameter": 0.0,
        "spacing": float(spacing),
        "relative_to": relative_to,
    }
    if spacing_range is not None:
        spawn["spacing_range"] = _ranged(spacing_range)
    return spawn


def goal_point(
    x: float, z: float, *, y: float = 0.0, radius: float = 0.0
) -> dict[str, Any]:
    """A goal at one place. An agent that reaches within ``radius`` is done."""
    return {
        "type": "point",
        "ref": None,
        "position": {"x": float(x), "y": float(y), "z": float(z)},
        "zone": None,
        "target": None,
        "radius": float(radius),
    }


def goal_ref(ref: str, *, radius: float = 0.0) -> dict[str, Any]:
    """A goal that names an entry of ``points``, instead of a bare position."""
    return {
        "type": "point",
        "ref": str(ref),
        "position": None,
        "zone": None,
        "target": None,
        "radius": float(radius),
    }


def value_range(minimum: float, maximum: float) -> dict[str, float]:
    """A value the run draws between two bounds instead of using a fixed one.

    A range is written beside the value it replaces, under a ``*_range`` sibling:
    ``count`` and ``count_range``, ``speed`` and ``speed_range``, and so on. The
    fixed value stays in the document as the nominal one - what a reader that
    predates ranges sees, and what the scenario editor shows when the author
    switches back to a fixed value - while the run draws from the range when one
    is present. A scenario that names no range draws nothing, so a file written
    before ranges existed still runs exactly as it did.

    The two bounds are swapped if they arrive the wrong way round, so a caller
    cannot write a range the runtime would have to refuse.
    """
    low = float(minimum)
    high = float(maximum)
    if high < low:
        low, high = high, low
    return {"min": low, "max": high}


def read_range(value: Any) -> tuple[float, float] | None:
    """Read one scenario value back as a ``(low, high)`` pair.

    Three shapes are accepted, which are the three a scenario can carry: the
    mapping :func:`value_range` writes, a plain ``(min, max)`` pair, and a scalar,
    which reads back as the degenerate pair a fixed value is. ``None`` reads back
    as ``None``, meaning the scenario leaves that value alone.
    """
    if value is None:
        return None
    if isinstance(value, Mapping):
        low = value.get("min", value.get("max"))
        high = value.get("max", value.get("min"))
        if low is None or high is None:
            return None
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        low, high = value
    else:
        try:
            scalar = float(value)
        except (TypeError, ValueError):
            return None
        return (scalar, scalar)

    try:
        low_value = float(low)
        high_value = float(high)
    except (TypeError, ValueError):
        return None
    return (min(low_value, high_value), max(low_value, high_value))


def _ranged(value: Any) -> dict[str, float]:
    """Normalise the optional ``*_range`` argument of a builder into a mapping."""
    if isinstance(value, Mapping):
        low = value.get("min")
        high = value.get("max")
        if low is None or high is None:
            raise TypeError(
                f"a range needs both a min and a max, not {value!r}"
            )
        return value_range(low, high)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return value_range(value[0], value[1])
    raise TypeError(
        f"a range must be value_range(min, max) or an (min, max) pair, not {value!r}"
    )


def _reference(value: Any, fallback_name: str, points: dict[str, Any]) -> str:
    """Return the name a robot field refers to, naming a bare point if given one.

    A field is either the name of an entry of ``points``, a mapping from
    :func:`point`, or a plain ``(x, z)`` or ``(x, z, yaw)`` pair, which is
    turned into a point. Anything else is a mistake worth reporting: a start
    that silently became a string would produce a scenario Unity refuses, far
    from the line that caused it.
    """
    if isinstance(value, Mapping):
        points.setdefault(fallback_name, dict(value))
        return fallback_name
    if isinstance(value, (list, tuple)) and len(value) in (2, 3):
        yaw = value[2] if len(value) == 3 else None
        points.setdefault(fallback_name, point(value[0], value[1], yaw=yaw))
        return fallback_name
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise TypeError(
        f"a start, goal or waypoint must be a point name, a point or an (x, z) pair, not {value!r}"
    )


def robot(
    robot_id: str,
    robot_type: str,
    start: Any,
    goal: Any,
    *,
    waypoints: Sequence[Any] | None = None,
    behavior: str = "normal",
    speed: float = 1.0,
    speed_range: Any = None,
    start_yaw_range: Any = None,
) -> dict[str, Any]:
    """One entry of the ``robots`` list of a scenario.

    ``start`` and ``goal`` name an entry of ``points``; given a mapping from
    :func:`point` instead, :func:`build` names it ``<robot_id>_start`` and
    ``<robot_id>_goal`` for you. ``waypoints`` are intermediate references
    walked in order before the goal. ``speed`` is in metres per second and is
    what the scenario's own planner drives at; the ROS controller has its own
    limit.

    ``robot_type`` is one of :data:`ROBOT_TYPES`, the ids of the project's
    robot catalog; an unknown one leaves the robot unspawned.

    ``speed_range`` and ``start_yaw_range`` let the run draw the robot's speed
    and its heading at spawn between two bounds instead of using ``speed`` and
    the yaw of its start point, the two being :func:`value_range` or a
    ``(min, max)`` pair. ``None`` - the default - keeps the robot fixed.
    """
    entry = {
        "id": str(robot_id),
        "type": str(robot_type),
        "start": start,
        "goal": goal,
        "waypoints": [point_ref for point_ref in waypoints] if waypoints else None,
        "behavior": str(behavior),
        "speed": float(speed),
    }
    if speed_range is not None:
        entry["speed_range"] = _ranged(speed_range)
    if start_yaw_range is not None:
        entry["start_yaw_range"] = _ranged(start_yaw_range)
    return entry


def crowd(
    crowd_id: str,
    count: int,
    *,
    spawn: Mapping[str, Any],
    goal: Mapping[str, Any],
    speed: float = 1.0,
    end_behavior: str = "stay",
    group: str | None = None,
    controller: str = "SFM",
    behavior: str = "normal",
    spawn_window: float = 0.0,
    goals: Sequence[Mapping[str, Any]] | None = None,
    count_range: Any = None,
    speed_range: Any = None,
    spawn_window_range: Any = None,
) -> dict[str, Any]:
    """One entry of the ``humans`` list: a route a number of agents walk.

    ``spawn`` and ``goal`` come from :func:`spawn_point`, :func:`spawn_zone`,
    :func:`goal_point` or :func:`goal_ref`. ``end_behavior`` is what the agents
    do once the route runs out: ``stay``, ``disappear`` or ``loop``.
    ``spawn_window`` is the number of seconds over which they enter after the
    scenario is applied, zero releasing them all at once. ``group`` makes every
    entry sharing it walk together.

    Three values may be drawn by the run instead of fixed: ``count_range`` for
    how many agents the route carries, ``speed_range`` for their walking speed
    and ``spawn_window_range`` for the entry window. Each is a
    :func:`value_range` or a ``(min, max)`` pair, and ``None`` keeps the route
    fixed. The count is the one a study randomizes most often, and both of its
    bounds are included in the draw.
    """
    route = {
        "id": str(crowd_id),
        "count": int(count),
        "spawn_window": float(spawn_window),
        "spawn": dict(spawn),
        "goal": dict(goal),
        "goals": list(goals) if goals else None,
        "end_behavior": str(end_behavior),
        "group": None if group is None else str(group),
        "movement_controller": {"type": str(controller), "sfm_params": None},
        "behavior": str(behavior),
        "speed": float(speed),
        "color": None,
        "personality": None,
    }
    if count_range is not None:
        route["count_range"] = _ranged(count_range)
    if speed_range is not None:
        route["speed_range"] = _ranged(speed_range)
    if spawn_window_range is not None:
        route["spawn_window_range"] = _ranged(spawn_window_range)
    return route


def build(
    name: str,
    *,
    map_name: str,
    robots: Sequence[Mapping[str, Any]],
    humans: Sequence[Mapping[str, Any]] = (),
    points: Mapping[str, Any] | None = None,
    description: str = "",
    tags: Sequence[str] = (),
    robot_type: str | None = None,
    author: str = "RobotSNAP",
    location: str = "",
    preview: str = "",
    duration: float = 0.0,
    type: str = "Custom",
    created: str | None = None,
) -> dict[str, Any]:
    """The document of a whole scenario, ready for :func:`write`.

    ``map_name`` is what the ``map`` key of the file holds, either a specific
    map such as ``basic/crowd`` or the name of a collection such as ``basic``,
    which draws a random map of it. ``points`` names the places the robots
    refer to; a robot that was given a mapping for its start or its goal adds
    the matching entry itself, under ``<robot_id>_start`` and
    ``<robot_id>_goal``.

    A scenario is refused by the simulator when it has no robot, or when a
    robot has no start or no goal, so those are checked here as well - writing
    a file the simulator will reject helps nobody.
    """
    if not str(name).strip():
        raise ValueError("a scenario needs a name")
    if not robots:
        raise ValueError("a scenario needs at least one robot")

    named_points: dict[str, Any] = dict(points or {})
    robot_entries: list[dict[str, Any]] = []
    for entry in robots:
        built = dict(entry)
        robot_id = str(built.get("id") or f"robot_{len(robot_entries) + 1}")
        built["id"] = robot_id
        if not built.get("start"):
            raise ValueError(f"robot {robot_id!r} has no start")
        if not built.get("goal"):
            raise ValueError(f"robot {robot_id!r} has no goal")
        built["start"] = _reference(built["start"], f"{robot_id}_start", named_points)
        built["goal"] = _reference(built["goal"], f"{robot_id}_goal", named_points)
        waypoints = built.get("waypoints")
        if waypoints:
            built["waypoints"] = [
                _reference(entry_ref, f"{robot_id}_waypoint_{index}", named_points)
                for index, entry_ref in enumerate(waypoints)
            ]
        robot_entries.append(built)

    document: dict[str, Any] = {
        "scenario_info": {
            "name": str(name),
            "type": str(type),
            "description": str(description),
            "version": "1.0",
            "author": str(author),
            "created": created or _datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "tags": [str(tag) for tag in tags],
            "map": str(map_name),
            "location": str(location),
            "dataset": None,
            "preview": str(preview),
            "robot_type": str(
                robot_type
                if robot_type is not None
                else robot_entries[0].get("type") or "TurtleBot4"
            ),
            "duration": float(duration),
        },
        "points": named_points,
        "robots": robot_entries,
        "robot": None,
        "humans": [dict(entry) for entry in humans],
    }
    return document


# -- writing the file --------------------------------------------------------


_BARE_SCALAR = re.compile(r"[A-Za-z_][A-Za-z0-9_./-]*\Z")
_RESERVED = {"null", "true", "false", "yes", "no", "on", "off", "y", "n", "~"}


def _scalar(value: Any) -> str:
    """One YAML scalar, quoted whenever a reader could take it for something else."""
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    text = str(value)
    if _BARE_SCALAR.match(text) and text.lower() not in _RESERVED:
        return text
    return "'" + text.replace("'", "''") + "'"


def _inline(value: Any) -> str | None:
    """How ``value`` reads on the line of its key, or ``None`` when it needs a block."""
    if isinstance(value, Mapping):
        return "{}" if not value else None
    if isinstance(value, (list, tuple)):
        return "[]" if not value else None
    return _scalar(value)


def _emit_pair(prefix: str, value: Any, indent: int, lines: list[str]) -> None:
    """Append ``prefix:`` and everything under it, nested when it needs to be."""
    text = _inline(value)
    if text is not None:
        lines.append(f"{prefix}: {text}")
        return
    lines.append(f"{prefix}:")
    _emit(value, indent + 2, lines)


def _emit(node: Any, indent: int, lines: list[str]) -> None:
    """Append ``node`` as block YAML, indented by ``indent`` spaces."""
    pad = " " * indent
    if isinstance(node, Mapping):
        for key, value in node.items():
            _emit_pair(f"{pad}{key}", value, indent, lines)
        return
    if isinstance(node, (list, tuple)):
        for item in node:
            if isinstance(item, Mapping) and item:
                # The first key rides on the dash, so a list of mappings reads
                # the way the scenarios Unity writes already do.
                pairs = list(item.items())
                first_key, first_value = pairs[0]
                text = _inline(first_value)
                if text is not None:
                    lines.append(f"{pad}- {first_key}: {text}")
                else:
                    lines.append(f"{pad}- {first_key}:")
                    _emit(first_value, indent + 4, lines)
                for key, value in pairs[1:]:
                    _emit_pair(f"{pad}  {key}", value, indent + 2, lines)
            elif isinstance(item, (list, tuple)) and item:
                lines.append(f"{pad}-")
                _emit(item, indent + 2, lines)
            else:
                lines.append(f"{pad}- {_inline(item)}")
        return
    lines.append(f"{pad}{_inline(node)}")


def to_yaml(document: Mapping[str, Any]) -> str:
    """Serialise a document built by :func:`build` to scenario YAML."""
    lines: list[str] = []
    _emit(dict(document), 0, lines)
    return "\n".join(lines) + "\n"


def _file_stem(name: str) -> str:
    """The name a scenario is loaded under: the stem of its file."""
    text = str(name).strip()
    lowered = text.lower()
    for extension in _SCENARIO_EXTENSIONS:
        if lowered.endswith(extension):
            text = text[: -len(extension)]
            break
    if not text:
        raise ValueError("a scenario file needs a name")
    if text != Path(text).name or "/" in text or "\\" in text:
        raise ValueError(f"scenario name {name!r} must not contain a path separator")
    return text


def write(
    document: Mapping[str, Any],
    name: str | None = None,
    directory: str | os.PathLike[str] | None = None,
    overwrite: bool = True,
) -> Path:
    """Write ``document`` into the scenarios directory and return the file.

    ``name`` is the id the session will know the scenario by and defaults to the
    name inside the document; the file is ``<name>.yaml``. ``directory``
    defaults to :func:`scenarios_dir`, so a plain call writes where Unity reads.

    ``overwrite`` is checked, not a promise: Unity caches a scenario it has
    already loaded, so rewriting a file under a name a session has used does
    *not* change what that session loads. Give a new name for a new document.
    """
    target_dir = Path(directory) if directory is not None else scenarios_dir()
    scenario_name = name or document["scenario_info"]["name"]
    stem = _file_stem(scenario_name)
    path = target_dir / f"{stem}.yaml"
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists")
    target_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(to_yaml(document), encoding="utf-8")
    return path


def list_names(directory: str | os.PathLike[str] | None = None) -> list[str]:
    """The ids of the scenarios of a directory, sorted, extension left out."""
    target_dir = Path(directory) if directory is not None else scenarios_dir()
    if not target_dir.is_dir():
        return []
    names = [
        child.stem
        for child in target_dir.iterdir()
        if child.is_file() and child.suffix.lower() in _SCENARIO_EXTENSIONS
    ]
    return sorted(names)


def delete(
    name: str,
    directory: str | os.PathLike[str] | None = None,
) -> bool:
    """Remove a scenario file, and say whether one was there to remove."""
    path = find(name, directory)
    if path is None:
        return False
    path.unlink()
    return True


def find(
    name: str,
    directory: str | os.PathLike[str] | None = None,
) -> Path | None:
    """The file a scenario id is stored in, or ``None`` when there is none.

    An id and a file name are two spellings of the same thing - the file is the
    id under one of the extensions the application reads - so the pairing is
    written once, here, and everything else asks this.
    """
    stem = _file_stem(name)
    target_dir = Path(directory) if directory is not None else scenarios_dir()
    for extension in _SCENARIO_EXTENSIONS:
        path = target_dir / f"{stem}{extension}"
        if path.is_file():
            return path
    return None
