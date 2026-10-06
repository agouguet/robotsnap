"""Reading and summarising the episode metrics a Unity session produces.

Unity records one document per episode - the outcome, the timings, the social numbers and the trajectories -
and hands it to the outside world twice: on ``/simulation/metrics`` while the session runs, and as files under
``StreamingAssets/metrics/`` for a reader that was not there when the episode was played. This module is the
second half of both: it knows the shape of an episode, it reads what was exported to disk, and it turns a
list of episodes into the averages a benchmarking campaign compares.

Nothing here needs Unity, ROS or a socket. The live stream is read through
:class:`~robotsnap.client.RobotSNAPClient`, which asks Unity for the same documents over the control topic;
the exported files are read here, from a directory a caller names.

The definitions the numbers carry are written down in the Unity recorder that computes them and repeated
here, because a summary is only useful if the reader can tell what was averaged:

``index``
    1-based place of the episode in its session, in the order the episodes finished. Every episode of one
    session that runs the same scenario shares its scenario id, so the index is what tells two of them
    apart on screen. It is display metadata: ``id`` stays the stable key an export and a client address an
    episode by. It is the one key a reader does not insist on - an export written before it existed is
    still read, without the ordinal - which is what :data:`OPTIONAL_EPISODE_KEYS` records.
``world_seconds``
    Simulation seconds between the first and the last sample, i.e. how far the world moved. It is the value
    an episode time limit is measured in.
``wall_seconds``
    Wall-clock seconds the same episode took, i.e. how long a human waited for it.
``path_length_m``
    Length of the polyline the robot actually travelled.
``straight_line_m``
    Distance from the episode's start pose to its goal: the shortest a perfect run could have been.
``min_human_distance_m`` / ``avg_human_distance_m``
    Closest and mean distance to the nearest crowd member. ``-1`` means the episode saw no human at all and
    the value is left out of every average computed here.
``min_clearance_m``
    Closest the two *bodies* came: the human distance with the robot's footprint radius and the crowd
    member's taken off. It is the gap a bystander would measure, it is what the personal-space count is
    counted on, and it may be negative when the outlines overlap. ``-1`` means the episode saw no human.
    Optional: an export written before this key existed simply carries no series for it.
``personal_space_intrusions``
    Number of times the robot *entered* the personal space of a human; sitting inside it counts once.
``personal_space_seconds``
    Simulated seconds spent with at least one human inside that space. "Inside" is a clearance of
    ``personal_space_radius_m`` or less between the two outlines, not a centre-to-centre distance.
``robots``
    Roster ids of every robot the episode saw, the tracked one first. The scalar metrics above describe
    ``robot`` only, so a multi-robot scenario is still one episode of one controlled robot; this list - and
    the ``robot_*`` keys of the trajectory map - is what says how many robots ran beside it.

The export folder used to hold one big session document per session. It now holds three append-only files
instead, and a folder can carry both shapes at once while a machine is being migrated, so every reader here
accepts either:

``sessions.jsonl``
    One JSON object per line, one line per session: ``{"id": "s_...", "started_at": "...", "schema": 1}``.
    It replaces ``metrics_index.json``, which is still read from a folder that predates it.
``catalogue.jsonl``
    One JSON object per finished episode, the same document this module summarises, except that the inline
    ``trajectories`` map is replaced by ``trajectory_ref``: ``{"file": "trajectories/s_....rbt", "offset":
    ... , "length": ..., "agents": ..., "points": ..., "encoding": "int16mm"}``. An export that could not
    write the archive keeps ``trajectory_ref: null`` and the inline map instead; when both are present the
    inline map wins, because it needs no file to be read.
``trajectories/<sessionId>.rbt``
    Append-only binary archive, little-endian throughout: a 12-byte file header (``"RSNPTRAJ"``, ``uint16``
    version 1, ``uint16`` reserved) followed by one record per episode. A record starts with ``"REC1"``, the
    total record length in bytes (these 8 included) and an agent count, then one block per agent: id length,
    UTF-8 id, point count, stride, ``float64`` t0, the ``int32`` origin x and z in millimetres for the whole
    track, and the per-point arrays ``uint32`` tMs, ``int16`` dxMm and ``int16`` dzMm. A record ends exactly
    at its declared length and an agent block is ``26 + idLength + 8 * pointCount`` bytes; a record that was
    cut short, or a reference that points past the file, is skipped rather than raised. The reader converts
    with ``t = t0 + tMs/1000``, ``x = (originXmm + dxMm)/1000`` and the same for ``z``.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

__all__ = [
    "METRICS_TOPIC",
    "EPISODE_KEYS",
    "OPTIONAL_EPISODE_KEYS",
    "OUTCOMES",
    "NO_HUMAN_DISTANCE",
    "is_episode",
    "episode_trajectories",
    "summarize",
    "read_session",
    "episodes_from_session",
    "read_index",
    "load_export",
    "session_files",
    "read_catalogue",
    "read_sessions",
    "load_trajectories",
]

#: Topic Unity publishes one finished episode on, as JSON, under the environment prefix.
METRICS_TOPIC = "/simulation/metrics"

#: Keys every episode document carries; a document missing one of them is not an episode. ``trajectories`` is
#: the inline trajectory map of the old export; the append-only catalogue replaces it with ``trajectory_ref``,
#: and :func:`is_episode` accepts either, so the tuple stays the full list the old shape must carry.
EPISODE_KEYS: tuple[str, ...] = (
    "id",
    "scenario",
    "robot",
    "started_at",
    "outcome",
    "world_seconds",
    "wall_seconds",
    "steps",
    "path_length_m",
    "straight_line_m",
    "avg_speed_mps",
    "max_speed_mps",
    "min_human_distance_m",
    "avg_human_distance_m",
    "personal_space_intrusions",
    "personal_space_seconds",
    "trajectories",
)

#: Keys an episode may carry that a reader must not insist on. ``index`` - the 1-based place of the episode
#: in its session - arrived after the first exports were written, and an episode recorded before it is still
#: an episode: requiring it here would throw away sessions a caller already has on disk, which is a worse
#: answer than reading them without the one field that only the display needs. ``robots`` - every robot the
#: episode saw, the tracked one first - is the same kind of addition: a multi-robot scenario writes it, a
#: single-robot export written before it is still read, and the trajectory keys alone say what ran.
OPTIONAL_EPISODE_KEYS: tuple[str, ...] = ("index", "robots")

#: How an episode can end, in the vocabulary the simulator and the environment share.
OUTCOMES: tuple[str, ...] = (
    "goal",
    "collision",
    "out_of_bounds",
    "timeout",
    "stopped",
    "unknown",
)

#: Distance a distance metric carries when the episode held no human to measure against.
NO_HUMAN_DISTANCE = -1.0

#: The append-only export files of a folder, and the binary archive they point into.
SESSIONS_FILE = "sessions.jsonl"
CATALOGUE_FILE = "catalogue.jsonl"
TRAJECTORY_DIRECTORY = "trajectories"

#: The magic and version of the binary trajectory archive, as the Unity writer lays them down.
TRAJECTORY_MAGIC = b"RSNPTRAJ"
TRAJECTORY_VERSION = 1
TRAJECTORY_RECORD_MAGIC = b"REC1"

#: The reference an episode of the append-only catalogue carries instead of the inline trajectory map.
TRAJECTORY_REF_KEY = "trajectory_ref"

#: Metrics summarised as a mean over the episodes that reported them.
_SCALAR_METRICS: tuple[str, ...] = (
    "world_seconds",
    "wall_seconds",
    "steps",
    "path_length_m",
    "straight_line_m",
    "avg_speed_mps",
    "max_speed_mps",
    "min_human_distance_m",
    "avg_human_distance_m",
    "min_clearance_m",
    "personal_space_intrusions",
    "personal_space_seconds",
)


def is_episode(value: Any) -> bool:
    """Whether ``value`` looks like an episode document rather than anything else Unity publishes.

    The check is on the keys, not on the values: a document that carries them all is one this module can
    summarise, and a document that is missing one is refused here rather than producing a mean of ``None``
    three functions later.

    The trajectories are the one thing the two export shapes disagree on: an old session document carries the
    inline ``trajectories`` map, while a line of the append-only catalogue carries ``trajectory_ref`` pointing
    into the binary archive instead. Either key satisfies the check, and the remaining keys are still required
    as they were - an episode we cannot summarise must not pass because it happens to name an archive.
    """
    if not isinstance(value, Mapping):
        return False
    if all(key in value for key in EPISODE_KEYS):
        return True
    required_without_trajectories = (key for key in EPISODE_KEYS if key != "trajectories")
    return all(key in value for key in required_without_trajectories) and TRAJECTORY_REF_KEY in value


def episode_trajectories(episode: Mapping[str, Any]) -> dict[str, list[tuple[float, float, float]]]:
    """The episode's trajectories as ``agent key -> [(t, x, z), ...]``.

    Unity writes them as ``[[t, x, z], ...]``; a top-down view wants tuples it can unpack, so the conversion
    happens once, here, instead of in every dashboard. A row that is not three numbers is skipped rather than
    allowed to break a whole session's drawing.
    """
    raw = episode.get("trajectories")
    if not isinstance(raw, Mapping):
        return {}

    tracks: dict[str, list[tuple[float, float, float]]] = {}
    for key, points in raw.items():
        if not isinstance(points, Sequence) or isinstance(points, (str, bytes)):
            continue
        track: list[tuple[float, float, float]] = []
        for point in points:
            if not isinstance(point, Sequence) or isinstance(point, (str, bytes)) or len(point) != 3:
                continue
            try:
                track.append((float(point[0]), float(point[1]), float(point[2])))
            except (TypeError, ValueError):
                continue
        tracks[str(key)] = track
    return tracks


def _numbers(episodes: Iterable[Mapping[str, Any]], key: str) -> list[float]:
    """The values of ``key`` across ``episodes``, skipping blanks and the ``-1`` no-value sentinel."""
    values: list[float] = []
    for episode in episodes:
        value = episode.get(key)
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if key in ("min_human_distance_m", "avg_human_distance_m") and number == NO_HUMAN_DISTANCE:
            continue
        # A clearance is allowed to be negative, so the sentinel is only read as "no human" when the episode
        # that carries it says the same about its human distance: a robot whose bodies overlap by exactly one
        # metre is a measurement, not a missing value.
        if key == "min_clearance_m" and number == NO_HUMAN_DISTANCE and _is_blank_distance(
            episode.get("min_human_distance_m")
        ):
            continue
        values.append(number)
    return values


def _is_blank_distance(value: Any) -> bool:
    """Whether a distance field says nothing: absent, unreadable, or the ``-1`` no-human sentinel."""
    if value is None:
        return True
    try:
        return float(value) == NO_HUMAN_DISTANCE
    except (TypeError, ValueError):
        return True


def _describe(values: list[float]) -> dict[str, float] | None:
    """Mean, minimum and maximum of a series, or ``None`` when there is nothing to average."""
    if not values:
        return None
    return {
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }


def summarize(episodes: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate a list of episodes into the numbers a campaign compares between two runs.

    The result is one flat document: how many episodes ran, how many ended each way - the success rate and
    the failure rates a benchmark reports first - and, for every scalar metric, its mean, minimum and maximum
    over the episodes that reported it. An empty list summarises to zero episodes with every series ``None``,
    which is the honest answer for a session that has not run yet.
    """
    collected = [episode for episode in episodes if is_episode(episode)]
    total = len(collected)

    outcomes = {outcome: 0 for outcome in OUTCOMES}
    for episode in collected:
        outcome = str(episode.get("outcome", "unknown"))
        outcomes[outcome] = outcomes.get(outcome, 0) + 1

    summary: dict[str, Any] = {
        "episodes": total,
        "outcomes": outcomes,
        "successes": outcomes.get("goal", 0),
        "success_rate": outcomes.get("goal", 0) / total if total else None,
    }

    for outcome in OUTCOMES:
        summary[f"{outcome}_rate"] = outcomes.get(outcome, 0) / total if total else None

    for key in _SCALAR_METRICS:
        summary[key] = _describe(_numbers(collected, key))

    return summary


# -- exported files ------------------------------------------------------


def read_session(path: str | Path) -> dict[str, Any]:
    """Parse one exported session document, the shape :class:`MetricsStore` writes.

    Raises :class:`FileNotFoundError` when the path does not exist and :class:`ValueError` when the file is
    not JSON or not an object: a caller reading a file it named itself is better served by a real error than
    by a silently empty session.
    """
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"{path} does not hold a JSON object")
    return document


def episodes_from_session(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The episode list of a session document, empty when it holds none."""
    episodes = document.get("episodes")
    if not isinstance(episodes, list):
        return []
    return [episode for episode in episodes if is_episode(episode)]


def read_index(directory: str | Path) -> dict[str, Any]:
    """The session index of ``directory``, empty when the folder has neither index yet.

    ``sessions.jsonl`` is preferred because it is what a session writes as it runs; ``metrics_index.json`` is
    the index the old export wrote in one go, and a folder that predates the switch is still read from it.
    Both answers have the same shape - a ``sessions`` list - so a caller does not have to know which one it
    is looking at.
    """
    root = Path(directory)
    if (root / SESSIONS_FILE).exists():
        return {"sessions": read_sessions(root)}

    path = root / "metrics_index.json"
    if not path.exists():
        return {"sessions": []}
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"{path} does not hold a JSON object")
    sessions = document.get("sessions")
    if not isinstance(sessions, list):
        document["sessions"] = []
    return document


def _read_json_lines(path: Path) -> list[dict[str, Any]]:
    """The JSON objects of an append-only JSONL file, skipping lines that are blank or unreadable.

    An append-only file is written while a session runs, so a reader can catch it mid-line: refusing the
    whole file over a partial last line would throw away every episode that was already complete. A line that
    is not a JSON object is skipped for the same reason - it is one line lost, not the folder.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []

    documents: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            document = json.loads(line)
        except ValueError:
            continue
        if isinstance(document, dict):
            documents.append(document)
    return documents


def read_sessions(directory: str | Path) -> list[dict[str, Any]]:
    """The session lines of ``directory``'s ``sessions.jsonl``, in the order they were appended."""
    return _read_json_lines(Path(directory) / SESSIONS_FILE)


def read_catalogue(directory: str | Path) -> list[dict[str, Any]]:
    """The episode lines of ``directory``'s ``catalogue.jsonl``, in the order they were appended."""
    return _read_json_lines(Path(directory) / CATALOGUE_FILE)


def session_files(directory: str | Path) -> list[Path]:
    """Every session document of ``directory``, oldest name first.

    The folder is scanned rather than the index read, so a session file whose index entry was lost - an
    interrupted export, a file copied by hand - is still found.
    """
    return sorted(Path(directory).glob("session_*.json"))


def load_export(directory: str | Path) -> list[dict[str, Any]]:
    """Every episode of every exported session under ``directory``, in file order.

    The append-only catalogue is read first, and an episode that reached it is not read a second time from a
    session document - a folder that holds both the new files and the old ones must not count the same
    episode twice. Episode ids are the key the archive and the catalogue agree on, so an episode carrying one
    is de-duplicated by it, and an episode without one is kept as it is.

    A missing directory is an empty list, not an error: a caller asking a machine that never ran a session
    is asking a question whose answer is "none".
    """
    seen: set[str] = set()
    episodes: list[dict[str, Any]] = []

    for episode in read_catalogue(directory):
        if not is_episode(episode):
            continue
        identifier = episode.get("id")
        if isinstance(identifier, str):
            if identifier in seen:
                continue
            seen.add(identifier)
        episodes.append(episode)

    for path in session_files(directory):
        try:
            document = read_session(path)
        except (OSError, ValueError):
            continue
        for episode in episodes_from_session(document):
            identifier = episode.get("id")
            if isinstance(identifier, str):
                if identifier in seen:
                    continue
                seen.add(identifier)
            episodes.append(episode)
    return episodes


# -- binary trajectory archive -------------------------------------------------


def load_trajectories(
    directory: str | Path, episode: Mapping[str, Any]
) -> dict[str, list[tuple[float, float, float]]]:
    """The trajectories of ``episode`` as ``agent key -> [(t, x, z), ...]``.

    An episode may carry its trajectories twice: the old export wrote the inline map, and the append-only
    catalogue writes ``trajectory_ref`` instead. The inline map wins when both are there, because it is the
    answer that needs no file to be present; otherwise the record ``trajectory_ref`` points at is read from
    the binary archive under ``directory``. A missing file, a reference that points past the end of the
    archive and a record that was cut short all read as an empty map - a caller drawing a session should see
    the sessions it can, not lose them to one unreadable record.
    """
    raw = episode.get("trajectories")
    if isinstance(raw, Mapping):
        return episode_trajectories(episode)

    reference = episode.get(TRAJECTORY_REF_KEY)
    if not isinstance(reference, Mapping):
        return {}
    return _read_trajectory_record(directory, reference)


def _read_trajectory_record(
    directory: str | Path, reference: Mapping[str, Any]
) -> dict[str, list[tuple[float, float, float]]]:
    """Read the one archive record ``reference`` points at, empty when it cannot be read whole."""
    file_name = reference.get("file")
    if not isinstance(file_name, str) or not file_name:
        return {}
    try:
        offset = int(reference.get("offset"))
        length = int(reference.get("length"))
    except (TypeError, ValueError):
        return {}
    if offset < 0 or length <= 0:
        return {}

    path = Path(directory) / file_name
    try:
        with path.open("rb") as handle:
            handle.seek(offset)
            window = handle.read(length)
    except OSError:
        return {}
    if len(window) < length:
        # The archive stops before the record does: the writer was interrupted between the two.
        return {}
    return _decode_trajectory_record(window)


def _decode_trajectory_record(window: bytes) -> dict[str, list[tuple[float, float, float]]]:
    """Decode one ``REC1`` record, empty when its declared length or any agent block does not fit."""
    if len(window) < 12 or window[0:4] != TRAJECTORY_RECORD_MAGIC:
        return {}
    record_length = int.from_bytes(window[4:8], "little")
    # A record ends exactly at recordLength bytes, so a declared length that runs past the bytes the caller
    # handed in is the truncated record it must skip.
    if record_length < 12 or record_length > len(window):
        return {}
    agent_count = int.from_bytes(window[8:12], "little")

    tracks: dict[str, list[tuple[float, float, float]]] = {}
    cursor = 12
    try:
        for _ in range(agent_count):
            if cursor + 2 > record_length:
                raise ValueError("record ends before the agent id length")
            id_length = int.from_bytes(window[cursor : cursor + 2], "little")
            cursor += 2

            if cursor + id_length > record_length:
                raise ValueError("record ends inside the agent id")
            agent_id = window[cursor : cursor + id_length].decode("utf-8")
            cursor += id_length

            # point count, stride, t0, origin x and origin z: 24 bytes before the point arrays.
            if cursor + 24 > record_length:
                raise ValueError("record ends before the agent header")
            point_count = int.from_bytes(window[cursor : cursor + 4], "little")
            # ``stride`` is the writer's sampling step; the points are already the kept ones, so it is walked
            # over here rather than used, keeping the read a straight copy of the layout Unity writes.
            cursor += 8
            t0 = struct.unpack_from("<d", window, cursor)[0]
            cursor += 8
            origin_x_mm, origin_z_mm = struct.unpack_from("<ii", window, cursor)
            cursor += 8

            block_length = 8 * point_count
            if cursor + block_length > record_length:
                raise ValueError("record ends inside the point arrays")
            times, xs, zs = _point_arrays(
                window[cursor : cursor + block_length], point_count, t0, origin_x_mm, origin_z_mm
            )
            cursor += block_length
            tracks[agent_id] = list(zip(times, xs, zs))
    except (UnicodeDecodeError, ValueError, struct.error):
        return {}
    return tracks


def _point_arrays(
    block: bytes, point_count: int, t0: float, origin_x_mm: int, origin_z_mm: int
) -> tuple[list[float], list[float], list[float]]:
    """Decode the three per-point arrays of one agent block, in world units.

    numpy is the fast path and reads the whole block in one buffer, which is what a session with millions of
    points needs. It is imported here rather than at module level because the package promises that
    ``import robotsnap.analysis`` stays free of it, and a reader without numpy gets the same answer through
    :mod:`struct` - one bulk unpack per array, never a point-by-point read of the file.
    """
    try:
        import numpy
    except ImportError:
        numpy = None

    if numpy is not None:
        t_ms = numpy.frombuffer(block, dtype="<u4", count=point_count, offset=0)
        dx_mm = numpy.frombuffer(block, dtype="<i2", count=point_count, offset=4 * point_count)
        dz_mm = numpy.frombuffer(block, dtype="<i2", count=point_count, offset=6 * point_count)
        times = (t0 + t_ms.astype("float64") / 1000.0).tolist()
        xs = ((origin_x_mm + dx_mm.astype("float64")) / 1000.0).tolist()
        zs = ((origin_z_mm + dz_mm.astype("float64")) / 1000.0).tolist()
        return times, xs, zs

    if point_count == 0:
        return [], [], []
    t_ms = struct.unpack_from(f"<{point_count}I", block, 0)
    dx_mm = struct.unpack_from(f"<{point_count}h", block, 4 * point_count)
    dz_mm = struct.unpack_from(f"<{point_count}h", block, 6 * point_count)
    return (
        [t0 + value / 1000.0 for value in t_ms],
        [(origin_x_mm + value) / 1000.0 for value in dx_mm],
        [(origin_z_mm + value) / 1000.0 for value in dz_mm],
    )
