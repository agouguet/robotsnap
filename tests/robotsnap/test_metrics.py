"""Tests for the episode metrics: the pure reader and the client surface over the fake peer."""

import json
import struct
import sys
import time

import pytest

from robotsnap.analysis import metrics
from robotsnap.client import METRICS_TOPIC


def _wait_for(predicate, timeout=2.0, interval=0.01):
    """Poll ``predicate`` with short sleeps; returns its last value."""
    deadline = time.monotonic() + timeout
    while True:
        value = predicate()
        if value:
            return value
        if time.monotonic() >= deadline:
            return value
        time.sleep(interval)


def episode(identifier="ep_1", outcome="goal", **overrides):
    """One episode document in the shape Unity writes."""
    document = {
        "id": identifier,
        "index": 1,
        "scenario": "corridor",
        "robot": "robot_1",
        "robots": ["robot_1"],
        "started_at": "2026-09-30T10:00:00.0000000Z",
        "outcome": outcome,
        "world_seconds": 12.0,
        "wall_seconds": 1.2,
        "steps": 600,
        "path_length_m": 15.0,
        "straight_line_m": 10.0,
        "avg_speed_mps": 1.25,
        "max_speed_mps": 1.6,
        "min_human_distance_m": 0.7,
        "avg_human_distance_m": 3.2,
        "personal_space_intrusions": 2,
        "personal_space_seconds": 1.5,
        "personal_space_radius_m": 0.5,
        "trajectory_stride": 1,
        "trajectories": {"robot_1": [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]], "human_3": [[0.0, 2.0, 1.0]]},
        "session": "s_test",
    }
    document.update(overrides)
    return document


# -- the pure reader -----------------------------------------------------


def test_an_episode_is_recognised_by_the_keys_it_must_carry():
    assert metrics.is_episode(episode())
    assert not metrics.is_episode({"id": "x"})
    assert not metrics.is_episode(None)


def test_the_index_is_the_one_key_a_reader_does_not_insist_on():
    """An export written before the ordinal existed is still read, without the ordinal.

    The ordinal is display metadata - it tells two episodes of the same scenario apart on screen - so a
    session a caller already has on disk must not become unreadable the day it was added.
    """
    assert "index" not in metrics.EPISODE_KEYS
    assert "index" in metrics.OPTIONAL_EPISODE_KEYS

    without_index = episode()
    del without_index["index"]
    assert metrics.is_episode(without_index)
    assert metrics.summarize([without_index])["episodes"] == 1

    assert metrics.is_episode(episode(index=7)), "an episode that carries its ordinal is read back whole"


def test_trajectories_come_back_as_unpackable_points():
    tracks = metrics.episode_trajectories(episode())

    assert tracks["robot_1"][1] == (1.0, 1.0, 0.0)
    assert tracks["human_3"] == [(0.0, 2.0, 1.0)]


def test_every_robot_of_a_multi_robot_episode_has_its_own_track():
    """A scenario with several robots files one trajectory per robot under its own roster id.

    The episode stays the tracked robot's - ``robot`` names the subject of the scalar metrics - and the other
    robots travel beside it as tracks, so a reader draws all of them without the numbers becoming a mean of
    two different runs.
    """
    multi = episode(
        robots=["robot_1", "robot_2", "robot_3"],
        trajectories={
            "robot_1": [[0.0, 0.0, 0.0]],
            "robot_2": [[0.0, 1.0, 0.0]],
            "robot_3": [[0.0, 2.0, 0.0]],
            "human_3": [[0.0, 3.0, 1.0]],
        },
    )

    tracks = metrics.episode_trajectories(multi)

    assert set(tracks) == {"robot_1", "robot_2", "robot_3", "human_3"}
    assert multi["robot"] == "robot_1", "the scalar metrics still name one tracked robot"
    assert multi["robots"] == ["robot_1", "robot_2", "robot_3"]


def test_the_robot_list_is_one_more_key_a_reader_does_not_insist_on():
    without_robots = episode()
    del without_robots["robots"]

    assert metrics.is_episode(without_robots), "an export written before the list existed is still an episode"
    assert metrics.summarize([without_robots])["episodes"] == 1


def test_a_malformed_point_is_skipped_not_fatal():
    tracks = metrics.episode_trajectories(episode(trajectories={"robot_1": [[0.0, 1.0], "nope", [2.0, 0.0, 0.0]]}))

    assert tracks["robot_1"] == [(2.0, 0.0, 0.0)]


def test_the_summary_reports_rates_and_means():
    summary = metrics.summarize(
        [
            episode("a", "goal"),
            episode("b", "collision", world_seconds=8.0, min_human_distance_m=0.3),
        ]
    )

    assert summary["episodes"] == 2
    assert summary["successes"] == 1
    assert summary["success_rate"] == pytest.approx(0.5)
    assert summary["collision_rate"] == pytest.approx(0.5)
    assert summary["world_seconds"]["mean"] == pytest.approx(10.0)
    assert summary["min_human_distance_m"]["min"] == pytest.approx(0.3)


def test_a_metric_with_no_value_is_left_out_of_its_mean():
    summary = metrics.summarize(
        [
            episode("a", min_human_distance_m=2.0, avg_human_distance_m=3.0),
            episode("b", min_human_distance_m=-1.0, avg_human_distance_m=-1.0),
        ]
    )

    assert summary["min_human_distance_m"]["mean"] == pytest.approx(2.0)
    assert summary["avg_human_distance_m"]["mean"] == pytest.approx(3.0)


def test_the_between_bodies_clearance_is_summarised_when_the_episode_carries_it():
    """``min_clearance_m`` is optional, so an episode that names it is averaged and the rest are skipped."""
    summary = metrics.summarize(
        [
            episode("a", min_clearance_m=0.4, robot_radius_m=0.3, human_radius_m=0.3),
            episode("b", min_clearance_m=-0.2),
            episode("c"),  # written before the key existed
        ]
    )

    assert summary["episodes"] == 3
    assert summary["min_clearance_m"]["mean"] == pytest.approx(0.1)
    assert summary["min_clearance_m"]["min"] == pytest.approx(-0.2), "overlapping bodies stay negative"


def test_a_clearance_is_only_blank_when_the_episode_saw_no_human():
    """The ``-1`` sentinel means "no human" here too, but a real passage of -1 m is still a measurement."""
    summary = metrics.summarize(
        [
            episode("a", min_clearance_m=-1.0, min_human_distance_m=0.2),
            episode("b", min_clearance_m=-1.0, min_human_distance_m=-1.0),
        ]
    )

    assert summary["min_clearance_m"]["mean"] == pytest.approx(-1.0)


def test_an_empty_campaign_summarises_to_nothing():
    summary = metrics.summarize([])

    assert summary["episodes"] == 0
    assert summary["success_rate"] is None
    assert summary["path_length_m"] is None


# -- the exported files --------------------------------------------------


def test_an_exported_session_is_read_without_unity(tmp_path):
    document = {"session": "s_test", "episode_count": 1, "episodes": [episode()]}
    (tmp_path / "session_s_test.json").write_text(json.dumps(document), encoding="utf-8")
    (tmp_path / "metrics_index.json").write_text(
        json.dumps({"sessions": [{"id": "s_test", "file": "session_s_test.json"}]}), encoding="utf-8"
    )

    found = metrics.load_export(tmp_path)

    assert [entry["id"] for entry in found] == ["ep_1"]
    assert metrics.read_index(tmp_path)["sessions"][0]["id"] == "s_test"


def test_a_directory_that_never_ran_a_session_reads_empty(tmp_path):
    assert metrics.load_export(tmp_path / "missing") == []
    assert metrics.read_index(tmp_path / "missing")["sessions"] == []


# -- the append-only export ----------------------------------------------


def catalogue_episode(identifier="ep_1", **overrides):
    """One episode as the append-only catalogue writes it: a reference, not the inline map."""
    document = episode(identifier, **overrides)
    document.pop("trajectories", None)
    return document


def _agent_block(agent_id, stride, t0, origin_mm, points):
    """One agent block of a ``REC1`` record, laid out exactly as the Unity writer lays it down."""
    identifier = agent_id.encode("utf-8")
    block = bytearray()
    block += struct.pack("<H", len(identifier))
    block += identifier
    block += struct.pack("<II", len(points), stride)
    block += struct.pack("<d", t0)
    block += struct.pack("<ii", origin_mm[0], origin_mm[1])
    block += struct.pack(f"<{len(points)}I", *(point[0] for point in points))
    block += struct.pack(f"<{len(points)}h", *(point[1] for point in points))
    block += struct.pack(f"<{len(points)}h", *(point[2] for point in points))
    return bytes(block)


def _trajectory_record(agents):
    """One ``REC1`` record: the 8-byte record header, the agent count and one block per agent."""
    body = b"".join(_agent_block(*agent) for agent in agents)
    return struct.pack("<4sII", b"REC1", 12 + len(body), len(agents)) + body


def _trajectory_file(record):
    """A ``.rbt`` archive: the 12-byte file header followed by ``record``."""
    return b"RSNPTRAJ" + struct.pack("<HH", 1, 0) + record


def _trajectory_reference(length, offset=12, file="trajectories/s_test.rbt"):
    """The map an episode of the catalogue carries in place of its inline trajectories."""
    return {"file": file, "offset": offset, "length": length, "agents": 2, "points": 5, "encoding": "int16mm"}


# Times and millimetre offsets below are multiples of 125, so the world values they convert to are exact in
# binary and the read-back can be compared to literals without a tolerance.
_ROBOT = ("robot_1", 2, 1.5, (2000, -1250), [(0, 0, 0), (250, -500, 125), (1375, 250, -375)])
_HUMAN = ("human_3", 1, 1.5, (-1250, 3750), [(0, 125, -250), (500, -375, 250)])
_ROBOT_POINTS = [(1.5, 2.0, -1.25), (1.75, 1.5, -1.125), (2.875, 2.25, -1.625)]
_HUMAN_POINTS = [(1.5, -1.125, 3.5), (2.0, -1.625, 4.0)]


def _write_archive(tmp_path, record, name="s_test.rbt"):
    """Write ``record`` as a ``.rbt`` archive under ``tmp_path`` and return its byte length."""
    archive = tmp_path / "trajectories"
    archive.mkdir(exist_ok=True)
    (archive / name).write_bytes(_trajectory_file(record))
    return len(record)


def test_a_binary_trajectory_record_is_read_back_exactly(tmp_path):
    record = _trajectory_record([_ROBOT, _HUMAN])
    length = _write_archive(tmp_path, record)

    document = catalogue_episode(trajectory_ref=_trajectory_reference(length))

    assert metrics.is_episode(document), "a catalogue line is an episode even though it carries no inline map"
    assert metrics.load_trajectories(tmp_path, document) == {
        "robot_1": _ROBOT_POINTS,
        "human_3": _HUMAN_POINTS,
    }


def test_the_trajectory_archive_is_read_the_same_without_numpy(tmp_path, monkeypatch):
    """The numpy fast path and the ``struct`` fallback must agree to the last point."""
    record = _trajectory_record([_ROBOT, _HUMAN])
    length = _write_archive(tmp_path, record)
    document = catalogue_episode(trajectory_ref=_trajectory_reference(length))

    monkeypatch.setitem(sys.modules, "numpy", None)

    assert metrics.load_trajectories(tmp_path, document) == {
        "robot_1": _ROBOT_POINTS,
        "human_3": _HUMAN_POINTS,
    }


def test_the_inline_map_wins_when_the_episode_carries_both(tmp_path):
    document = episode("ep_both")
    document["trajectory_ref"] = _trajectory_reference(10, file="trajectories/never_written.rbt")

    tracks = metrics.load_trajectories(tmp_path, document)

    assert tracks == metrics.episode_trajectories(document)
    assert tracks["robot_1"][0] == (0.0, 0.0, 0.0)


def test_a_catalogue_and_an_old_session_are_merged_without_duplicates(tmp_path):
    length = _write_archive(tmp_path, _trajectory_record([_ROBOT, _HUMAN]))
    referenced = catalogue_episode("ep_1", trajectory_ref=_trajectory_reference(length))
    inline = catalogue_episode("ep_2")
    inline["trajectory_ref"] = None
    inline["trajectories"] = {"robot_1": [[0.0, 0.0, 0.0]]}

    (tmp_path / "catalogue.jsonl").write_text(
        "\n".join([json.dumps(referenced), "", json.dumps(inline), "{not json"]) + "\n", encoding="utf-8"
    )
    (tmp_path / "sessions.jsonl").write_text(
        json.dumps({"id": "s_test", "started_at": "2026-10-05T09:36:16.1234567Z", "schema": 1}) + "\n",
        encoding="utf-8",
    )
    old_session = {"session": "s_test", "episode_count": 2, "episodes": [episode("ep_1"), episode("ep_3")]}
    (tmp_path / "session_s_test.json").write_text(json.dumps(old_session), encoding="utf-8")
    (tmp_path / "metrics_index.json").write_text(json.dumps({"sessions": [{"id": "stale"}]}), encoding="utf-8")

    found = metrics.load_export(tmp_path)

    assert [entry["id"] for entry in found] == ["ep_1", "ep_2", "ep_3"]
    assert found[0]["trajectory_ref"]["file"] == "trajectories/s_test.rbt"
    assert found[2]["id"] == "ep_3", "the old session document still contributes what the catalogue lacks"
    assert metrics.summarize(found)["episodes"] == 3
    assert [entry["id"] for entry in metrics.read_catalogue(tmp_path)] == ["ep_1", "ep_2"]
    assert metrics.read_sessions(tmp_path)[0]["schema"] == 1
    assert metrics.read_index(tmp_path)["sessions"][0]["id"] == "s_test", "the append-only index wins"


def test_a_truncated_trajectory_reference_reads_empty(tmp_path):
    record = _trajectory_record([_ROBOT, _HUMAN])
    length = _write_archive(tmp_path, record)

    short = catalogue_episode(trajectory_ref=_trajectory_reference(length - 4))
    assert metrics.load_trajectories(tmp_path, short) == {}

    whole = catalogue_episode(trajectory_ref=_trajectory_reference(length))
    (tmp_path / "trajectories" / "s_test.rbt").write_bytes(_trajectory_file(record)[:-4])
    assert metrics.load_trajectories(tmp_path, whole) == {}

    missing = catalogue_episode(trajectory_ref=_trajectory_reference(length, file="trajectories/never.rbt"))
    assert metrics.load_trajectories(tmp_path, missing) == {}


def test_a_folder_without_the_append_only_files_is_still_read(tmp_path):
    document = {"session": "s_old", "episode_count": 1, "episodes": [episode("ep_old")]}
    (tmp_path / "session_s_old.json").write_text(json.dumps(document), encoding="utf-8")
    (tmp_path / "metrics_index.json").write_text(json.dumps({"sessions": [{"id": "s_old"}]}), encoding="utf-8")

    assert metrics.read_catalogue(tmp_path) == []
    assert metrics.read_sessions(tmp_path) == []
    assert [entry["id"] for entry in metrics.load_export(tmp_path)] == ["ep_old"]
    assert metrics.read_index(tmp_path)["sessions"][0]["id"] == "s_old", "the old index is the fallback"


def test_an_episode_with_a_trajectory_reference_is_an_episode():
    referenced = catalogue_episode(trajectory_ref=_trajectory_reference(100))
    assert metrics.is_episode(referenced)

    broken = dict(referenced)
    del broken["outcome"]
    assert not metrics.is_episode(broken), "the reference replaces the map, not the rest of the keys"

    neither = episode()
    neither.pop("trajectories")
    assert not metrics.is_episode(neither), "an episode must carry one of the two trajectory keys"


# -- the client surface --------------------------------------------------


def test_the_session_is_empty_before_any_episode(client, unity):
    unity.start_responder()
    client.wait_until_ready()

    assert client.episodes() == []
    assert client.episode("nothing") is None


def test_an_episode_is_read_back_by_the_client(client, unity):
    unity.start_responder()
    client.wait_until_ready()
    unity.episodes.append(episode())

    listed = client.episodes()
    fetched = client.episode("ep_1")

    assert [entry["id"] for entry in listed] == ["ep_1"]
    assert fetched["outcome"] == "goal"
    assert fetched["trajectories"]["robot_1"][0] == [0.0, 0.0, 0.0]
    assert client.metrics_session()["count"] == 1


def test_the_summary_comes_from_the_same_episodes(client, unity):
    unity.start_responder()
    client.wait_until_ready()
    unity.episodes.append(episode("a", "goal"))
    unity.episodes.append(episode("b", "timeout"))

    summary = client.metrics_summary()

    assert summary["episodes"] == 2
    assert summary["success_rate"] == pytest.approx(0.5)


def test_clearing_the_session_says_what_it_cleared(client, unity):
    unity.start_responder()
    client.wait_until_ready()
    unity.episodes.append(episode())

    assert client.clear_episodes() == 1
    assert client.episodes() == []


def test_the_published_stream_is_read_as_it_runs(client, unity):
    unity.start_responder()
    client.wait_until_ready()

    assert client.latest_episode() is None

    unity.publish_episode(episode("live", "goal"))

    live = _wait_for(lambda: client.latest_episode())
    assert live is not None and live["id"] == "live"
    assert METRICS_TOPIC == "/simulation/metrics"


def test_the_client_talks_metrics_over_the_control_topic(client, unity):
    unity.start_responder()
    client.wait_until_ready()

    client.episodes()

    bodies = [body for _, body in unity.commands if body.get("command") == "metrics_episodes"]
    assert bodies, "the list is asked for over /simulation/control"


def test_an_exported_summary_needs_a_directory(client, unity):
    unity.start_responder()
    client.wait_until_ready()

    assert client.exported_episodes() == []
    assert client.last_error is not None
