"""Tests for benchmark campaigns: the run loop, the JSON round-trip and the comparison.

The environment is a fake with the Gymnasium surface the run loop uses - a
``reset(seed=..., options=...)``, a ``step`` that ends the episode at once, and
a ``close`` it must never be asked to call - so a campaign is tested without a
socket, a Unity Editor or a single real episode.
"""

import json

import pytest

from robotsnap.analysis.campaign import (
    Campaign,
    CampaignError,
    ScenarioResult,
    compare_campaigns,
    format_comparison,
    load_campaign,
    run_campaign,
    save_campaign,
)
from robotsnap.analysis.metrics import summarize


def episode_doc(scenario, index, outcome="goal", **overrides):
    """One episode document in the shape the metrics module reads."""
    document = {
        "id": f"{scenario}_{index}",
        "index": index,
        "scenario": scenario,
        "robot": "robot_1",
        "robots": ["robot_1"],
        "started_at": "2026-10-05T10:00:00.0000000Z",
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
        "min_clearance_m": 0.4,
        "personal_space_intrusions": 2,
        "personal_space_seconds": 1.5,
        "trajectories": {"robot_1": [[0.0, 0.0, 0.0]]},
    }
    document.update(overrides)
    return document


class FakeEnvironment:
    """A Gymnasium-shaped environment that records resets and ends each episode at once."""

    def __init__(self):
        self.client = None
        self.resets = []
        self.steps = 0
        self.closed = False

    def reset(self, *, seed=None, options=None):
        self.resets.append({"seed": seed, "options": options})
        return {"observation": len(self.resets)}, {"scenario": (options or {}).get("scenario")}

    def step(self, action):
        self.steps += 1
        return {}, 0.0, True, False, {"action": action}

    def close(self):
        self.closed = True


class ScriptedClient:
    """A client whose answers grow, one per read, like a live session's episode list."""

    def __init__(self, *payloads):
        self._payloads = list(payloads)
        self.calls = 0

    def episodes(self):
        payload = self._payloads[min(self.calls, len(self._payloads) - 1)]
        self.calls += 1
        return payload


def _act(observation, info):
    return 0.0


# -- the run loop --------------------------------------------------------


def test_run_campaign_plays_each_scenario_in_order():
    environment = FakeEnvironment()
    seen = []

    def reader(scenario, index):
        seen.append((scenario, index))
        return episode_doc(scenario, index)

    result = run_campaign(
        environment=environment,
        scenarios=("corner", "crowd"),
        episodes_per_scenario=2,
        act=_act,
        episode_reader=reader,
        name="test",
    )

    assert [reset["options"] for reset in environment.resets] == [
        {"scenario": "corner", "launch": True},
        {"scenario": "corner", "launch": True},
        {"scenario": "crowd", "launch": True},
        {"scenario": "crowd", "launch": True},
    ]
    assert seen == [("corner", 1), ("corner", 2), ("crowd", 1), ("crowd", 2)]
    assert [item.scenario for item in result.results] == ["corner", "crowd"]
    assert [len(item.episodes) for item in result.results] == [2, 2]
    assert result.name == "test"
    assert result.scenarios == ("corner", "crowd")
    assert result.episodes_per_scenario == 2
    assert result.started_at and result.finished_at
    assert environment.closed is False


def test_run_campaign_summarises_like_metrics_summarize():
    environment = FakeEnvironment()
    documents = [episode_doc("corner", 1, "goal"), episode_doc("corner", 2, "collision")]

    result = run_campaign(
        environment=environment,
        scenarios=("corner",),
        episodes_per_scenario=2,
        act=_act,
        episode_reader=lambda scenario, index: documents[index - 1],
    )

    assert result.results[0].summary == summarize(documents)
    assert result.episodes == documents
    assert result.summary == summarize(documents)
    assert result.summary["success_rate"] == 0.5
    assert result.summary["collision_rate"] == 0.5
    assert result.results[0].summary["min_human_distance_m"]["mean"] == 0.7


def test_run_campaign_reads_the_last_episode_of_the_client():
    environment = FakeEnvironment()
    first = episode_doc("corner", 1, "collision")
    second = episode_doc("corner", 2, "goal")
    environment.client = ScriptedClient([first], [first, second])

    result = run_campaign(
        environment=environment,
        scenarios=("corner",),
        episodes_per_scenario=2,
        act=_act,
    )

    # The first read sees one finished episode, the second sees two: each time the
    # campaign takes the last, never a mix of the two.
    assert [episode["outcome"] for episode in result.results[0].episodes] == [
        "collision",
        "goal",
    ]


def test_a_seed_makes_each_episode_reproducible():
    environment = FakeEnvironment()

    run_campaign(
        environment=environment,
        scenarios=("corner",),
        episodes_per_scenario=3,
        act=_act,
        episode_reader=lambda scenario, index: episode_doc(scenario, index),
        seed=100,
    )

    assert [reset["seed"] for reset in environment.resets] == [100, 101, 102]


def test_a_missing_episode_is_counted_as_a_failure():
    environment = FakeEnvironment()

    with pytest.warns(RuntimeWarning):
        result = run_campaign(
            environment=environment,
            scenarios=("corner",),
            episodes_per_scenario=2,
            act=_act,
            episode_reader=lambda scenario, index: None,
        )

    assert result.summary["episodes"] == 2
    assert result.summary["success_rate"] == 0.0
    assert result.summary["outcomes"]["unknown"] == 2


def test_a_partial_document_is_treated_as_a_missing_episode():
    environment = FakeEnvironment()

    with pytest.warns(RuntimeWarning):
        result = run_campaign(
            environment=environment,
            scenarios=("corner",),
            episodes_per_scenario=1,
            act=_act,
            episode_reader=lambda scenario, index: {"id": "not an episode"},
        )

    assert result.summary["outcomes"]["unknown"] == 1


def test_should_stop_ends_the_campaign_between_episodes():
    environment = FakeEnvironment()
    stop = {"flag": False}
    lines = []

    def on_episode(index, scenario, line):
        lines.append(line)
        stop["flag"] = True

    result = run_campaign(
        environment=environment,
        scenarios=("corner", "crowd"),
        episodes_per_scenario=3,
        act=_act,
        episode_reader=lambda scenario, index: episode_doc(scenario, index),
        on_episode=on_episode,
        should_stop=lambda: stop["flag"],
    )

    assert len(result.episodes) == 1
    assert [item.scenario for item in result.results] == ["corner"]
    assert lines and "corner" in lines[0] and "goal" in lines[0]


def test_a_campaign_needs_at_least_one_scenario_and_episode():
    environment = FakeEnvironment()
    with pytest.raises(CampaignError):
        run_campaign(environment=environment, scenarios=(), episodes_per_scenario=1, act=_act)
    with pytest.raises(CampaignError):
        run_campaign(
            environment=environment,
            scenarios=("corner",),
            episodes_per_scenario=0,
            act=_act,
        )


# -- saving and loading --------------------------------------------------


def _sample_campaign():
    episodes = [episode_doc("corner", 1, "goal"), episode_doc("corner", 2, "collision")]
    return Campaign(
        name="demo",
        suite="basic_short",
        scenarios=("corner",),
        episodes_per_scenario=2,
        policy="scripted",
        started_at="2026-10-05T10:00:00+00:00",
        finished_at="2026-10-05T10:05:00+00:00",
        results=[ScenarioResult("corner", episodes, summarize(episodes))],
    )


def test_to_document_is_json_serialisable_and_stable():
    document = _sample_campaign().to_document()
    assert json.loads(json.dumps(document)) == document
    assert document["scenarios"] == ["corner"]
    assert document["results"][0]["scenario"] == "corner"


def test_save_and_load_are_a_round_trip(tmp_path):
    campaign = _sample_campaign()
    path = save_campaign(campaign, tmp_path / "nested" / "demo.json")

    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8"))["name"] == "demo"
    loaded = load_campaign(path)
    assert loaded == campaign
    assert loaded.episodes == campaign.episodes
    assert loaded.summary == campaign.summary


def test_from_document_reverses_to_document():
    campaign = _sample_campaign()
    assert Campaign.from_document(campaign.to_document()) == campaign


def test_load_campaign_refuses_a_document_that_is_not_a_campaign(tmp_path):
    path = tmp_path / "no.json"
    path.write_text('{"hello": "world"}', encoding="utf-8")
    with pytest.raises(CampaignError):
        load_campaign(path)


# -- comparing two campaigns ---------------------------------------------


def _campaign(name, outcomes, **overrides):
    episodes = [
        episode_doc("corner", index + 1, outcome, **overrides)
        for index, outcome in enumerate(outcomes)
    ]
    return Campaign(
        name=name,
        suite="basic_short",
        scenarios=("corner",),
        episodes_per_scenario=len(episodes),
        policy="scripted",
        started_at="2026-10-05T10:00:00+00:00",
        finished_at="2026-10-05T10:05:00+00:00",
        results=[ScenarioResult("corner", episodes, summarize(episodes))],
    )


def _entry_of(comparison, name):
    return next(item for item in comparison["metrics"] if item["name"] == name)


def test_compare_campaigns_reports_deltas_and_changes():
    a = _campaign("A", ["goal", "collision"])
    b = _campaign("B", ["goal", "goal"])

    comparison = compare_campaigns(a, b)

    assert comparison["a"] == "A"
    assert comparison["b"] == "B"
    assert comparison["episodes"]["a"] == 2
    assert comparison["episodes"]["delta"] == 0

    success = _entry_of(comparison, "success_rate")
    assert success["a"] == 0.5
    assert success["b"] == 1.0
    assert success["delta"] == pytest.approx(0.5)
    assert success["change"] == pytest.approx(1.0)

    collision = _entry_of(comparison, "collision_rate")
    assert collision["delta"] == pytest.approx(-0.5)
    assert collision["change"] == pytest.approx(-1.0)

    distance = _entry_of(comparison, "min_human_distance_m.mean")
    assert distance["a"] == 0.7
    assert distance["delta"] == 0.0


def test_compare_campaigns_names_a_metric_only_one_side_has():
    a = _campaign("A", ["goal"])
    b = _campaign("B", ["goal"], min_clearance_m=None)

    comparison = compare_campaigns(a, b)

    assert "min_clearance_m.mean" in comparison["only_in_a"]
    assert "min_clearance_m.mean" not in [item["name"] for item in comparison["metrics"]]


def test_compare_campaigns_survives_a_zero_baseline():
    a = _campaign("A", ["collision"])
    b = _campaign("B", ["goal"])

    comparison = compare_campaigns(a, b)
    success = _entry_of(comparison, "success_rate")

    assert success["a"] == 0.0
    assert success["delta"] == pytest.approx(1.0)
    assert success["change"] is None
    assert "n/a" in format_comparison(comparison)


def test_format_comparison_is_a_readable_table():
    text = format_comparison(
        compare_campaigns(_campaign("A", ["goal"]), _campaign("B", ["collision"]))
    )

    assert isinstance(text, str)
    assert text.strip()
    lines = text.splitlines()
    assert len(lines) >= 4
    assert "metric" in lines[0]
    assert "delta" in lines[0]
    assert any("success_rate" in line for line in lines)
    assert any("episodes" in line for line in lines)
