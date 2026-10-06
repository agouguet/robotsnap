"""A campaign driven against the fake Unity peer, and the CLI's two ends of it.

The arithmetic of a campaign is covered by ``test_campaign.py``; what this file
checks is the wiring the command line adds on top - a suite of scenarios really
being launched one after the other, the policy really driving the session, and
the saved file really being what ``--compare`` reads back.
"""

import json
import time

from robotsnap import cli, runs
from robotsnap.analysis import load_campaign
from robotsnap.analysis.metrics import EPISODE_KEYS

_WAIT_TIMEOUT = 2.0
_CONTROL_PERIOD = 0.05


def _episode(identifier: str, scenario: str, outcome: str = "goal") -> dict:
    """One episode document with every key :func:`is_episode` insists on."""
    document = {
        "id": identifier,
        "scenario": scenario,
        "robot": "robot_1",
        "started_at": "2026-10-05T10:00:00.0000000Z",
        "outcome": outcome,
        "world_seconds": 12.0,
        "wall_seconds": 1.0,
        "steps": 60,
        "path_length_m": 9.0,
        "straight_line_m": 8.0,
        "avg_speed_mps": 0.75,
        "max_speed_mps": 1.0,
        "min_human_distance_m": 0.9,
        "avg_human_distance_m": 3.0,
        "personal_space_intrusions": 0,
        "personal_space_seconds": 0.0,
        "trajectories": {},
    }
    assert set(document) == set(EPISODE_KEYS)
    return document


def _publish_world(unity, x=0.0, y=0.0, yaw=0.0, goal=(5.0, 0.0), sim_time=10.0):
    """Publish the whole world one environment read needs, in one go."""
    unity.publish_state(
        {
            "simulation_state": "running",
            "playing": True,
            "paused": False,
            "stopped": False,
            "scenario_applied": True,
            "scenario_id": "demo",
            "scenario_name": "demo",
            "sim_time": sim_time,
            "time_scale": 1.0,
            "robots": [
                {
                    "id": "robot_1",
                    "type": "jackal",
                    "is_primary": True,
                    "x": x,
                    "y": y,
                    "z": 0.0,
                    "yaw": yaw,
                    "has_goal": True,
                    "goal": {"x": goal[0], "y": goal[1], "z": 0.0},
                    "start_pose": None,
                    "target_pose": None,
                }
            ],
            "humans": [],
        }
    )


def _start(unity, bridge, **world):
    """Bring the fake session up: responder thread, then one world."""
    deadline = time.monotonic() + _WAIT_TIMEOUT
    while bridge.topic_types().get("simulation/control_result") != "std_msgs/msg/String":
        assert time.monotonic() < deadline, "the peer never registered its streams"
        time.sleep(0.01)
    unity.start_responder()
    _publish_world(unity, **world)


def test_a_campaign_runs_every_scenario_of_its_suite(
    client, unity, bridge, tmp_path, capsys, monkeypatch
):
    """Two scenarios in, two scenarios out, and a file that says so."""
    project = tmp_path / "Project"
    (project / "Assets" / "StreamingAssets" / "Scenarios").mkdir(parents=True)
    _start(unity, bridge)

    documents = [_episode("e1", "front_approach"), _episode("e2", "corner")]
    monkeypatch.setattr(client, "episodes", lambda: list(documents))

    # The fake peer answers a launch but never republishes the world under the
    # new name, and the client waits for that to decide a scenario is applied.
    # Recording the asks is what this test is about: the campaign has to launch
    # each scenario of the suite, in order, and the session's own handshake is
    # the peer's business - covered where the peer lives.
    launched: list[str] = []

    def launch(scenario_id, **kwargs):
        launched.append(scenario_id)
        return True

    monkeypatch.setattr(client, "launch_scenario", launch)

    out = tmp_path / "campaign.json"
    code = runs.run_scenarios(
        suite="front_approach,corner",
        episodes=1,
        policy="scripted",
        client=client,
        port=bridge.port,
        host="127.0.0.1",
        unity_project=str(project),
        control_period=_CONTROL_PERIOD,
        seconds=0.2,
        max_steps=3,
        pacing="free",
        keep=True,
        stop=False,
        out=str(out),
    )
    printed = capsys.readouterr().out
    assert code == 0, printed
    assert "benchmark" in printed
    assert launched == ["front_approach", "corner"]

    campaign = load_campaign(out)
    assert tuple(campaign.scenarios) == ("front_approach", "corner")
    assert [result.scenario for result in campaign.results] == [
        "front_approach",
        "corner",
    ]
    assert len(campaign.episodes) == 2
    assert campaign.summary["success_rate"] == 1.0


def test_compare_prints_a_table_for_two_saved_campaigns(tmp_path, capsys):
    """The other end of the command: two files in, a comparison out."""
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    first.write_text(
        json.dumps(
            {
                "name": "a",
                "suite": "basic",
                "scenarios": ["corner"],
                "episodes_per_scenario": 2,
                "policy": "scripted",
                "started_at": "2026-10-05T10:00:00+00:00",
                "finished_at": "2026-10-05T10:01:00+00:00",
                "results": [
                    {
                        "scenario": "corner",
                        "episodes": [_episode("a1", "corner", "collision")],
                        "summary": {},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    second.write_text(
        json.dumps(
            {
                "name": "b",
                "suite": "basic",
                "scenarios": ["corner"],
                "episodes_per_scenario": 2,
                "policy": "scripted",
                "started_at": "2026-10-05T11:00:00+00:00",
                "finished_at": "2026-10-05T11:01:00+00:00",
                "results": [
                    {
                        "scenario": "corner",
                        "episodes": [_episode("b1", "corner", "goal")],
                        "summary": {},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert cli.main(["benchmark", "--compare", str(first), str(second)]) == 0
    printed = capsys.readouterr().out
    assert "success_rate" in printed
    assert "metric" in printed
