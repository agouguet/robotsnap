"""A run removes the scenario it wrote, and never one it merely played.

The two are told apart by the environment, which is what knows whether the file
was written from the run's own fields or was already there under the name the
run asked for. The guard is pinned here on a stub environment, because the
question is which file survives and not how a session is driven.
"""

from types import SimpleNamespace

from robotsnap import scenario
from robotsnap.runs.session import _remove_scratch_scenario


def _document(name: str = "demo"):
    return scenario.build(
        name,
        map_name="basic/crowd",
        robots=[scenario.robot("robot_1", "jackal", (2.0, -6.0, 90), (20.0, -6.0))],
    )


def test_a_scenario_the_run_wrote_is_removed(tmp_path):
    scenario.write(_document(), name="demo", directory=tmp_path)
    environment = SimpleNamespace(wrote_scenario=True)

    assert _remove_scratch_scenario(environment, "demo", tmp_path) is True
    assert scenario.find("demo", tmp_path) is None


def test_a_scenario_the_run_only_played_is_left_where_it_is(tmp_path):
    """Naming a scenario the project ships must not take it away."""
    scenario.write(_document("front_approach"), name="front_approach", directory=tmp_path)
    environment = SimpleNamespace(wrote_scenario=False)

    assert _remove_scratch_scenario(environment, "front_approach", tmp_path) is False
    assert scenario.find("front_approach", tmp_path) is not None


def test_a_caller_that_is_not_a_session_keeps_its_scenario(tmp_path):
    """An environment that says nothing about writing has no claim on the file."""
    scenario.write(_document(), name="demo", directory=tmp_path)

    assert _remove_scratch_scenario(object(), "demo", tmp_path) is False
    assert scenario.find("demo", tmp_path) is not None
