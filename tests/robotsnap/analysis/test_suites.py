"""Tests for the scenario suites: names, YAML files and comma-separated lists."""

from pathlib import Path

import pytest

from robotsnap.analysis import (
    DEFAULT_SUITE,
    DEFAULT_SUITES,
    SuitesError,
    load_suite,
    scenarios_for,
    suite_names,
)

_REPO = Path(__file__).resolve().parents[3]


def test_suite_names_lists_the_built_in_suites():
    assert "basic" in suite_names()
    assert "basic_short" in suite_names()
    assert DEFAULT_SUITE == "basic"


def test_a_known_suite_name_resolves_to_its_scenarios():
    assert scenarios_for("basic") == DEFAULT_SUITES["basic"]
    assert len(scenarios_for("basic")) == 7
    assert scenarios_for("basic")[0] == "front_approach"
    assert scenarios_for("basic_short") == DEFAULT_SUITES["basic"][:6]


def test_a_suite_yaml_path_is_read(tmp_path):
    path = tmp_path / "mine.yaml"
    path.write_text(
        "name: mine\n"
        "description: un essai\n"
        "scenarios:\n"
        "  - corner\n"
        "  - crowd\n"
        "episodes_per_scenario: 3\n",
        encoding="utf-8",
    )

    assert scenarios_for(str(path)) == ("corner", "crowd")
    loaded = load_suite(path)
    assert loaded["name"] == "mine"
    assert loaded["description"] == "un essai"
    assert loaded["episodes_per_scenario"] == 3


def test_the_shipped_suite_files_read_back():
    basic = load_suite(_REPO / "configs" / "benchmarks" / "basic.yaml")
    assert basic["scenarios"] == DEFAULT_SUITES["basic"]
    assert basic["episodes_per_scenario"] == 5

    short = load_suite(_REPO / "configs" / "benchmarks" / "basic_short.yaml")
    assert short["scenarios"] == DEFAULT_SUITES["basic_short"]
    assert short["episodes_per_scenario"] == 1


def test_a_comma_separated_list_is_split():
    assert scenarios_for(" front_approach , corner ,, ") == ("front_approach", "corner")


def test_an_unknown_suite_is_an_error():
    with pytest.raises(SuitesError):
        scenarios_for("does_not_exist")
    with pytest.raises(SuitesError):
        scenarios_for("   ")


def test_a_missing_suite_file_is_an_error(tmp_path):
    with pytest.raises(SuitesError):
        scenarios_for(str(tmp_path / "nope.yaml"))


def test_a_suite_file_without_scenarios_is_an_error(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("name: empty\ndescription: rien\n", encoding="utf-8")
    with pytest.raises(SuitesError):
        scenarios_for(str(path))


def test_a_suite_file_with_a_bad_episode_budget_is_an_error(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        "name: bad\nscenarios:\n  - corner\nepisodes_per_scenario: 0\n", encoding="utf-8"
    )
    with pytest.raises(SuitesError):
        load_suite(path)
