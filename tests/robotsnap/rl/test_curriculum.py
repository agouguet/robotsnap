"""The curriculum ladder: loading, validation, and the two ways a stage ends.

The stages here are plain data, so nothing in this file touches a simulator:
the tests load the shipped YAML, build ladders in memory, and drive ``observe``
directly to watch a stage end on its episode budget or on its success window.
"""

from pathlib import Path

import pytest

from robotsnap.rl.curriculum import (
    DEFAULT_CURRICULUM_DIRECTORY,
    Curriculum,
    CurriculumError,
    Stage,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_the_shipped_yaml_loads_by_name_and_by_path():
    by_name = Curriculum.load("social_navigation")
    by_absolute = Curriculum.load(
        REPO_ROOT / "configs" / "curriculum" / "social_navigation.yaml"
    )
    by_relative = Curriculum.load("configs/curriculum/social_navigation.yaml")

    assert by_name.stages == by_absolute.stages == by_relative.stages
    assert by_name.current.name == "empty"
    assert by_name.reset_options()["scenario"] == "default"
    assert len(by_name.stages) >= 4
    assert all(isinstance(stage, Stage) for stage in by_name.stages)


def test_the_default_directory_is_the_repository_configs():
    assert DEFAULT_CURRICULUM_DIRECTORY == REPO_ROOT / "configs" / "curriculum"
    assert DEFAULT_CURRICULUM_DIRECTORY.is_dir()


def test_the_second_example_ladder_loads():
    crowd = Curriculum.load("crowd")
    assert [stage.scenario for stage in crowd.stages] == ["crowd", "big_crowd"]


def test_an_unknown_name_is_refused():
    with pytest.raises(CurriculumError):
        Curriculum.load("no_such_curriculum")


def test_an_empty_or_missing_stage_list_is_refused():
    with pytest.raises(CurriculumError):
        Curriculum.from_document({"stages": []})
    with pytest.raises(CurriculumError):
        Curriculum.from_document({"name": "empty"})


def test_a_stage_without_a_name_is_refused():
    with pytest.raises(CurriculumError):
        Curriculum.from_document({"stages": [{"scenario": "default"}]})


def test_a_threshold_outside_the_unit_interval_is_refused():
    for bad in (-0.1, 1.5):
        with pytest.raises(CurriculumError):
            Curriculum.from_document(
                {"stages": [{"name": "a", "success_threshold": bad}]}
            )


def test_a_non_list_stage_block_is_refused():
    with pytest.raises(CurriculumError):
        Curriculum.from_document({"stages": "empty"})
    with pytest.raises(CurriculumError):
        Curriculum.from_document(["not", "a", "mapping"])


def test_an_unknown_stage_key_is_refused():
    with pytest.raises(CurriculumError):
        Curriculum.from_document({"stages": [{"name": "a", "episode": 5}]})


def test_a_malformed_yaml_file_is_refused(tmp_path):
    broken = tmp_path / "broken.yaml"
    broken.write_text("stages: [unclosed\n", encoding="utf-8")
    with pytest.raises(CurriculumError):
        Curriculum.load(broken)


def test_a_window_defaults_when_only_a_threshold_is_given():
    curriculum = Curriculum.from_document(
        {"stages": [{"name": "a", "success_threshold": 0.5}]}
    )
    stage = curriculum.current
    assert stage.window == 20
    assert stage.min_episodes == 0


def test_reset_options_follow_the_stage_and_the_budget_advances_it():
    curriculum = Curriculum.from_document(
        {
            "stages": [
                {"name": "a", "scenario": "default", "episodes": 2},
                {"name": "b", "scenario": "front_approach", "options": {"launch": True}},
            ]
        }
    )
    assert curriculum.index == 0
    assert curriculum.current.name == "a"
    assert curriculum.reset_options() == {"scenario": "default"}

    assert curriculum.observe(success=False) is False
    assert curriculum.index == 0
    assert curriculum.observe(success=False) is True  # the budget of two is spent
    assert curriculum.index == 1
    assert curriculum.current.name == "b"
    assert curriculum.reset_options() == {"scenario": "front_approach", "launch": True}

    # The last stage has no exit rule of its own, so it never moves on.
    for _ in range(50):
        assert curriculum.observe(success=True) is False
    assert curriculum.finished is False


def test_the_success_window_advances_a_stage_early():
    curriculum = Curriculum.from_document(
        {
            "stages": [
                {
                    "name": "a",
                    "success_threshold": 0.75,
                    "window": 4,
                    "min_episodes": 4,
                    "episodes": 100,
                },
                {"name": "b"},
            ]
        }
    )
    assert curriculum.observe(success=True) is False
    assert curriculum.observe(success=True) is False
    assert curriculum.observe(success=True) is False
    assert curriculum.observe(success=True) is True  # 4/4 = 1.0, well before 100
    assert curriculum.index == 1


def test_the_success_window_respects_its_minimum():
    curriculum = Curriculum.from_document(
        {
            "stages": [
                {
                    "name": "a",
                    "success_threshold": 0.5,
                    "window": 4,
                    "min_episodes": 6,
                },
                {"name": "b"},
            ]
        }
    )
    for _ in range(4):
        assert curriculum.observe(success=True) is False  # window full, minimum not met
    assert curriculum.index == 0
    assert curriculum.observe(success=True) is False
    assert curriculum.observe(success=True) is True
    assert curriculum.index == 1


def test_a_low_recent_window_does_not_advance():
    curriculum = Curriculum.from_document(
        {
            "stages": [
                {
                    "name": "a",
                    "success_threshold": 0.8,
                    "window": 4,
                    "min_episodes": 4,
                    "episodes": 100,
                },
                {"name": "b"},
            ]
        }
    )
    for _ in range(3):
        assert curriculum.observe(success=False) is False
    assert curriculum.observe(success=True) is False  # window rate 0.25
    assert curriculum.index == 0


def test_the_budget_still_ends_a_stage_the_window_never_cleared():
    curriculum = Curriculum.from_document(
        {
            "stages": [
                {
                    "name": "a",
                    "success_threshold": 0.9,
                    "window": 2,
                    "min_episodes": 2,
                    "episodes": 5,
                },
                {"name": "b"},
            ]
        }
    )
    results = [curriculum.observe(success=False) for _ in range(5)]
    assert results == [False, False, False, False, True]
    assert curriculum.index == 1


def test_advance_moves_by_hand_and_stops_at_the_last_stage():
    curriculum = Curriculum.from_document(
        {"stages": [{"name": "a"}, {"name": "b"}]}
    )
    assert curriculum.advance() is True
    assert curriculum.index == 1
    assert curriculum.advance() is False
    assert curriculum.finished is False


def test_the_last_stage_completes_the_curriculum():
    curriculum = Curriculum.from_document(
        {"stages": [{"name": "a", "episodes": 1}, {"name": "b", "episodes": 1}]}
    )
    assert curriculum.observe(success=False) is True  # a -> b
    assert curriculum.observe(success=False) is True  # b -> finished
    assert curriculum.finished is True
    assert curriculum.observe(success=True) is False  # nothing left to climb


def test_state_and_summary_report_the_current_stage():
    curriculum = Curriculum.from_document({"stages": [{"name": "a", "episodes": 3}]})
    curriculum.observe(success=True)
    curriculum.observe(success=False)

    assert curriculum.state() == {
        "stage": 0,
        "name": "a",
        "episodes": 2,
        "successes": 1,
        "success_rate": 0.5,
        "finished": False,
    }
    assert "a" in curriculum.summary()
    assert "0.50" in curriculum.summary()
