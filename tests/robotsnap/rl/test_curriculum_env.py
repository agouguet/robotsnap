"""The curriculum wrapper: options on reset, success on step, and its ladder.

The environment the wrapper drives here is a handful of lines - no Unity, no
sockets - because what is under test is only the plumbing between a curriculum
and whatever environment it is handed.
"""

import os
import subprocess
import sys
from pathlib import Path

import gymnasium
import numpy as np
import pytest

from robotsnap.rl.curriculum import Curriculum, CurriculumError
from robotsnap.rl.curriculum_env import CurriculumEnv, episode_succeeded

REPO_ROOT = Path(__file__).resolve().parents[3]


class FakeEnv(gymnasium.Env):
    """The smallest environment the wrapper can drive: it records its resets."""

    def __init__(self):
        self.observation_space = gymnasium.spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32
        )
        self.action_space = gymnasium.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        self.resets: list[dict] = []
        self.actions: list = []
        self.success = True

    def reset(self, *, seed=None, options=None):
        self.resets.append(dict(options or {}))
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        self.actions.append(action)
        return np.zeros(1, dtype=np.float32), 1.0, True, False, {
            "goal_reached": self.success
        }


def _ladder():
    return Curriculum.from_document(
        {
            "stages": [
                {
                    "name": "a",
                    "scenario": "default",
                    "episodes": 2,
                    "options": {"launch": True},
                },
                {"name": "b", "scenario": "front_approach", "episodes": 1},
            ]
        }
    )


def test_reset_hands_the_stage_options_to_the_environment():
    env = CurriculumEnv(FakeEnv(), _ladder())
    env.reset(seed=7)
    assert env.env.resets[0] == {"scenario": "default", "launch": True}


def test_caller_options_win_over_the_stage_defaults():
    env = CurriculumEnv(FakeEnv(), _ladder())
    env.reset(options={"scenario": "special"})
    assert env.env.resets[0]["scenario"] == "special"
    assert env.env.resets[0]["launch"] is True


def test_step_feeds_a_finished_episode_to_the_curriculum():
    curriculum = _ladder()
    env = CurriculumEnv(FakeEnv(), curriculum)
    env.reset()
    env.step([0.0, 0.0])
    assert curriculum.state()["episodes"] == 1
    assert curriculum.state()["successes"] == 1


def test_the_scenario_changes_once_the_stage_advances():
    curriculum = _ladder()
    env = CurriculumEnv(FakeEnv(), curriculum)
    env.reset()
    assert env.env.resets[-1]["scenario"] == "default"

    env.step([0.0, 0.0])  # episode 1 of 2 on stage a
    env.reset()
    assert env.env.resets[-1]["scenario"] == "default"

    env.step([0.0, 0.0])  # episode 2 of 2 -> promote to b
    assert curriculum.index == 1
    env.reset()
    assert env.env.resets[-1]["scenario"] == "front_approach"

    env.step([0.0, 0.0])  # the last episode -> finished
    assert curriculum.finished is True


def test_curriculum_state_is_exposed_for_logging():
    env = CurriculumEnv(FakeEnv(), _ladder())
    env.reset()
    assert env.curriculum_state["name"] == "a"
    assert env.curriculum_state["stage"] == 0


def test_a_stage_scenario_fields_are_copied_onto_the_environment():
    class Fielded(FakeEnv):
        scenario_fields = {"old": 1}
        _scenario_written = True

    env = CurriculumEnv(
        Fielded(),
        Curriculum.from_document(
            {
                "stages": [
                    {
                        "name": "a",
                        "scenario": "default",
                        "scenario_fields": {"humans": 3},
                    }
                ]
            }
        ),
    )
    env.reset()
    assert env.env.scenario_fields == {"humans": 3}
    assert env.env._scenario_written is False


def test_a_curriculum_is_required():
    with pytest.raises(TypeError):
        CurriculumEnv(FakeEnv(), object())


def test_success_is_read_from_each_plausible_key():
    assert episode_succeeded({"success": True}) is True
    assert episode_succeeded({"goal_reached": True}) is True
    assert episode_succeeded({"reached_goal": True}) is True
    assert episode_succeeded({"goal": 1}) is True
    assert episode_succeeded({"outcome": "goal"}) is True
    assert episode_succeeded({"goal_reached": False}) is False


def test_failure_keys_beat_the_terminated_fallback():
    assert episode_succeeded({"collision": True}, terminated=True) is False
    assert episode_succeeded({"timeout": True}, terminated=True) is False
    assert episode_succeeded({}, terminated=True) is True
    assert episode_succeeded({}, terminated=False) is False


def test_a_goal_position_is_not_mistaken_for_a_boolean():
    assert episode_succeeded({"goal": [1.0, 2.0]}, terminated=False) is False


def test_the_names_are_exported_lazily_from_the_package():
    from robotsnap.rl import Curriculum as ExportedCurriculum
    from robotsnap.rl import CurriculumEnv as ExportedEnv
    from robotsnap.rl import CurriculumError as ExportedError
    from robotsnap.rl import Stage as ExportedStage

    assert ExportedCurriculum is Curriculum
    assert ExportedError is CurriculumError
    assert ExportedEnv is CurriculumEnv
    assert ExportedStage.__name__ == "Stage"


def test_an_unknown_package_attribute_is_refused():
    import robotsnap.rl as rl

    with pytest.raises(AttributeError):
        rl.no_such_name


def test_importing_the_package_does_not_pull_gymnasium():
    environment = dict(os.environ)
    environment["PYTHONPATH"] = (
        str(REPO_ROOT / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    )
    probe = (
        "import sys, robotsnap.rl; print('gymnasium' in sys.modules); "
        "import robotsnap.rl.curriculum; print('gymnasium' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=environment,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["False", "False"]
