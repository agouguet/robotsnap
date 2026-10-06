"""The training runs' end of the curriculum: a name resolves, and wraps.

What a curriculum *is* lives in ``tests/robotsnap/rl``; what this file checks is
that the three training loops reach it - the flag survives the command line, the
name resolves to the file under ``configs/curriculum``, and the environment a
run builds is the one the wrapper supplies scenarios to.
"""

import gymnasium
import numpy as np

from robotsnap import cli, runs
from robotsnap.runs.session import _apply_curriculum, _curriculum


class _FakeEnvironment(gymnasium.Env):
    """The smallest environment a curriculum wrapper will accept."""

    def __init__(self):
        self.observation_space = gymnasium.spaces.Box(-1.0, 1.0, shape=(2,))
        self.action_space = gymnasium.spaces.Box(-1.0, 1.0, shape=(2,))
        self.resets: list[dict] = []

    def reset(self, *, seed=None, options=None):
        self.resets.append(dict(options or {}))
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(2, dtype=np.float32), 0.0, True, False, {"goal_reached": True}


def test_a_curriculum_name_resolves_and_wraps_the_environment():
    assert _curriculum(None) is None

    plan = _curriculum("social_navigation")
    assert plan.stages
    assert plan.current.scenario

    environment = _FakeEnvironment()
    assert _apply_curriculum(environment, None) is environment

    wrapped = _apply_curriculum(environment, plan)
    assert wrapped is not environment
    wrapped.reset()
    assert environment.resets[-1]["scenario"] == plan.current.scenario


def test_train_forwards_a_curriculum_to_every_training_loop(monkeypatch):
    """One flag, three loops: REINFORCE, Stable-Baselines3 and a named method."""
    seen = {}

    def recording(loop):
        def fake(**kwargs):
            seen[loop] = kwargs
            return 0

        return fake

    monkeypatch.setattr(runs, "run_training", recording("reinforce"))
    monkeypatch.setattr(runs, "run_sb3_training", recording("sb3"))
    monkeypatch.setattr(runs, "run_social_training", recording("social"))

    assert (
        cli.main(["train", "--episodes", "1", "--curriculum", "social_navigation"])
        == 0
    )
    assert (
        cli.main(
            ["train", "--algo", "ppo", "--timesteps", "10", "--curriculum", "crowd"]
        )
        == 0
    )
    assert (
        cli.main(["train", "--method", "cadrl", "--episodes", "1", "--curriculum", "crowd"])
        == 0
    )

    assert seen["reinforce"]["curriculum"] == "social_navigation"
    assert seen["sb3"]["curriculum"] == "crowd"
    assert seen["social"]["curriculum"] == "crowd"
