"""Tests for the shared trainer of ``robotsnap.models.social``.

The trainer is duck-typed - it asks its environment to ``reset``, to ``step``
and for nothing else - so the loop that trains a policy can be checked here
against a stand-in environment, without Unity and without a socket. What is
worth checking is exactly what a real run depends on: that every episode is
logged, that the history lines up column for column, that a training episode
writes to the replay buffer while a played-back one does not, and that a saved
agent is what plays.
"""

import numpy as np
import pytest

pytest.importorskip("torch")

from robotsnap.models.cadrl import CadrlAgent  # noqa: E402 - after the torch skip
from robotsnap.models.sarl import SarlAgent  # noqa: E402
from robotsnap.models.social import (  # noqa: E402 - after the torch skip
    EGO_SIZE,
    GOAL_SIZE,
    HISTORY_KEYS,
    NEIGHBOUR_SIZE,
    play,
    train,
)


class StubEnv:
    """The smallest environment the trainer can drive: a fixed-length episode."""

    def __init__(self, steps: int = 4, reward: float = 1.0):
        self.steps = int(steps)
        self.reward = float(reward)
        self.episodes = 0
        self._left = 0
        self._rng = np.random.default_rng(0)

    def reset(self, *, seed=None, options=None):
        self.episodes += 1
        self._left = self.steps
        return self._observation(), {}

    def step(self, action):
        self._left -= 1
        done = self._left <= 0
        info = {
            "goal_reached": done,
            "collision": False,
            "out_of_bounds": False,
            "action": (0.0, 0.0),
        }
        return self._observation(), self.reward, done, False, info

    def _observation(self):
        mask = np.zeros(4, dtype=bool)
        mask[:2] = True
        return {
            "ego": self._rng.normal(size=EGO_SIZE).astype(np.float32),
            "goal": self._rng.normal(size=GOAL_SIZE).astype(np.float32),
            "neighbours": self._rng.normal(size=(4, NEIGHBOUR_SIZE)).astype(np.float32),
            "mask": mask,
        }


@pytest.mark.parametrize("cls", [CadrlAgent, SarlAgent])
def test_train_runs_every_episode_and_returns_one_column_per_key(cls):
    env = StubEnv(steps=4, reward=2.5)
    agent = cls(actions=5, hidden=8, batch_size=4, seed=0)
    lines: list[str] = []
    history = train(env, agent, episodes=3, seed=11, log=lines.append)

    assert env.episodes == 3
    assert len(lines) == 3
    assert all("episode" in line and "outcome goal" in line for line in lines)
    assert tuple(history) == HISTORY_KEYS
    assert history["episode"] == [0, 1, 2]
    assert history["reward"] == pytest.approx([10.0, 10.0, 10.0])
    assert history["steps"] == [4, 4, 4]
    assert history["outcome"] == ["goal", "goal", "goal"]
    assert all(epsilon < agent.epsilon_start for epsilon in history["epsilon"])


def test_train_learns_once_the_buffer_holds_a_batch():
    env = StubEnv(steps=8)
    agent = CadrlAgent(actions=4, hidden=8, batch_size=4, seed=0)
    history = train(env, agent, episodes=1, log=None)
    assert np.isfinite(history["loss"][0])
    assert agent.epsilon < agent.epsilon_start


def test_train_can_be_capped_by_max_steps():
    env = StubEnv(steps=10)
    agent = CadrlAgent(actions=4, hidden=8, batch_size=4, seed=0)
    history = train(env, agent, episodes=1, max_steps=3, log=None)
    assert history["steps"] == [3]
    assert history["outcome"] == ["timeout"]


def test_play_loads_a_checkpoint_and_leaves_it_alone(tmp_path):
    env = StubEnv(steps=3)
    agent = SarlAgent(actions=5, hidden=8, seed=0)
    path = tmp_path / "agent.pt"
    agent.save(path)
    before = agent.epsilon

    lines: list[str] = []
    history = play(path, env, episodes=2, log=lines.append)

    assert env.episodes == 2
    assert len(lines) == 2
    assert history["steps"] == [3, 3]
    assert history["outcome"] == ["goal", "goal"]
    assert history["epsilon"] == pytest.approx([before, before])
    assert agent.epsilon == pytest.approx(before)
