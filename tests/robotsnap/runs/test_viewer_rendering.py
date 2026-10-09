"""Every loop that can run with ``--viewer`` draws the window it was asked for.

The package's own loops call ``render`` themselves. The two that do not own
their loop - Stable-Baselines3's ``learn`` and ``play`` - and the shared social
trainer, reach the same window through the seams pinned here. Nothing opens a
Unity session: what is checked is that the frame is drawn once per step, on an
environment that counts the calls, and never on one that asked for no window.
"""

from robotsnap.runs.inference_sb3 import _viewer_step
from robotsnap.runs.training_sb3 import _one_callback, _viewer_callback


class _CountingEnv:
    """An environment whose only job here is to say how often it was drawn."""

    def __init__(self) -> None:
        self.frames = 0

    def render(self) -> None:
        self.frames += 1


def test_the_training_callback_draws_one_frame_per_step():
    env = _CountingEnv()
    callback = _viewer_callback(env)

    assert callback._on_step() is True
    assert callback._on_step() is True
    assert env.frames == 2


def test_the_played_back_policy_draws_one_frame_per_step():
    env = _CountingEnv()
    on_step = _viewer_step(env)

    on_step(1, 0.0)
    on_step(2, 1.0)
    assert env.frames == 2


def test_the_run_callbacks_are_the_single_one_stable_baselines3_takes():
    from stable_baselines3.common.callbacks import CallbackList

    alone = _CountingEnv()
    assert _one_callback([None, None]) is None
    assert _one_callback([None, alone]) is alone
    assert isinstance(_one_callback([alone, alone]), CallbackList)
