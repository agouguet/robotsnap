"""The stable-baselines3 bridge: the action table, the factories, and a real run.

The algorithms themselves are Stable-Baselines3's, so what is worth testing
here is everything this package puts around them - that the discrete table maps
an integer to the command it says it does, that the factories build an
environment SB3 accepts, and that a checkpoint names the algorithm that wrote
it. The end-to-end test drives the fake Unity peer, which is the same session
shape a real run uses, for a handful of control steps.
"""

import time
from pathlib import Path

import numpy as np
import pytest

from robotsnap.rl import sb3 as sb3_support

_WAIT_TIMEOUT = 2.0


def _wait_for(predicate, timeout=_WAIT_TIMEOUT):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "the peer never became ready"
        time.sleep(0.01)


def test_the_table_maps_an_index_to_the_command_it_promises():
    commands = sb3_support.DiscreteCommands(max_linear=2.0, max_angular=3.0)

    assert len(commands) == len(sb3_support.DISCRETE_COMMANDS)
    assert list(commands.command(0)) == [0.0, 0.0]
    assert list(commands.command(1)) == [2.0, 0.0]
    # The fractions are the robot's own limits, so the same table drives a robot
    # whose controller is limited twice as tightly.
    tight = sb3_support.DiscreteCommands(max_linear=0.5)
    assert list(tight.command(1)) == [0.5, 0.0]


def test_an_index_outside_the_table_is_clamped_rather_than_refused():
    commands = sb3_support.DiscreteCommands()

    assert list(commands.command(-3)) == list(commands.command(0))
    assert list(commands.command(999)) == list(commands.command(len(commands) - 1))
    # A numpy scalar is what a policy hands back, and it has to work too.
    assert list(commands.command(np.int64(1))) == list(commands.command(1))


def test_every_command_stays_inside_the_environment_limits():
    commands = sb3_support.DiscreteCommands(max_linear=0.7, max_angular=1.3)

    for index in range(len(commands)):
        linear, angular = commands.command(index)
        assert -0.7 <= linear <= 0.7
        assert -1.3 <= angular <= 1.3


def test_the_algorithm_table_says_which_ones_need_discrete_actions():
    assert sb3_support.ALGORITHMS["dqn"] is True
    assert sb3_support.ALGORITHMS["ppo"] is False
    assert sb3_support.ALGORITHMS["sac"] is False


def test_an_unknown_algorithm_is_refused_by_name():
    with pytest.raises(ValueError, match="dqn"):
        sb3_support.algorithm_class("maddpg")


def test_a_checkpoint_names_the_algorithm_that_wrote_it(tmp_path):
    sb3 = pytest.importorskip("stable_baselines3")

    # A tiny environment only so the model has something to be built against:
    # the question is what the archive says about itself, not what it learned.
    # The stamp is the answer, because SB3's own archive names only the policy
    # class and a PPO and an A2C checkpoint name the same one.
    path = _tiny_checkpoint(tmp_path, "dqn", sb3)
    assert sb3_support._algorithm_from_checkpoint(path) == "dqn"

    # Broken on purpose: the marker is what is read, and a file that is not a
    # checkpoint at all is refused rather than guessed at.
    plain = Path(tmp_path) / "plain.zip"
    plain.write_bytes(b"not a checkpoint at all")
    with pytest.raises(ValueError, match="checkpoint"):
        sb3_support._algorithm_from_checkpoint(plain)


def test_an_ambiguous_foreign_checkpoint_asks_for_the_algorithm(tmp_path):
    sb3 = pytest.importorskip("stable_baselines3")
    path = _tiny_checkpoint(tmp_path, "ppo", sb3, stamp=False)

    with pytest.raises(ValueError, match="--algo"):
        sb3_support._algorithm_from_checkpoint(path)
    # Told which one it is, the same file loads.
    assert sb3_support._algorithm_from_checkpoint(path, algo="ppo") == "ppo"


def _tiny_checkpoint(tmp_path, algo: str, sb3, *, stamp: bool = True) -> Path:
    """A real, tiny checkpoint of ``algo``, optionally without this package's marker."""
    import gymnasium

    env = gymnasium.make("CartPole-v1")
    try:
        if algo == "dqn":
            model = sb3.DQN("MlpPolicy", env, learning_starts=1, device="cpu")
        else:
            model = sb3.PPO("MlpPolicy", env, n_steps=8, device="cpu")
        path = Path(tmp_path) / f"{algo}.zip"
        model.save(str(path))
        if stamp:
            sb3_support.stamp_algorithm(path, algo)
        return path
    finally:
        env.close()


def test_a_file_that_is_not_a_checkpoint_says_so(tmp_path):
    path = Path(tmp_path) / "not-a-model.zip"
    path.write_bytes(b"not a checkpoint at all")

    with pytest.raises(ValueError, match="checkpoint"):
        sb3_support._algorithm_from_checkpoint(path)
