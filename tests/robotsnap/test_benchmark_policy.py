"""How ``benchmark`` decides what a policy needs before the session is built.

The three writers of a checkpoint do not share an environment: a named method
brings its own observation and action space, and a value-based rule needs the
discrete command set. Getting that wrong is a campaign that scores nothing while
looking like it ran, so what the resolution promises is checked here rather than
against a live session.
"""

from pathlib import Path

import pytest

from robotsnap.runs.campaign import _policy_and_environment


def _tiny_sb3_checkpoint(tmp_path, algo: str, sb3, *, stamp: bool = True) -> Path:
    """A real, tiny checkpoint of ``algo``, the way ``test_sb3`` builds one."""
    import gymnasium

    from robotsnap.rl import sb3 as sb3_support

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


def test_the_two_floors_need_no_checkpoint():
    for name in ("scripted", "random"):
        chosen = _policy_and_environment(
            policy=name, load=None, method=None, algo="auto"
        )
        assert chosen.label == name
        assert chosen.make_act is None
        assert chosen.discrete is False


def test_a_policy_that_does_not_exist_is_refused():
    with pytest.raises(ValueError, match="unknown policy"):
        _policy_and_environment(policy="maddpg", load=None, method=None, algo="auto")


def test_a_checkpoint_policy_without_a_file_is_refused():
    with pytest.raises(ValueError, match="--load"):
        _policy_and_environment(
            policy="checkpoint", load=None, method=None, algo="auto"
        )


def test_an_unknown_method_is_refused(tmp_path):
    pytest.importorskip("torch")
    with pytest.raises(ValueError, match="unknown method"):
        _policy_and_environment(
            policy="checkpoint",
            load=str(tmp_path / "whatever.pt"),
            method="maddpg",
            algo="auto",
        )


def test_a_method_checkpoint_brings_its_own_environment(tmp_path):
    """CADRL's checkpoint says CADRL, so the run builds CADRL's task."""
    pytest.importorskip("torch")
    from robotsnap.models import make_agent
    from robotsnap.models.cadrl import CadrlEnv
    from robotsnap.models.social import SocialActionSpace

    class _FakeEnv:
        social_actions = SocialActionSpace()

    path = tmp_path / "cadrl.pt"
    make_agent("cadrl", _FakeEnv(), hidden=8, seed=0).save(path)

    chosen = _policy_and_environment(
        policy="checkpoint", load=str(path), method=None, algo="auto"
    )
    assert chosen.environment_class is CadrlEnv
    assert chosen.label.startswith("method:cadrl:")
    assert chosen.discrete is False
    assert chosen.make_act is not None


def test_a_value_based_checkpoint_asks_for_the_command_set(tmp_path):
    sb3 = pytest.importorskip("stable_baselines3")
    path = _tiny_sb3_checkpoint(tmp_path, "dqn", sb3)
    chosen = _policy_and_environment(
        policy="checkpoint", load=str(path), method=None, algo="auto"
    )
    assert chosen.discrete is True
    assert chosen.label.startswith("sb3:")


def test_a_continuous_checkpoint_keeps_the_box_action_space(tmp_path):
    sb3 = pytest.importorskip("stable_baselines3")
    path = _tiny_sb3_checkpoint(tmp_path, "ppo", sb3)
    chosen = _policy_and_environment(
        policy="checkpoint", load=str(path), method=None, algo="auto"
    )
    assert chosen.discrete is False
