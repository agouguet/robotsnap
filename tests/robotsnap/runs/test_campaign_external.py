"""The ``external`` policy: what a campaign builds and what it hands the step.

Measured against a process outside Python, the campaign still owns the suite,
the pacing and the metrics. These tests check the binding - which environment
class, which label, that the environment is told to command nothing - and that
the neutral action the loop still passes is shaped like the environment's own
space. Nothing here opens a session.
"""

import numpy as np
import pytest
from gymnasium import spaces

from robotsnap import cli
from robotsnap.envs import RobotSNAPEnv
from robotsnap.runs import campaign as campaign_support
from robotsnap.runs.campaign import _neutral_action, _policy_and_environment


class _FakeEnvironment:
    """Just the one attribute a neutral action is derived from."""

    def __init__(self, space):
        self.action_space = space


def test_the_external_policy_builds_the_plain_environment():
    chosen = _policy_and_environment(
        policy="external", load=None, method=None, algo="auto"
    )
    assert chosen.environment_class is RobotSNAPEnv
    assert chosen.label == "external"
    assert chosen.external_control is True
    assert chosen.make_act is not None


def test_ros2_is_an_alias_of_the_external_policy():
    chosen = _policy_and_environment(policy="ros2", load=None, method=None, algo="auto")
    assert chosen.environment_class is RobotSNAPEnv
    assert chosen.label == "external"
    assert chosen.external_control is True


def test_the_other_policies_keep_owning_their_command():
    for name in ("scripted", "random"):
        chosen = _policy_and_environment(
            policy=name, load=None, method=None, algo="auto"
        )
        assert chosen.external_control is False


def test_the_neutral_action_is_zeros_for_a_box():
    space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    action = _neutral_action(_FakeEnvironment(space))
    assert action.shape == (2,)
    assert np.all(action == 0.0)
    assert space.contains(action)


def test_the_neutral_action_is_zero_for_a_discrete_space():
    space = spaces.Discrete(7)
    action = _neutral_action(_FakeEnvironment(space))
    assert action == 0
    assert space.contains(action)


def test_the_external_make_act_answers_a_neutral_action():
    space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    chosen = _policy_and_environment(
        policy="external", load=None, method=None, algo="auto"
    )
    act = chosen.make_act(_FakeEnvironment(space))
    action = act(observation=None, info=None)
    assert space.contains(action)
    assert np.all(action == 0.0)


def test_run_scenarios_hands_the_external_mode_to_the_environment(monkeypatch, tmp_path):
    """``run_scenarios`` must forward ``external_control`` to the constructor options."""
    from robotsnap.analysis import suites as suite_support
    from robotsnap.runs import session as session_support

    captured = {}

    def spy(**kwargs):
        captured.update(kwargs)
        return kwargs

    class _Stop(Exception):
        """Raised where the environment would be built, to stop before a session."""

    def stop(*args, **kwargs):
        raise _Stop

    monkeypatch.setattr(campaign_support, "_scenario_directory", lambda project: tmp_path)
    monkeypatch.setattr(suite_support, "scenarios_for", lambda suite: ("demo",))
    monkeypatch.setattr(campaign_support, "_environment_options", spy)
    monkeypatch.setattr(session_support, "_build_environment", stop)

    with pytest.raises(_Stop):
        campaign_support.run_scenarios(policy="external", episodes=1)

    assert captured["external_control"] is True


def test_the_benchmark_parser_accepts_the_external_policies():
    parser = cli.build_parser()
    for spelling in ("external", "ros2"):
        args = parser.parse_args(["benchmark", "--policy", spelling])
        assert args.policy == spelling


def test_the_benchmark_parser_refuses_an_unknown_policy():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["benchmark", "--policy", "autre"])
