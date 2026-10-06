"""``external_control``: the Python side paces and measures, a peer commands.

An externally driven episode is one whose velocity is published by a process
outside Python - a ROS2 node driving the bridge's ``cmd_vel``. The environment
must then release the pacing and read the world without writing a command that
would race the one the external policy sends, while every other part of a step
stays the step it always was. The default mode must keep sending the command,
so these tests are written against the same fake Unity peer the environment
suite uses, in ``test_env.py``, whose session helpers are reused here.
"""

import importlib.util
import time
from pathlib import Path

import pytest

from robotsnap.envs import RobotSNAPEnv

#: The environment suite's helpers: the fake lockstep gate, the world publisher
#: and the ``/cmd_vel`` reader. Loaded by path so the sibling module keeps its
#: own name and pytest still collects it once.
_HELPER_PATH = Path(__file__).with_name("test_env.py")
_HELPER_SPEC = importlib.util.spec_from_file_location(
    "_robotsnap_env_helpers", _HELPER_PATH
)
helpers = importlib.util.module_from_spec(_HELPER_SPEC)
_HELPER_SPEC.loader.exec_module(helpers)


def _cmd_vel_frames(peer):
    """The ``/cmd_vel`` frames the peer has read, in arrival order."""
    return [frame for frame in peer.inbound if frame[0].strip("/") == "cmd_vel"]


def _release_commands(peer):
    """The ``release_pacing`` commands the session was sent."""
    return [
        body
        for _, body in peer.commands
        if isinstance(body, dict) and body.get("command") == "release_pacing"
    ]


def test_an_external_lockstep_step_writes_no_command_but_releases_the_pacing(
    client, unity, bridge
):
    """The whole point of the mode: pace and read, never command."""
    environment = helpers._lockstep_env(client, external_control=True)
    gate = helpers._start_lockstep(unity, bridge, period=helpers._LOCKSTEP_PERIOD)
    try:
        environment.reset()
        # The episode-start command is sent, so wait for the peer to have read
        # it before taking the baseline the step must not move.
        helpers._wait_for(lambda: _cmd_vel_frames(unity))
        before = list(_cmd_vel_frames(unity))
        releases_before = len(_release_commands(unity))

        observation, reward, terminated, truncated, info = environment.step([0.0, 0.0])

        # Give any frame the step could have written the same time the peer
        # needs to read one, then prove none appeared.
        helpers._wait_for(lambda: gate.releases > 0)
        assert _cmd_vel_frames(unity) == before
        # The period was still released: the gate spent it and moved the world.
        assert len(_release_commands(unity)) > releases_before
        assert environment.observation_space.contains(observation)
        assert info["episode_seconds"] == pytest.approx(helpers._LOCKSTEP_PERIOD)
    finally:
        gate.stop()
        environment.close()


def test_the_default_mode_still_sends_the_command(client, unity, bridge):
    """The historical behaviour, step for step, on the same fake session."""
    helpers._start(unity, bridge, x=0.0, y=0.0, goal=(5.0, 0.0))
    environment = RobotSNAPEnv(
        client=client,
        scenario="demo",
        control_period=helpers._CONTROL_PERIOD,
        wait_timeout=helpers._WAIT_TIMEOUT,
    )
    try:
        assert environment.external_control is False
        environment.reset()
        helpers._publish_world(unity, x=1.0, y=0.0, goal=(5.0, 0.0), sim_time=10.5)

        observation, reward, terminated, truncated, info = environment.step([0.5, 0.2])

        twist = helpers._last_twist(unity)
        assert (twist.linear.x, twist.angular.z) == pytest.approx((0.5, 0.2))
        assert environment.observation_space.contains(observation)
    finally:
        environment.close()


def test_an_external_episode_resets_and_steps_into_an_observation(client, unity, bridge):
    """A reset then a step answer as usual, with no command written."""
    helpers._start(unity, bridge, x=0.0, y=0.0, goal=(5.0, 0.0))
    environment = RobotSNAPEnv(
        client=client,
        scenario="demo",
        control_period=helpers._CONTROL_PERIOD,
        wait_timeout=helpers._WAIT_TIMEOUT,
        external_control=True,
    )
    try:
        observation, info = environment.reset()
        assert environment.observation_space.contains(observation)
        assert info["distance_to_goal"] == pytest.approx(5.0)

        helpers._wait_for(lambda: _cmd_vel_frames(unity))
        before = list(_cmd_vel_frames(unity))
        helpers._publish_world(unity, x=1.0, y=0.0, goal=(5.0, 0.0), sim_time=10.5)
        observation, reward, terminated, truncated, info = environment.step([0.0, 0.0])

        # The peer reads on its own thread: give a command the step might have
        # written the moment it needs to be seen, then prove none arrived.
        time.sleep(0.05)
        assert _cmd_vel_frames(unity) == before
        assert environment.observation_space.contains(observation)
        assert info["distance_to_goal"] == pytest.approx(4.0)
    finally:
        environment.close()
