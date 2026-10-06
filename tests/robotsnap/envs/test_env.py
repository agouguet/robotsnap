"""Tests for ``robotsnap.envs.base.RobotSNAPEnv``, against the fake Unity peer."""

import socket
import threading
import time

import numpy as np
import pytest
from gymnasium import spaces

from robotsnap import topics
from robotsnap.bridge import codec
from robotsnap.client import CONTROL_RESULT_TOPIC, RobotSNAPClient
from robotsnap.envs import ENV_ID, RobotSNAPEnv, RobotSNAPEnvError, register_envs
from robotsnap.envs.base import _lockstep_grace

_WAIT_TIMEOUT = 2.0
_CONTROL_PERIOD = 0.05
#: The control period the lockstep tests use. A tenth of a second makes the
#: budget arithmetic below exact: five periods is half a second.
_LOCKSTEP_PERIOD = 0.1
#: The period and the budget the boundary tests of the episode clock step with.
#: Twenty seconds at a fifth of a second a step is a hundred steps, and it is
#: the case measured in the live session: the world's clock lands a hair under
#: the budget, so a strict comparison spends a hundred and one.
_BUDGET = 20.0
_BUDGET_PERIOD = 0.2


@pytest.fixture
def env(client, unity):
    """An environment over the fixture bridge, with no scenario of its own."""
    environment = RobotSNAPEnv(
        client=client,
        scenario="demo",
        control_period=_CONTROL_PERIOD,
        wait_timeout=_WAIT_TIMEOUT,
    )
    try:
        yield environment
    finally:
        environment.close()


def _wait_for(predicate, timeout=_WAIT_TIMEOUT):
    deadline = time.monotonic() + timeout
    while True:
        value = predicate()
        if value:
            return value
        if time.monotonic() >= deadline:
            return value
        time.sleep(0.01)


def _state(
    x=0.0,
    y=0.0,
    yaw=0.0,
    goal=(5.0, 0.0),
    sim_time=10.0,
    scenario_id="demo",
    applied=True,
    time_scale=1.0,
    held=None,
    robots=None,
):
    """A snapshot as ``/simulation/state`` publishes it, roster included.

    ``held`` is the session's own flag for a world stopped at a zero time
    scale. Left as ``None`` the key is left out entirely, which is what a build
    older than the flag publishes, so a test can ask for both shapes.
    """
    entry = {
        "id": "robot_1",
        "type": "jackal",
        "is_primary": True,
        "x": x,
        "y": y,
        "z": 0.0,
        "yaw": yaw,
        "has_goal": goal is not None,
        "goal": None if goal is None else {"x": goal[0], "y": goal[1], "z": 0.0},
        "start_pose": None,
        "target_pose": None,
    }
    body = {
        "simulation_state": "running",
        "playing": applied,
        "paused": False,
        "stopped": not applied,
        "scenario_applied": applied,
        "scenario_id": scenario_id,
        "scenario_name": scenario_id,
        "sim_time_seconds": sim_time,
        "time_scale": time_scale,
        "map_width": 4,
        "map_height": 4,
        "map_resolution": 1.0,
        "map_origin_x": 0.0,
        "map_origin_y": 0.0,
        "human_count": 0,
        "humans": [],
        "robots": [entry] if robots is None else robots,
    }
    if held is not None:
        body["held"] = bool(held)
    return body


def _publish_world(
    unity,
    x=0.0,
    y=0.0,
    yaw=0.0,
    goal=(5.0, 0.0),
    sim_time=10.0,
    ranges=None,
    cells=None,
    scenario_id="demo",
    applied=True,
    time_scale=1.0,
    held=None,
    robots=None,
    with_scan=True,
    with_map=True,
):
    """Publish the whole world one environment read needs, in one go."""
    unity.publish_state(
        _state(
            x=x,
            y=y,
            yaw=yaw,
            goal=goal,
            sim_time=sim_time,
            scenario_id=scenario_id,
            applied=applied,
            time_scale=time_scale,
            held=held,
            robots=robots,
        )
    )
    unity.publish_odom(x=x, y=y, yaw=yaw, linear_x=0.0, angular_z=0.0, stamp=sim_time)
    if with_scan:
        unity.publish_scan(
            ranges if ranges is not None else [3.0] * 8,
            angle_min=-np.pi / 4,
            angle_increment=np.pi / 16,
            range_max=5.0,
            stamp=sim_time,
        )
    if with_map:
        unity.publish_map(
            # Wide enough that the default goal of the fixtures is inside it:
            # a robot off the grid ends its episode on the out-of-bounds test.
            cells if cells is not None else [[0] * 12 for _ in range(12)],
            resolution=1.0,
            origin_x=0.0,
            origin_y=0.0,
            stamp=sim_time,
        )
    unity.publish_agents({"agents": [], "frame": "robot"})


def _start(world_peer, bridge, **world):
    """Bring the fake session up: reader thread, closed loop, one world."""
    _wait_for(
        lambda: bridge.topic_types().get("simulation/control_result")
        == "std_msgs/msg/String"
    )
    world_peer.start_responder()
    _publish_world(world_peer, **world)
    # Publishing is not reading: the bridge decodes on its own thread, and a test that reads the world
    # straight after the send could see a session that has published nothing yet. Wait for the snapshot
    # the rest of the read is built around, so the race stays out of the tests that follow.
    _wait_for(lambda: bridge.topic_counts().get("simulation/state", 0) > 0)


def _last_twist(peer, timeout=_WAIT_TIMEOUT):
    """The last ``/cmd_vel`` the peer read, decoded.

    Waits for it: the peer reads the socket on its own thread, so a command the
    environment has just written may not have been picked up yet when a test
    looks.
    """

    def seen():
        return [
            frame for frame in peer.inbound if frame[0].strip("/") == "cmd_vel"
        ]

    frames = _wait_for(seen, timeout)
    assert frames, "the peer never read a /cmd_vel"
    return codec.decode(topics.TWIST_TYPE, frames[-1][1])


def _publish_later(unity, delay=0.05, **world):
    """Publish a world from a thread, to feed the session while reset() blocks."""

    def run():
        time.sleep(delay)
        _publish_world(unity, **world)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


class _LockstepGate:
    """A stand-in for ``SimulationPacingGate``, for the lockstep tests.

    One release spends one control period: the gate watches the control topic
    for ``release_pacing``, moves the world's clock a full period and publishes
    the snapshots the real gate produces. ``held_reports`` is what those
    snapshots carry: the first while the world is still running and the last
    once the gate has stopped it. The real gate publishes the clock the period
    spent with ``held`` false and then repeats it with ``held`` true; a single
    entry models a session that stops the world at once - an older build with no
    ``held`` key at all (``None``) or one that stops it but keeps saying
    ``False``.

    ``early`` counts the releases that arrived while a period was still being
    spent. That count must stay at zero: the real ``Release`` re-arms the gate's
    step counter, so a release that goes out before the previous period has
    stopped the world lets the world run on between two steps and makes every
    step cover more than its period - the drift the environment must not fall
    into.

    ``spend`` is the wall window the world stays "still running" after the
    clock moved, and it has to be longer than the delay the fake socket puts on
    a small write (tens of milliseconds of Nagle and delayed acknowledgment), or
    a release that went out far too early would simply not be seen arriving
    early. The tests use a fifth of a second, several times that delay.
    """

    def __init__(
        self,
        unity,
        *,
        period,
        sim_time,
        publish,
        held_reports=(False, True),
        spend=0.15,
    ):
        self.unity = unity
        self.period = float(period)
        self.sim_time = float(sim_time)
        self.publish = publish
        self.held_reports = tuple(held_reports)
        self.spend = float(spend)
        self.releases = 0
        self.early = 0
        self._seen = 0
        self._spending = False
        self._stopped_at = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._loop, name="fake-lockstep-gate", daemon=True
        )

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()

    def _releases_seen(self):
        return sum(
            1
            for _, body in self.unity.commands
            if isinstance(body, dict) and body.get("command") == "release_pacing"
        )

    def _loop(self):
        """Spend every release, and count the ones that arrive before the world stops."""
        while not self._stop.is_set():
            if self._spending:
                self._note_early_releases()
                if time.monotonic() >= self._stopped_at:
                    self._stop_the_world()
                time.sleep(0.002)
                continue
            if not self._take_one_release():
                time.sleep(0.002)

    def _take_one_release(self) -> bool:
        """Spend one waiting release: move the clock a period and say the world is moving again."""
        seen = self._releases_seen()
        if seen <= self._seen:
            return False
        self._seen += 1
        self.releases += 1
        self._spending = True
        self.sim_time += self.period
        self.publish(self.sim_time, self.held_reports[0])
        self._stopped_at = time.monotonic() + self.spend
        return True

    def _note_early_releases(self) -> None:
        """Count the releases that arrived with the period still in flight - the drift, on the wire."""
        seen = self._releases_seen()
        if seen > self._seen:
            self.early += seen - self._seen
            self.releases += seen - self._seen
            self._seen = seen

    def _stop_the_world(self) -> None:
        """Stop the world at zero scale: the clock is final and the last snapshot says so."""
        if len(self.held_reports) > 1:
            self.publish(self.sim_time, self.held_reports[-1])
        self._spending = False


# -- reset ------------------------------------------------------------------=


def test_reset_launches_the_scenario_of_a_session_that_has_none(env, unity, bridge):
    """The first reset of a fresh session loads the scenario and waits for it."""
    _start(unity, bridge, x=0.0, y=0.0, goal=(5.0, 0.0), applied=False, scenario_id=None)
    # Unity builds the world after the load, and reports it as the current one.
    _publish_later(unity, x=0.0, y=0.0, goal=(5.0, 0.0), sim_time=10.2)

    observation, info = env.reset(seed=7)

    assert env.observation_space.contains(observation)
    assert observation.dtype == np.float32
    assert info["distance_to_goal"] == pytest.approx(5.0)
    assert info["scenario_id"] == "demo"
    assert info["episode_seconds"] == pytest.approx(0.0)
    assert info["wall_seconds"] == pytest.approx(0.0)
    assert info["steps"] == 0
    assert [body["command"] for _, body in unity.commands][:2] == [
        "set_random_seed",
        "load_scenario",
    ]


def test_reset_reapplies_the_current_scenario_instead_of_reloading_it(
    env, unity, bridge
):
    """The cheap reset is the point: same start, no rebuild of the world."""
    _start(unity, bridge)

    env.reset()

    commands = [body["command"] for _, body in unity.commands]
    assert commands[0] == "reset"
    assert "load_scenario" not in commands


def test_reset_reloads_the_scenario_when_asked_to(client, unity, bridge):
    environment = RobotSNAPEnv(
        client=client,
        scenario="demo",
        control_period=_CONTROL_PERIOD,
        wait_timeout=_WAIT_TIMEOUT,
        reload_on_reset=True,
    )
    try:
        _start(unity, bridge)
        environment.reset()
        assert unity.commands[0][1]["command"] == "load_scenario"
    finally:
        environment.close()


def test_reset_takes_the_robot_in_hand(env, unity, bridge):
    """ROS control mode, a zero command and the episode's own clock."""
    _start(unity, bridge)

    env.reset(seed=3)

    bodies = [body for _, body in unity.commands]
    assert [body["command"] for body in bodies] == [
        "set_random_seed",
        "reset",
        "set_control_mode",
    ]
    assert bodies[0]["seed"] == 3
    assert (bodies[2]["mode"], bodies[2].get("robot")) == ("ros", None)
    assert env.client.scenario_time_seconds == pytest.approx(0.0)
    twist = _last_twist(unity)
    assert (twist.linear.x, twist.angular.z) == pytest.approx((0.0, 0.0))


def test_reset_without_a_scenario_says_what_to_do(client, unity, bridge):
    environment = RobotSNAPEnv(
        client=client, control_period=_CONTROL_PERIOD, wait_timeout=_WAIT_TIMEOUT
    )
    try:
        _start(unity, bridge, applied=False, scenario_id=None)
        with pytest.raises(RobotSNAPEnvError, match="no scenario is applied"):
            environment.reset()
    finally:
        environment.close()


# -- step -------------------------------------------------------------------=


def test_step_applies_the_action_and_pays_the_progress(env, unity, bridge):
    _start(unity, bridge, x=0.0, y=0.0, goal=(5.0, 0.0))
    env.reset()

    # The robot advanced one metre toward the goal while the step ran.
    _publish_world(unity, x=1.0, y=0.0, goal=(5.0, 0.0), sim_time=10.5)
    observation, reward, terminated, truncated, info = env.step([0.5, 0.2])

    assert env.observation_space.contains(observation)
    assert info["distance_to_goal"] == pytest.approx(4.0)
    assert info["episode_seconds"] > 0.0
    assert terminated is False and truncated is False
    # One metre of progress, less the price of the time the step took.
    assert reward == pytest.approx(1.0 - 0.1 * info["episode_seconds"])
    twist = _last_twist(unity)
    assert (twist.linear.x, twist.angular.z) == pytest.approx((0.5, 0.2))
    assert info["action"] == pytest.approx((0.5, 0.2))


def test_step_waits_for_the_control_period(env, unity, bridge):
    _start(unity, bridge)
    env.reset()

    started = time.monotonic()
    _, _, _, _, info = env.step([0.0, 0.0])
    elapsed = time.monotonic() - started

    assert elapsed >= _CONTROL_PERIOD * 0.9
    # The world is not publishing here, so the grace is what bounds the step: a
    # session that stopped publishing must not hold the loop for the whole
    # timeout.
    assert elapsed < _CONTROL_PERIOD * 4
    assert info["fresh"] is False


def test_a_step_that_sees_the_world_move_on_is_fresh(env, unity, bridge):
    _start(unity, bridge)
    env.reset()
    # A session publishing at its own rate while the action is applied: this is
    # what the real publishers do, ten state snapshots and fifty odometry a
    # second.
    _publish_later(unity, delay=_CONTROL_PERIOD / 4.0, sim_time=10.4)

    _, _, _, _, info = env.step([0.2, 0.0])

    assert info["fresh"] is True


def test_the_command_is_clipped_to_the_limits(env, unity, bridge):
    _start(unity, bridge)
    env.reset()

    env.step([9.0, -9.0])

    twist = _last_twist(unity)
    assert (twist.linear.x, twist.angular.z) == pytest.approx((1.0, -1.0))


def test_reaching_the_goal_ends_the_episode_with_its_bonus(env, unity, bridge):
    _start(unity, bridge, x=0.0, y=0.0, goal=(5.0, 0.0))
    env.reset()

    _publish_world(unity, x=4.8, y=0.0, goal=(5.0, 0.0), sim_time=11.0)
    _, reward, terminated, truncated, info = env.step([0.4, 0.0])

    assert info["goal_reached"] is True
    assert terminated is True and truncated is False
    assert reward > env.goal_reward


def test_the_grid_decides_a_collision(env, unity, bridge):
    """The cell under the robot holds a wall: that is the crash."""
    cells = [[0] * 8 for _ in range(8)]
    cells[2][2] = 100
    _start(unity, bridge, x=2.5, y=2.5, goal=(7.0, 0.0), cells=cells)
    env.reset()

    _, reward, terminated, truncated, info = env.step([0.2, 0.0])

    assert info["collision"] is True
    assert terminated is True and truncated is False
    assert reward < 0.0


def test_leaving_the_grid_is_out_of_bounds(env, unity, bridge):
    _start(unity, bridge, x=40.0, y=40.0, goal=(0.0, 0.0))
    env.reset()

    _, reward, terminated, _, info = env.step([0.5, 0.0])

    assert info["out_of_bounds"] is True
    assert terminated is True
    assert reward < 0.0


def test_a_robot_facing_a_wall_collides_without_a_grid(client, unity, bridge):
    """The lidar is the fallback when no grid has arrived."""
    environment = RobotSNAPEnv(
        client=client,
        scenario="demo",
        control_period=_CONTROL_PERIOD,
        wait_timeout=_WAIT_TIMEOUT,
        collision_distance=0.3,
    )
    try:
        _start(unity, bridge, ranges=[0.1] * 8, with_map=False)
        environment.reset()
        _, _, terminated, _, info = environment.step([0.1, 0.0])
        assert info["collision"] is True
        assert terminated is True
    finally:
        environment.close()


def test_the_episode_truncates_on_its_time_budget(client, unity, bridge):
    environment = RobotSNAPEnv(
        client=client,
        scenario="demo",
        control_period=_CONTROL_PERIOD,
        wait_timeout=_WAIT_TIMEOUT,
        max_episode_seconds=2.0,
    )
    try:
        _start(unity, bridge, sim_time=10.0)
        environment.reset()
        # The world lived past the budget while the step ran: the episode has to end on the simulator's
        # clock, which is the only one that says how far the world actually moved.
        _publish_world(unity, sim_time=12.4)
        _, _, terminated, truncated, info = environment.step([0.0, 0.0])
        assert terminated is False
        assert truncated is True
        assert info["episode_seconds"] >= 2.0
    finally:
        environment.close()


def test_the_episode_clock_is_the_simulators_and_not_the_wall(env, unity, bridge):
    """A step covers the simulated time the session reports, whatever the wall cost of getting there.

    The wall rule this replaces multiplied the seconds the step took by the scale the caller *asked* for, so
    a session that reached a tenth of its ask still rang up the full amount, and a stalled step added time
    the world never lived. The two answers are far apart here on purpose: five seconds of a world running at
    ten times speed, against the fifth of a second a control period of that session costs.
    """
    _start(unity, bridge, x=0.0, y=0.0, goal=(5.0, 0.0), sim_time=40.0, time_scale=10.0)
    env.reset()

    _publish_world(unity, x=1.0, y=0.0, goal=(5.0, 0.0), sim_time=45.0, time_scale=10.0)
    _, _, terminated, truncated, info = env.step([0.5, 0.0])

    assert terminated is False and truncated is False
    assert info["episode_seconds"] == pytest.approx(5.0)


def test_step_before_reset_is_refused(env):
    with pytest.raises(RobotSNAPEnvError, match="call reset"):
        env.step([0.0, 0.0])


def test_an_action_of_the_wrong_size_is_refused(env, unity, bridge):
    _start(unity, bridge)
    env.reset()
    with pytest.raises(ValueError, match="linear_x, angular_z"):
        env.step([1.0, 2.0, 3.0])


# -- what a subclass replaces -----------------------------------------------=


def test_a_subclass_can_replace_the_observation_and_the_reward(client, unity, bridge):
    """The whole point of the split: two methods, no simulator plumbing."""

    class LidarEnv(RobotSNAPEnv):
        def make_observation_space(self):
            return spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        def observation(self, world, task):
            closest = 1.0 if task.min_lidar is None else min(task.min_lidar, 1.0)
            return np.array([closest], dtype=np.float32)

        def reward(self, previous, current, action, world):
            return -1.0

    environment = LidarEnv(
        client=client, scenario="demo", control_period=_CONTROL_PERIOD, wait_timeout=_WAIT_TIMEOUT
    )
    try:
        _start(unity, bridge)
        observation, _ = environment.reset()
        assert environment.observation_space.shape == (1,)
        assert observation.shape == (1,)
        assert environment.observation_space.contains(observation)

        observation, reward, _, _, _ = environment.step([0.0, 0.0])
        assert observation.shape == (1,)
        assert reward == pytest.approx(-1.0)
    finally:
        environment.close()


def test_a_subclass_can_replace_the_action_space(client, unity, bridge):
    """A discrete policy changes two methods together: the space and the mapping."""

    class DiscreteEnv(RobotSNAPEnv):
        STEPS = ((0.5, 0.0), (0.0, 0.5), (0.0, 0.0))

        def make_action_space(self):
            return spaces.Discrete(len(self.STEPS))

        def action_to_command(self, action):
            return self.STEPS[int(action)]

    environment = DiscreteEnv(
        client=client, scenario="demo", control_period=_CONTROL_PERIOD, wait_timeout=_WAIT_TIMEOUT
    )
    try:
        _start(unity, bridge)
        environment.reset()
        assert environment.action_space.n == 3
        environment.step(1)
        twist = _last_twist(unity)
        assert (twist.linear.x, twist.angular.z) == pytest.approx((0.0, 0.5))
    finally:
        environment.close()


def test_the_environment_reads_the_robot_it_was_given(client, unity, bridge):
    """A fleet: the commands carry the id, and the streams are its own."""
    roster = [
        {
            "id": "robot_1",
            "type": "jackal",
            "is_primary": True,
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "yaw": 0.0,
            "has_goal": False,
            "goal": None,
        },
        {
            "id": "robot_2",
            "type": "kuri",
            "is_primary": False,
            "x": 3.0,
            "y": 4.0,
            "z": 0.0,
            "yaw": 0.0,
            "has_goal": True,
            "goal": {"x": 3.0, "y": 9.0, "z": 0.0},
        },
    ]
    environment = RobotSNAPEnv(
        client=client,
        scenario="demo",
        robot="robot_2",
        control_period=_CONTROL_PERIOD,
        wait_timeout=_WAIT_TIMEOUT,
    )
    try:
        _start(unity, bridge, robots=roster)
        _, info = environment.reset()

        assert info["distance_to_goal"] == pytest.approx(5.0)
        assert info["action"] == (0.0, 0.0)
        bodies = [body for _, body in unity.commands]
        assert [body["command"] for body in bodies][:2] == ["reset", "set_control_mode"]
        assert bodies[1]["robot"] == "robot_2"
    finally:
        environment.close()


# -- registration and rendering ---------------------------------------------=


def test_the_environment_registers_with_gymnasium():
    assert register_envs() == ENV_ID
    import gymnasium

    assert ENV_ID in gymnasium.registry


def test_render_does_nothing_without_the_human_mode(env):
    assert env.render_mode is None
    assert env.render() is None


def test_the_environment_leaves_a_client_it_was_given_running(client, unity, bridge):
    environment = RobotSNAPEnv(client=client, wait_timeout=_WAIT_TIMEOUT)
    environment.close()

    assert client.is_connected is True


def test_the_environment_stops_a_client_it_started():
    environment = RobotSNAPEnv(port=0, host="127.0.0.1", wait_timeout=0.1)
    assert environment.client.is_connected is False
    environment.close()

    assert environment.client.bridge.is_running is False


# -- pacing in simulated time -----------------------------------------------=


def test_a_scaled_session_paces_simulated_time_not_wall_time(env, unity, bridge):
    """Time.timeScale = 5: one step covers the same simulated slice, faster.

    The periods are averaged over a few steps on purpose: one step at scale 5
    holds for a tenth of the default period, so a single scheduler stall on a
    busy machine would read as a failure of the environment rather than of the
    clock the test is measuring.
    """
    steps = 4
    _start(unity, bridge, time_scale=5.0)
    env.reset()

    scaled, info = _average_step(env, steps)

    # Back to real time in the same session: the same step is the plain period.
    _publish_world(unity, time_scale=1.0)
    env.world()
    unscaled, _ = _average_step(env, steps)

    expected = _CONTROL_PERIOD / 5.0
    assert expected * 0.5 <= scaled <= expected * 3.0
    assert scaled < unscaled * 0.5
    assert info["fresh"] is False


def _average_step(environment, count):
    """Wall seconds one step of ``environment`` takes, averaged over ``count``.

    Returns the average and the last step's ``info``.
    """
    started = time.monotonic()
    for _ in range(count):
        _, _, _, _, info = environment.step([0.0, 0.0])
    return (time.monotonic() - started) / count, info


def test_a_step_still_holds_one_control_period_at_scale_one(env, unity, bridge):
    _start(unity, bridge)
    env.reset()

    started = time.monotonic()
    _, _, _, _, info = env.step([0.0, 0.0])
    elapsed = time.monotonic() - started

    assert _CONTROL_PERIOD * 0.9 <= elapsed <= _CONTROL_PERIOD * 4.0
    assert info["fresh"] is False


def test_an_absurd_time_scale_does_not_stall_a_step(env, unity, bridge):
    """A zero scale falls back to 1.0 instead of dividing the period by zero."""
    _start(unity, bridge, time_scale=0.0)
    env.reset()

    started = time.monotonic()
    _, _, _, _, info = env.step([0.0, 0.0])
    elapsed = time.monotonic() - started

    assert _CONTROL_PERIOD * 0.9 <= elapsed < _CONTROL_PERIOD * 4.0
    assert info["fresh"] is False


def test_a_non_positive_time_scale_is_refused(client):
    with pytest.raises(ValueError, match="time_scale"):
        RobotSNAPEnv(client=client, time_scale=0.0, wait_timeout=_WAIT_TIMEOUT)


def test_the_session_scale_wins_over_the_one_that_was_asked_for(env, unity, bridge):
    """A session that holds a lower scale than the caller asked for is the one that counts.

    A lockstep period runs at the largest scale whose frame still fits inside
    it, which is not always the scale that was requested - a fifth of a second
    cannot hold a hundred times speed on a twenty millisecond physics step - and
    the simulator answers with the one it can keep. Pacing on the requested
    number instead would shorten every wait by the ratio between the two and
    leave the control period a fiction.
    """
    _start(unity, bridge, time_scale=1.0)
    env.time_scale = 100.0
    env.world()

    assert env._effective_time_scale() == 1.0


def test_the_pacing_of_a_step_pays_for_its_own_latency():
    """A step is shortened by the cost of waking up, and never by more than a quarter."""
    period = 0.02  # what x10 makes of a 0.2 s control period

    assert RobotSNAPEnv._compensated_period(period, 0.0) == pytest.approx(period)
    assert RobotSNAPEnv._compensated_period(period, -1.0) == pytest.approx(period)
    # The measured cost of a step at x10: 21.3 ms for a 20 ms period.
    assert RobotSNAPEnv._compensated_period(period, 0.0013) == pytest.approx(0.0187)
    # A session the machine cannot follow reports a huge cost; the clamp holds.
    assert RobotSNAPEnv._compensated_period(period, 1.0) == pytest.approx(period * 0.75)


def test_the_pacing_learns_the_latency_of_a_step(env, unity, bridge):
    """The measured cost of the wait is what the next wait is shortened by."""
    _start(unity, bridge)
    env.reset()
    assert env._pacing_latency == 0.0

    for _ in range(12):
        env.step([0.0, 0.0])

    # A fake peer that never publishes anything new makes a step wait out its
    # grace, so the estimate settles around a period - the wait is bounded by
    # that grace, so it cannot run away - and what a step pays is the clamp.
    assert 0.0 < env._pacing_latency <= _CONTROL_PERIOD * 1.5
    assert RobotSNAPEnv._compensated_period(
        _CONTROL_PERIOD, env._pacing_latency
    ) == pytest.approx(_CONTROL_PERIOD * 0.75)


# -- lockstep: the world is held between two steps --------------------------=


def _lockstep_env(client, **overrides):
    """An environment asking for lockstep, with the options the tests want."""
    options = dict(
        client=client,
        scenario="demo",
        control_period=_LOCKSTEP_PERIOD,
        wait_timeout=_WAIT_TIMEOUT,
        pacing="lockstep",
    )
    options.update(overrides)
    return RobotSNAPEnv(**options)


def _start_lockstep(
    unity,
    bridge,
    *,
    period,
    sim_time=10.0,
    held=True,
    clock=None,
    held_reports=(False, True),
    spend=0.15,
):
    """Bring a fake lockstep session up and return the gate that paces it.

    ``sim_time`` is the clock of the snapshot published before the first
    release - the world as the episode begins - and ``held`` what that snapshot
    says about the world being stopped: true, because the gate holds the world
    from the pacing command until the first release, or ``None`` for a session
    that publishes no such flag. ``clock`` is the gate's own clock where a test
    wants it ahead of that snapshot, which is what a free-running setup that has
    stopped speaking leaves behind. ``spend`` is the wall window the gate keeps
    the world moving after it has advanced its clock, and the steps of the
    budget tests - a hundred of them - pass zero to keep their cost down.
    """
    _wait_for(
        lambda: bridge.topic_types().get("simulation/control_result")
        == "std_msgs/msg/String"
    )
    _no_delay(unity, bridge)
    unity.start_responder(sim_time_seconds=sim_time)
    _publish_world(unity, sim_time=sim_time, held=held)
    # The bridge decodes on its own thread: wait for the snapshot so the session the environment starts
    # against is the one this call published, and not an empty bridge.
    _wait_for(lambda: bridge.topic_counts().get("simulation/state", 0) > 0)
    return _LockstepGate(
        unity,
        period=period,
        sim_time=sim_time if clock is None else clock,
        publish=lambda sim, held: _publish_world(unity, sim_time=sim, held=held),
        held_reports=held_reports,
        spend=spend,
    ).start()


def _no_delay(peer, bridge):
    """Turn Nagle off on both ends of the fake session's socket.

    A lockstep step is a handful of small frames in each direction - the
    release, the snapshots that report the period spent and the hold, the
    velocity command - and Nagle holds each small write until the one before it
    is acknowledged, which on this loopback costs forty milliseconds per
    direction of *test* time. Unity's own socket is not what these tests are
    about, and the ones that drive a hundred steps say so here rather than
    spending eight seconds modelling a batching network nobody asked for.

    The peer's socket is public; the bridge's belongs to the server and is
    reached the way a test may reach it, guarded, since a bridge between two
    sessions would only make the steps slower and not wrong.
    """
    peer.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    session = getattr(bridge, "_session", None)
    sock = getattr(session, "_sock", None)
    if sock is not None:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)


def _pacing_modes(unity, since=0):
    """The ``set_pacing`` modes the session was sent, in order."""
    return [
        body.get("mode")
        for _, body in unity.commands[since:]
        if isinstance(body, dict) and body.get("command") == "set_pacing"
    ]


def test_a_lockstep_step_waits_for_the_world_to_be_held_before_the_next_release(
    client, unity, bridge
):
    """The next release must not go out while the period is still being spent.

    The snapshot that carries the clock a period spent is published by the very
    physics loop that spent it, so the world can read as at its target while the
    gate has not stopped it yet. A client that trusted that snapshot would
    release again there, the gate would re-arm, and the world would run on
    between two steps - which is the drift this test refuses: every step has to
    cover one period, and the gate has to see every release land on a stopped
    world.
    """
    environment = _lockstep_env(client)
    gate = _start_lockstep(unity, bridge, period=_LOCKSTEP_PERIOD)
    try:
        environment.reset()
        seconds = []
        for _ in range(3):
            _, _, _, _, info = environment.step([0.0, 0.0])
            seconds.append(info["episode_seconds"])

        assert seconds == pytest.approx(
            [_LOCKSTEP_PERIOD * step for step in (1, 2, 3)]
        )
        assert gate.early == 0
    finally:
        gate.stop()
        environment.close()


def test_a_lockstep_session_without_the_held_key_keeps_its_pacing(client, unity, bridge):
    """An older build says nothing about holding, so the clock alone paces the step."""
    environment = _lockstep_env(client)
    gate = _start_lockstep(
        unity, bridge, period=_LOCKSTEP_PERIOD, held=None, held_reports=(None,), spend=0.0
    )
    try:
        environment.reset()
        started = time.monotonic()
        _, _, _, _, info = environment.step([0.0, 0.0])
        elapsed = time.monotonic() - started

        assert info["episode_seconds"] == pytest.approx(_LOCKSTEP_PERIOD)
        # The clock decides, as it always did: no grace waited for a flag this session cannot send.
        assert elapsed < _lockstep_grace(_LOCKSTEP_PERIOD) * 0.75
    finally:
        gate.stop()
        environment.close()


def test_a_lockstep_step_returns_when_the_world_never_reports_itself_held(
    client, unity, bridge
):
    """The grace bounds the wait: a peer that never stops the world must not hang the loop."""
    environment = _lockstep_env(client)
    gate = _start_lockstep(
        unity, bridge, period=_LOCKSTEP_PERIOD, held=False, held_reports=(False,), spend=0.0
    )
    try:
        environment.reset()
        started = time.monotonic()
        _, _, _, _, info = environment.step([0.0, 0.0])
        elapsed = time.monotonic() - started

        assert elapsed >= _lockstep_grace(_LOCKSTEP_PERIOD) * 0.9
        assert elapsed < _WAIT_TIMEOUT
        assert info["episode_seconds"] == pytest.approx(_LOCKSTEP_PERIOD)
    finally:
        gate.stop()
        environment.close()


def test_the_episode_clock_never_overshoots_its_budget_by_more_than_a_period(
    client, unity, bridge
):
    """An episode of N seconds ends within N plus one control period.

    The budget is counted in the world's own seconds, so the only overshoot a
    step can add is the period it just spent - and only when that period crossed
    the budget. A step that covered more than its period, which is what a
    release-early drift makes of it, would overshoot by more, so this test also
    pins the step count to the periods the budget is worth.
    """
    budget = 0.5
    environment = _lockstep_env(client, max_episode_seconds=budget)
    gate = _start_lockstep(unity, bridge, period=_LOCKSTEP_PERIOD)
    try:
        environment.reset()
        seconds = []
        steps = 0
        terminated = truncated = False
        while not (terminated or truncated) and steps < 50:
            _, _, terminated, truncated, info = environment.step([0.0, 0.0])
            seconds.append(info["episode_seconds"])
            steps += 1

        assert truncated is True
        assert max(seconds) <= budget + _LOCKSTEP_PERIOD
        assert steps <= round(budget / _LOCKSTEP_PERIOD) + 1
        assert gate.early == 0
    finally:
        gate.stop()
        environment.close()


def test_the_world_is_held_when_the_episode_ends_and_close_leaves_it_held(
    client, unity, bridge
):
    """The mission's world stops with it, and closing the environment does not restart it."""
    environment = _lockstep_env(client, max_episode_seconds=2.5 * _LOCKSTEP_PERIOD)
    gate = _start_lockstep(unity, bridge, period=_LOCKSTEP_PERIOD)
    try:
        environment.reset()
        for _ in range(6):
            _, _, terminated, truncated, _ = environment.step([0.0, 0.0])
            if terminated or truncated:
                break
        assert truncated is True

        # The step that ended the episode asked the session to hold the world again ...
        ended_at = len(unity.commands)
        assert _pacing_modes(unity)[-1] == "lockstep"

        environment.close()

        # ... and close() left that hold alone instead of handing the clock back.
        assert _pacing_modes(unity, since=ended_at) == []
        assert _pacing_modes(unity)[-1] == "lockstep"
    finally:
        gate.stop()
        environment.close()


def test_the_episode_after_a_held_one_starts_again(client, unity, bridge):
    """reset() knows how to start from a world the mission before it left stopped."""
    environment = _lockstep_env(client, max_episode_seconds=2.5 * _LOCKSTEP_PERIOD)
    gate = _start_lockstep(unity, bridge, period=_LOCKSTEP_PERIOD)
    try:
        environment.reset()
        for _ in range(6):
            _, _, terminated, truncated, _ = environment.step([0.0, 0.0])
            if terminated or truncated:
                break
        assert truncated is True

        _, info = environment.reset()
        assert info["steps"] == 0
        assert info["episode_seconds"] == pytest.approx(0.0)

        _, _, terminated, truncated, info = environment.step([0.0, 0.0])
        assert (terminated, truncated) == (False, False)
        assert info["episode_seconds"] == pytest.approx(_LOCKSTEP_PERIOD)
    finally:
        gate.stop()
        environment.close()


# -- the budget of an episode -------------------------------------------------


def test_a_lockstep_episode_counts_its_budget_from_the_held_world(client, unity, bridge):
    """The budget starts at the clock the session reports once the world is held, first period included.

    The setup leaves the world free-running while the scenario is rebuilt, so the snapshot in flight when the
    episode begins can be one the world has already left behind: here the world ran on to 10.9 before the
    gate stopped it, while the last snapshot it published says 10.0. The gate stops the world before it
    answers the pacing command and the clock does not move while it lasts, so the snapshot that reports the
    hold carries the starting time; an epoch taken from the older one would hand the policy the world time
    the mission never lived, and an epoch taken after the first step would leave that first period out
    entirely.
    """
    held_clock = 10.9
    environment = _lockstep_env(client, max_episode_seconds=10.0 * _LOCKSTEP_PERIOD)
    gate = _start_lockstep(
        unity,
        bridge,
        period=_LOCKSTEP_PERIOD,
        sim_time=10.0,
        held=False,
        clock=held_clock,
        spend=0.0,
    )

    def say_the_world_is_held():
        # The hold lands with the pacing command; the snapshot that reports it follows a moment later.
        _wait_for(
            lambda: any(
                isinstance(body, dict) and body.get("mode") == "lockstep"
                for _, body in list(unity.commands)
            ),
            timeout=_WAIT_TIMEOUT,
        )
        time.sleep(0.2)
        _publish_world(unity, sim_time=held_clock, held=True)

    thread = threading.Thread(target=say_the_world_is_held, daemon=True)
    thread.start()
    try:
        environment.reset()
        thread.join(timeout=_WAIT_TIMEOUT)

        assert environment._episode_clock_start == pytest.approx(held_clock)

        _, _, terminated, truncated, info = environment.step([0.0, 0.0])
        assert (terminated, truncated) == (False, False)
        assert info["episode_seconds"] == pytest.approx(_LOCKSTEP_PERIOD)
    finally:
        gate.stop()
        environment.close()


def test_a_budget_on_a_multiple_of_the_period_ends_exactly_on_it(client, unity, bridge):
    """Twenty seconds of budget at a fifth of a second a step is a hundred steps, not a hundred and one.

    The world's clock is a ``float`` the simulator advances one physics step at a time, so a hundred periods
    of 0.2 s add up to 19.99999999999993 and not to 20.0. An episode that compared its budget strictly spent
    a whole extra period to cross a line the world was already standing on - a fifth of a second of world
    time, on every episode of a training run - which is the live measurement of 20.200 s over 101 steps.
    """
    environment = _lockstep_env(
        client, control_period=_BUDGET_PERIOD, max_episode_seconds=_BUDGET
    )
    gate = _start_lockstep(unity, bridge, period=_BUDGET_PERIOD, spend=0.0)
    try:
        environment.reset()
        steps = 0
        terminated = truncated = False
        while steps < 2 * round(_BUDGET / _BUDGET_PERIOD):
            _, _, terminated, truncated, info = environment.step([0.0, 0.0])
            steps += 1
            if terminated or truncated:
                break

        assert truncated is True
        assert steps == round(_BUDGET / _BUDGET_PERIOD)
        assert info["episode_seconds"] == pytest.approx(_BUDGET)
        assert gate.early == 0
    finally:
        gate.stop()
        environment.close()


def test_a_budget_off_the_period_grid_stops_within_one_period_of_it(client, unity, bridge):
    """A budget that is not a whole number of periods is reached by the first step past it, and no later.

    One control period is the smallest slice of world time an episode can end on, so the overshoot of a
    budget the steps straddle is that period and never more: 20.05 s is crossed by the hundred-and-first step
    at 20.2 s, which is 0.15 s past the budget.
    """
    budget = _BUDGET + 0.05
    environment = _lockstep_env(
        client, control_period=_BUDGET_PERIOD, max_episode_seconds=budget
    )
    gate = _start_lockstep(unity, bridge, period=_BUDGET_PERIOD, spend=0.0)
    try:
        environment.reset()
        steps = 0
        terminated = truncated = False
        while steps < 2 * round(budget / _BUDGET_PERIOD) + 2:
            _, _, terminated, truncated, info = environment.step([0.0, 0.0])
            steps += 1
            if terminated or truncated:
                break

        assert truncated is True
        assert budget <= info["episode_seconds"] <= budget + _BUDGET_PERIOD
        assert steps == round(budget / _BUDGET_PERIOD) + 1
        assert gate.early == 0
    finally:
        gate.stop()
        environment.close()


def test_a_session_without_a_clock_keeps_the_wall_fallback(env, unity, bridge):
    """A world that publishes no clock is budgeted in wall seconds times its scale, as it always was.

    The fallback exists for a session whose world clock cannot be read at all: pacing on simulated time is
    meaningless there, so the episode counts the wall time its steps took, scaled by the speed the session
    reports. The epoch of the budget must not take that case away.
    """
    _start(unity, bridge)
    # The state stream is there; the clock inside it is not.
    unity.publish_state(_state(sim_time=None))

    env.reset()

    started = time.monotonic()
    for _ in range(5):
        _, _, terminated, truncated, info = env.step([0.0, 0.0])
        assert (terminated, truncated) == (False, False)
    wall = time.monotonic() - started

    # Each step counts the wall time of the one before it, so five steps are the four waits they cost.
    assert 0.0 < info["episode_seconds"] <= wall
    assert info["episode_seconds"] >= wall * 0.5
