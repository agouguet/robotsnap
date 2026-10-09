"""Getting a RobotSNAP session ready and running one episode's clocks.

This is the simulator's own bookkeeping: waiting for Unity to connect and to
publish its first snapshot, writing and applying the scenario, waiting for the
world to report itself rebuilt after a reset, taking the robot in hand at the
start of an episode, and counting the episode in the simulator's seconds rather
than in wall seconds times a scale.

:class:`RobotSNAPEnvError` is defined here, next to the setup that raises it, and
re-exported by :mod:`robotsnap.envs.base`; the other mixins import it from this
module so no mixin has to import the environment back.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from robotsnap import scenario as scenarios
from robotsnap import topics
from robotsnap.envs.pacing import FREE, LOCKSTEP, _POLL
from robotsnap.envs.world import World

__all__ = ["RobotSNAPEnvError", "SessionMixin"]

#: How long the setup waits for a connected session's first snapshot before deciding it is not merely slow
#: but asleep, and waking it. A running session answers well inside this; a stopped one answers never.
_WAKE_GRACE = 1.0

#: How long the environment waits, at the start of a lockstep episode, for the session to report the world
#: held before it reads the clock the episode's budget counts from. The hold is applied before the pacing
#: command is acknowledged, so what is waited for is the snapshot that reports it: one frame of the
#: simulator, or its own heartbeat while it stands still, both well inside this. A session that stops
#: reporting - or a build whose flag never turns true - costs this wait once per episode and no step,
#: because the epoch then falls back to the clock at hand.
_LOCKSTEP_HOLD_GRACE = 0.5

#: How long the environment waits for a session to acknowledge the hold it writes at the end of an episode.
#: The episode is already over and the caller is owed its return value, so this is deliberately short: a
#: session that answered the step a moment ago answers this in one round trip, and one that has gone away
#: costs a bounded wait instead of the client's whole command timeout. The hold is a courtesy to the world
#: the mission is leaving, not the pacing the episode needs, so it is not raised when it fails.
_END_OF_EPISODE_TIMEOUT = 1.0


class RobotSNAPEnvError(RuntimeError):
    """Raised when the session cannot give the world an episode needs."""


class SessionMixin:
    """Session setup and episode clocks of :class:`~robotsnap.envs.base.RobotSNAPEnv`."""

    @property
    def wrote_scenario(self) -> bool:
        """Whether this session wrote the scenario file, rather than finding it.

        A run removes the scenario it wrote on its way out, and nothing else: an
        id that named a scenario the project already ships names a file the
        project owns, and a run that only played it must leave it where it is.
        """
        return self._scenario_written

    def _prepare_session(self, *, seed: int | None, options: Mapping[str, Any]) -> None:
        """Make the session ready and the wanted scenario the current one."""
        if not self.client.wait_until_ready(timeout=self.wait_timeout):
            # A socket has a port and a graph has a domain; the client says which one it is rather than
            # the message naming a port that means nothing over ROS2.
            where = getattr(self.client.bridge, "session_label", None) or (
                f"port {self.client.bridge.port}"
            )
            raise RobotSNAPEnvError(
                f"Unity never connected on {where}: {self.client.last_error}"
            )

        # A connection is not a world: the connector registers its publishers and starts sending a moment
        # later, and everything below - reading the scenario, waiting for the reset handshake, placing a
        # scan at its pose - is read from that stream. A session left stopped - the red button of the
        # application, or the ``--stop`` of the previous run - is even quieter: it has nothing to say until
        # it runs again. So: give the streams a moment, wake the session if it is asleep, and only then wait
        # for the first snapshot, which is what the rest of the setup is written against.
        if not self._wait_until(
            lambda: self._count(topics.SIMULATION_STATE) > 0, _WAKE_GRACE
        ):
            state = self.client.snapshot()
            if not isinstance(state, Mapping) or (
                state.get("paused") or state.get("stopped") or state.get("playing") is False
            ):
                self.client.play()
            if self.pacing == LOCKSTEP:
                # A lockstep session that lost its client is holding the world at a zero time scale, and a
                # world at a zero scale publishes nothing: the stream this wait is reading is the one the stop
                # silenced, so waiting longer cannot succeed. The clock is asked back before the real wait
                # below, and _begin_episode lends it out again. This costs one command on a session that was
                # merely slow, and it is what keeps a run from being refused for thirty seconds by a scene a
                # previous run left stopped.
                self.client.set_pacing(FREE)
        if not self._wait_until(
            lambda: self._count(topics.SIMULATION_STATE) > 0, self.wait_timeout
        ):
            raise RobotSNAPEnvError(
                "the session connected but published no state snapshot "
                f"(waited {self.wait_timeout:g}s)"
            )

        # A lockstep session is stopped between two releases, and a stopped world cannot rebuild itself: the
        # scenario's reset runs as a coroutine that waits on simulated time, so it never finishes while the
        # gate is holding the world at zero. That is what made an episode work and the next one time out
        # waiting for a world that could not come back. Giving the world its clock back for the length of the
        # setup is the fix; the episode takes it away again in _begin_episode.
        if self.pacing == LOCKSTEP:
            self.client.set_pacing(FREE)

        scenario = options.get("scenario", self.scenario)
        if scenario is not None and self.scenario_fields and not self._scenario_written:
            # A scenario of the project is played as it stands. Writing the run's
            # fields over it would replace a file the user owns, and the run
            # would then remove it on the way out because it named it.
            if scenarios.find(scenario, self.scenario_directory) is None:
                # Written once: Unity caches a scenario under the name it
                # loaded, so writing it again on every reset would change
                # nothing it loads.
                path: Path | None = self.client.create_scenario(
                    scenario,
                    launch=False,
                    directory=self.scenario_directory,
                    **self.scenario_fields,
                )
                if path is None:
                    raise RobotSNAPEnvError(
                        f"cannot write scenario {scenario!r}: {self.client.last_error}"
                    )
                self._scenario_written = True

        state = self.client.snapshot() or {}
        applied = bool(state.get("scenario_applied"))
        current = state.get("scenario_id")
        launch = options.get("launch")
        if launch is None:
            launch = self.reload_on_reset or not applied

        if scenario is not None and (launch or current != scenario):
            if not self.client.launch_scenario(
                scenario, seed=seed, timeout=self.wait_timeout
            ):
                raise RobotSNAPEnvError(
                    f"scenario {scenario!r} was not applied: {self.client.last_error}"
                )
            return

        if not applied:
            raise RobotSNAPEnvError(
                "no scenario is applied: pass scenario=..., or load one on the session"
            )

        before_done = self._count(topics.RESET_DONE)
        before_state = self._count(topics.SIMULATION_STATE)
        before_sim = self._sim_seconds()
        result = self.client.reset(seed=seed)
        if result is None or result.get("ok") is False:
            message = self.client.last_error
            if isinstance(result, dict) and result.get("message"):
                message = str(result["message"])
            raise RobotSNAPEnvError(f"the scenario was not reset: {message}")
        self._wait_for_world(before_done, before_state, before_sim)

    def _wait_for_world(
        self, before_done: int, before_state: int, before_sim: float | None = None
    ) -> None:
        """Wait for the world to report itself rebuilt after a reset.

        ``/reset_done`` is the simulator's own handshake and the signal this
        waits for. The second test is a fallback for a build that does not
        publish it: the state has to have been seen un-applied first, then come
        back applied on a newer snapshot, which is what a rebuild looks like
        from outside. A third signal covers a build that both omits the
        handshake and rebuilds faster than its own state stream can be sampled -
        the window in which the world reads as un-applied lasts a frame, and a
        snapshot every tenth of a *simulated* second will miss it: the clock the
        reset put back to zero is then the proof, because nothing else rewinds
        the simulator's clock.
        """
        seen_unapplied = False

        def ready() -> bool:
            nonlocal seen_unapplied
            if self._count(topics.RESET_DONE) > before_done:
                return True
            if self._count(topics.SIMULATION_STATE) <= before_state:
                return False
            if not self._scenario_applied():
                seen_unapplied = True
                return False
            if seen_unapplied:
                return True
            now = self._sim_seconds()
            return before_sim is not None and now is not None and now < before_sim

        if not self._wait_until(ready, self.wait_timeout):
            raise RobotSNAPEnvError(
                "the world never reported itself ready after the reset "
                f"(waited {self.wait_timeout:g}s)"
            )

    def _begin_episode(self) -> None:
        """Take the robot in hand and start this environment's own clocks."""
        # A session can be sitting paused - the scene starts that way, and a client that ran the stop button on
        # the previous run left it that way - and an episode on a paused world is an episode that never moves:
        # the clock the environment counts in is frozen, every step is a step of nothing, and the stall guard
        # ends it. Asking the session to run costs one round trip and is what the pacing below assumes.
        state = self.client.snapshot()
        if isinstance(state, Mapping) and (state.get("paused") or state.get("playing") is False):
            self.client.play()

        self.client.set_control_mode(self.control_mode, robot=self.robot)
        if self.pacing == LOCKSTEP:
            # Asked before the scale, and before the first command of the episode. Before the scale, because
            # the scale a lockstep period can hold is bounded by the physics step of the engine - asking for
            # a hundred times speed while the world is still free-running would let it run away for however
            # long the next round trip takes - and the pacing command is the one that clamps it. Before the
            # first command, because the world stops here and only the releases in step() move it, so no
            # reset window is spent running away from the client.
            applied = self.client.set_pacing(LOCKSTEP, step_seconds=self.control_period)
            if not applied or not applied.get("ok", False):
                raise RobotSNAPEnvError(
                    "the session refused lockstep pacing, so a step could not be held to one control "
                    f"period: {applied or self.client.last_error}"
                )
            self._note_lockstep_without_clock()
        if (
            self.time_scale is not None
            or self.fixed_timestep is not None
            or self.maximum_delta_time is not None
        ):
            # The scale is only sent when the caller gave one; a caller that only changed the pacing keeps
            # whatever the session was already running at.
            self.client.set_time_scale(
                self.time_scale
                if self.time_scale is not None
                else self._effective_time_scale(),
                fixed_timestep=self.fixed_timestep,
                maximum_delta_time=self.maximum_delta_time,
            )
        self.client.send_cmd_vel(0.0, 0.0, robot=self.robot)
        self.client.mark_scenario_start()

        # The budget counts the world from the moment this episode begins, so its epoch is the clock the
        # world reports here, before any step has spent a period - not the one it reports after the first
        # step, which would leave that first period out of the budget and cost the episode a whole step more
        # than it pays for. Which moment that is depends on the pacing. The command above has stopped a
        # lockstep world - the gate writes its zero time scale before it acknowledges - so the clock is
        # frozen, and the snapshot reporting the hold carries the starting time exactly: waiting for that
        # report is what keeps the epoch from being the last *free-running* value the setup left in flight.
        # Free running has no such moment - the world is moving, and the setup that rebuilt the scenario has
        # just rewound its clock - so the epoch is left to the first world read of the episode, which
        # reset() takes before it steps. A session that publishes no clock keeps the wall fallback.
        start: float | None = None
        if self.pacing == LOCKSTEP:
            start = self._sim_seconds()
            if start is not None and self.client.held is not None:
                if self._wait_until(lambda: self.client.held is True, _LOCKSTEP_HOLD_GRACE):
                    held_clock = self._sim_seconds()
                    if held_clock is not None:
                        start = held_clock

        # The episode holds the world only for its own length: the release below is what starts it, and the
        # hold _end_episode writes is a property of the mission that just ended, not of this one.
        self._world_held = False
        self._elapsed_seconds = 0.0
        self._wall_seconds = 0.0
        self._episode_clock_start = start
        self._last_sim_time = start
        self._last_sim_change = None
        self._last_wall_delta = 0.0
        self._steps = 0
        self._last_command = (0.0, 0.0)
        self._streams = {
            "world": self._count(topics.SIMULATION_STATE) > 0,
            "pose": self._count(self.odom_topic) > 0,
        }

    def _advance_episode_clock(self, world: World) -> None:
        """Count this episode in the *simulator's* seconds, not in wall seconds times the scale.

        ``sim_time_seconds`` is the clock the simulator stamps every one of its streams with, and it advances
        with the session's time scale whatever the machine manages to deliver. Summing wall time multiplied
        by the *requested* scale instead - which is what this used to do - drifts away from the world the
        moment the machine cannot follow the ask: a session asked for a hundred times speed that reaches
        forty would have its episode end after sixty seconds of a world that has lived a hundred and fifty,
        and a single stalled step would add a lump of time that never happened. The wall is kept only as the
        fallback for a scene that carries no clock at all, since then there is no simulator clock to trust.

        The epoch this counts from is the world at the start of the episode - see :meth:`_begin_episode` -
        and it is only ever taken once: a clock read is not allowed to move it later, which is what would
        leave the first period out of the budget. The simulator's clock restarts when a scenario is
        reloaded, so a value that went backwards re-bases the episode rather than reporting a negative one.
        """
        now = getattr(world, "sim_time_seconds", None)
        if now is None or not math.isfinite(float(now)):
            self._elapsed_seconds += self._last_wall_delta * float(
                getattr(world, "time_scale", None) or 1.0
            )
            return

        now = float(now)
        if now != self._last_sim_time:
            self._last_sim_change = time.monotonic()
        self._last_sim_time = now
        if self._episode_clock_start is None or now < self._episode_clock_start:
            self._episode_clock_start = now
        self._elapsed_seconds = now - self._episode_clock_start

    def _fresh(self, before: Mapping[str, int]) -> bool:
        """Whether every stream this session does publish has moved on."""
        if self._streams["world"] and self._count(topics.SIMULATION_STATE) <= before["world"]:
            return False
        if self._streams["pose"] and self._count(self.odom_topic) <= before["pose"]:
            return False
        return True

    def _count(self, topic: str) -> int:
        """How many messages of ``topic`` the bridge has decoded so far."""
        return int(self.client.bridge.topic_counts().get(topics.base(topic), 0))

    def _scenario_applied(self) -> bool:
        state = self.client.snapshot()
        return bool(state.get("scenario_applied")) if isinstance(state, dict) else False

    @staticmethod
    def _wait_until(predicate, timeout: float) -> bool:
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            if predicate():
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(_POLL)

    def _end_episode(self) -> None:
        """Stop the world with the episode, so it does not live on while Python prints or tidies up.

        A lockstep gate stops the world itself once the period is spent, so the
        step that ended the episode has normally left it held already. This
        re-asserts the hold once per episode anyway, which is what covers the
        step whose wait ran out its grace with the world still moving - a
        session that stopped publishing - and what makes the guarantee hold on
        a build whose snapshot carries no ``held`` key to check it against.

        It is also the state :meth:`close` now leaves alone: the environment
        took the clock away for the mission, and the mission is over. A session
        that refuses the command is noted through the client's ``last_error``
        and not raised - the outcome of the step is what the caller is told, not
        the outcome of this - and the next :meth:`reset` starts from a held
        world either way, which ``_prepare_session`` and ``_begin_episode``
        already handle.
        """
        if self.pacing != LOCKSTEP:
            return
        result = self.client.set_pacing(
            LOCKSTEP,
            step_seconds=self.control_period,
            timeout=_END_OF_EPISODE_TIMEOUT,
        )
        self._world_held = result is not None and result.get("ok", True) is not False
