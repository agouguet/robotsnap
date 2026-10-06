"""How a RobotSNAP step is held to a control period of *simulated* time.

Unity has no "advance exactly N fixed steps" primitive, so a step here holds
until the world has moved for ``control_period`` seconds of its own clock. On a
session running at a time scale that is ``control_period / time_scale`` seconds of
wall time, which is what makes a faster session shorten the wall length of an
episode without changing how much of the world each step covers.

Two modes: ``FREE`` is the historical one, where the world keeps moving while the
policy thinks; ``LOCKSTEP`` asks the simulator to spend exactly one control period
of its own seconds per step and to stop in between, so the period is a property of
the world rather than of how fast the client answers. The functions here are the
whole of that policy, kept apart from the task so an environment can be sped up
without a reward changing.
"""

from __future__ import annotations

import math
import time
import warnings
from collections.abc import Mapping
from typing import Any

from robotsnap import topics

__all__ = ["PacingMixin", "FREE", "LOCKSTEP"]

#: How long the waiting loops sleep between two polls, in seconds.
_POLL = 0.005

#: How much of a control period the pacing may take off to pay for its own
#: latency. A quarter keeps a step from ever shortening by more than that,
#: whatever a loaded machine makes the measurement say.
_MAX_PACING_CORRECTION = 0.25

#: Weight of the newest measurement in the pacing latency estimate. A fifth
#: converges in about ten steps, which is short enough to follow a session that
#: changes speed and long enough not to chase one scheduling hiccup.
_PACING_LATENCY_WEIGHT = 0.2

#: How far under its target a lockstep world may land and still count as having
#: spent the period, in simulated seconds.
#:
#: The simulator spends a period in whole physics steps and a physics step is a
#: ``float``: ten of the twenty-millisecond steps a fifth of a second asks for
#: add up to 0.19999999 and not to 0.2, so a strict ``>=`` against the period is
#: a boundary the world lands on rather than crosses. The client then waits out
#: its whole grace - four periods - for a period that was already spent, and the
#: world reads as running at a quarter speed. The tolerance is a thousandth of a
#: second: far above the engine's own rounding, which is microseconds, and far
#: below the interval the state stream is published at, which is what a snapshot
#: arriving mid-period would be recognised by.
_LOCKSTEP_ROUNDING = 1e-3

#: The two pacing modes. ``FREE`` is the historical one: the session runs at its time scale and a step waits
#: a control period of wall time divided by that scale. ``LOCKSTEP`` asks the simulator to spend exactly one
#: control period of *its own* seconds per step and stop in between, so the period is a property of the world
#: and not of how fast the client answers.
FREE = "free"
LOCKSTEP = "lockstep"
_PACING_MODES = (FREE, LOCKSTEP)

#: How long the environment keeps asking a lockstep world for its period before giving up on it, as a
#: multiple of the period's nominal wall cost and as a floor. This is the bound on the wait itself and it
#: does not depend on the world reporting itself held: a world at a hundred times speed that reaches half
#: of the ask still answers inside the floor, and a peer that stopped publishing - or stopped stopping -
#: returns a step rather than hanging the training loop. See :func:`_lockstep_grace`.
_LOCKSTEP_GRACE_FACTOR = 4.0
_LOCKSTEP_GRACE_FLOOR = 0.5


def _positive_scale(value: Any) -> float | None:
    """``value`` as a finite, strictly positive float, or ``None``.

    Every time scale an environment resolves passes through here, so a missing,
    non-numeric, non-finite or non-positive one is dropped rather than divided
    by: a zero or a ``nan`` must never turn the control period into a division
    by zero or a negative wait.
    """
    try:
        scale = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(scale) or scale <= 0.0:
        return None
    return scale


def _lockstep_grace(period: float) -> float:
    """Wall seconds a lockstep step may wait for the period it released.

    ``period`` is the nominal wall cost of one control period at the session's
    current scale, so the bound is a multiple of it - four periods - with a
    floor for the high scales, where the periods themselves are shorter than a
    scheduler tick. The floor keeps a session running at a hundred times speed
    from being declared silent while a fraction of its period is still in
    flight; the multiple keeps a session that answers late, but answers, from
    being cut off. Either way the bound is finite: a peer that stopped
    publishing, or that never reports itself held, costs one bounded wait and
    then a step returns. See :meth:`RobotSNAPEnv._wait_control_period`.
    """
    return max(_LOCKSTEP_GRACE_FACTOR * float(period), _LOCKSTEP_GRACE_FLOOR)


class PacingMixin:
    """Simulated-time pacing of :class:`~robotsnap.envs.base.RobotSNAPEnv`."""

    def _note_lockstep_without_clock(self) -> None:
        """Say so, once, when a lockstep episode has no clock to verify its steps against.

        The simulator holds a step to the control period by counting its own
        physics steps, so a session that publishes no clock is still paced. What
        the environment loses is its own half of the check: the world time a step
        waits for is read from ``sim_time_seconds``, and a snapshot without one
        leaves ``step`` to fall back on wall time - which is the very mode the
        caller asked to leave. That is worth a warning rather than silence,
        because the run then behaves like free running while looking like it
        worked.
        """
        if self._warned_no_clock:
            return
        state = self.client.snapshot()
        if not isinstance(state, Mapping):
            return
        if state.get("sim_time_seconds") is None and state.get("time_scale") is None:
            self._warned_no_clock = True
            warnings.warn(
                "lockstep was asked for, but the session publishes no simulation clock "
                "(sim_time_seconds is null): Unity still holds each step to the control "
                "period by counting physics steps, but this environment cannot verify it "
                "and paces on wall time instead. Give the scene a Clock component, or ask "
                "for --pacing free to say that is what you want.",
                RuntimeWarning,
                stacklevel=2,
            )

    def _wait_control_period(self) -> tuple[float, bool]:
        """Hold for one control period of simulated time, and report the world.

        The period is ``control_period`` seconds of simulated time, so it is
        ``control_period / time_scale`` seconds of wall time at the current
        scale.

        Waking up to check costs a fraction of a millisecond, and reading the
        streams a little more, so a step taken at its face value covers slightly
        more than one period. At x1 that is lost in the noise - a millisecond
        against two hundred - but the error is a *wall* one, so a higher scale
        multiplies it: measured at x10, a period of 20 ms came out at 21.3 ms,
        and a step covered 0.213 s of simulated time instead of 0.200. The cost
        of the last step is therefore measured and taken off the next deadline,
        bounded by :data:`_MAX_PACING_CORRECTION` so a session the machine
        cannot follow shortens a step without ever cancelling it.

        Free running, the wait ends when the deadline has passed and - for one
        period more - a state and an odometry message newer than the command
        have arrived, so the observation belongs to the world the action
        produced and not to the one it was chosen on.

        In lockstep the rule is stricter and it is the whole point of the mode:
        the wait ends when the world's own clock has reached the target **and**
        the session reports itself ``held`` - the gate has stopped the world at
        the end of the period. The clock alone is not enough. It is read from a
        snapshot published by the very physics loop the period just ran, so one
        can be seen at the target before the gate has stopped writing; a release
        that goes out then re-arms the gate's step counter
        (:meth:`~robotsnap.client.RobotSNAPClient.release_pacing`), the world
        keeps moving between two steps, and every step covers more than its
        period while the drift accumulates. A session built without the ``held``
        key does not answer the second question at all, and the clock alone then
        decides, which is the pacing those sessions always had.

        Both waits are bounded on purpose, and the lockstep bound does not
        depend on the world ever reporting itself held: a step that reaches
        neither its target nor a held world inside :func:`_lockstep_grace` -
        four periods, or half a second, whichever is longer, of wall time -
        returns anyway, with ``fresh`` as it found it. On a session that
        publishes slower than the control period, that has stopped publishing
        altogether, or that never stops the world, the answer is then
        ``fresh=False`` (or a step that covered less than its period) rather
        than a loop that hangs.
        """
        before = {
            "world": self._count(topics.SIMULATION_STATE),
            "pose": self._count(self.odom_topic),
        }
        period = self.control_period / self._effective_time_scale()
        wait = self._compensated_period(period, self._pacing_latency)
        started = time.monotonic()
        deadline = started + wait
        grace = deadline + period
        step_clock = self._lockstep_target()
        if step_clock is not None:
            # The world is stopped and only a release moves it, so the period is a count of *its* seconds and
            # no amount of overhead on this side lengthens it. The wall only says how long to keep asking.
            grace = started + _lockstep_grace(period)
        while True:
            now = time.monotonic()
            fresh = self._fresh(before)
            if step_clock is not None:
                if (self._lockstep_period_spent(step_clock) and fresh) or now >= grace:
                    return now - started, fresh
            elif now >= deadline and (fresh or now >= grace):
                elapsed = now - started
                self._pacing_latency += _PACING_LATENCY_WEIGHT * (
                    elapsed - wait - self._pacing_latency
                )
                return elapsed, fresh
            target = grace if step_clock is not None else (
                deadline if now < deadline else grace
            )
            time.sleep(min(_POLL, max(0.0, target - now)))

    def _lockstep_period_spent(self, target: float) -> bool:
        """Whether the released period is over: the clock reached ``target`` and the gate stopped the world.

        Two questions, both asked of the newest snapshot, and both have to be
        yes before another release may go out. The clock has to have reached
        ``target`` - landing on it rather than past it, see
        :data:`_LOCKSTEP_ROUNDING` - and the session has to report itself held,
        which is its own word for the gate having stopped the world at the end
        of the period. A release that goes out before that re-arms the gate and
        lets the world run on between two steps.

        A session built without the ``held`` key makes the second answer
        unknown rather than false, and the clock alone then decides - the pacing
        those sessions always had, which must not be refused over a flag they
        cannot report.
        """
        current = self._sim_seconds()
        if current is None or current < target - _LOCKSTEP_ROUNDING:
            return False
        return self.client.held is not False

    def _lockstep_target(self) -> float | None:
        """The world's clock to reach before this step is over, or ``None`` when the session free-runs.

        One control period after the world time this step started from. The gate on the other side stops the
        world at that point - rounded up to a physics step, the smallest quantum it has - so a step covers the
        control period and not the period plus however long the policy took to answer.
        """
        if self.pacing != LOCKSTEP:
            return None
        if self._last_sim_time is None:
            return None
        return self._last_sim_time + self.control_period

    def _sim_seconds(self) -> float | None:
        """The simulator's clock as the last state snapshot reports it, or ``None``."""
        state = self.client.snapshot()
        if not isinstance(state, Mapping):
            return None
        value = state.get("sim_time_seconds")
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    @staticmethod
    def _compensated_period(period: float, latency: float) -> float:
        """``period`` less the pacing latency it may pay for, never cancelled.

        The correction is only ever the part of the wait the environment does
        not control, and it is capped at :data:`_MAX_PACING_CORRECTION` of the
        period: a session whose streams lag makes the measurement large, and a
        step that stopped waiting altogether would answer an action with the
        world it was chosen on.
        """
        if latency <= 0.0:
            return period
        return max(period - min(float(latency), period * _MAX_PACING_CORRECTION), 0.0)

    def _effective_time_scale(self) -> float:
        """The strictly positive scale the current control period is paced by.

        Priority: the scale the session reported in the last world read, then
        the one the caller configured on the constructor, then the one the live
        snapshot carries. The session comes first because it is the one that
        applies the scale and it may hold less than what was asked: a lockstep
        period of a fifth of a second cannot run a hundred times speed on a
        twenty millisecond physics step, and the simulator answers with the
        scale it can keep rather than the one it was handed. A missing,
        non-numeric, non-finite or non-positive value is skipped, and 1.0 is
        the last resort, so an absurd scale can never divide the control period
        by zero or make the wait negative.
        """
        for candidate in (self._scale_from_world, self.time_scale):
            scale = _positive_scale(candidate)
            if scale is not None:
                return scale
        state = self.client.snapshot()
        if isinstance(state, Mapping):
            scale = _positive_scale(state.get("time_scale"))
            if scale is not None:
                return scale
        return 1.0
