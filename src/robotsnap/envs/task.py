"""The task of a RobotSNAP episode: what the world means to the agent.

Everything here is a *hook a subclass replaces*. The simulator's own business -
launching a scenario, pacing the world, reading the session, applying a command -
lives in the other mixins; this module answers the questions a task asks of a
world: where the goal is, whether the robot crashed or left the map, what one
step is paid, when the episode is over, and what the caller is told about it.

:class:`TaskState` is the single object a reward, a termination test and a
logging callback all read, computed once per step by :meth:`TaskMixin.task`.
:class:`~robotsnap.envs.base.RobotSNAPEnv` composes :class:`TaskMixin`, so a
custom environment is a subclass that replaces one or more of :meth:`task`,
:meth:`reward`, :meth:`terminated`, :meth:`truncated`, :meth:`stalled`,
:meth:`action_to_command` or :meth:`info`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from robotsnap.envs.world import World

__all__ = ["TaskState", "TaskMixin"]

#: How far under its budget the world's clock may sit and still count as having spent it, in seconds.
#:
#: The budget is a length of world time and the world's clock is a ``float`` the simulator advances one
#: physics step at a time, so a budget that falls exactly on the grid of control periods is reached by a
#: clock that reads a fraction of a microsecond under it: a hundred periods of a fifth of a second add up
#: to 19.99999999999993 and not to 20.0, and a strict comparison then spends a whole extra period crossing
#: a line the world is already standing on. The tolerance is a millisecond - the same order as
#: :data:`_LOCKSTEP_ROUNDING`, far above the engine's own rounding over any episode length and far below a
#: control period - so an episode can end a millisecond early but never late.
_BUDGET_ROUNDING = 1e-3


@dataclass(frozen=True)
class TaskState:
    """What the environment makes of one :class:`World`.

    A reward, a termination test and a logging callback all read the same
    quantities, so they are computed once per step, here, and handed to all
    three. A subclass that needs its own (a lane to stay in, a social distance
    to respect) overrides :meth:`RobotSNAPEnv.task` and returns its own object.
    """

    distance_to_goal: float | None
    bearing_to_goal: float | None
    goal_reached: bool
    collision: bool
    out_of_bounds: bool
    min_lidar: float | None
    nearest_agent_distance: float | None
    elapsed_seconds: float
    wall_seconds: float
    steps: int

    @property
    def done(self) -> bool:
        return self.goal_reached or self.collision or self.out_of_bounds


class TaskMixin:
    """The task hooks of :class:`~robotsnap.envs.base.RobotSNAPEnv`.

    A subclass replaces the methods it cares about - :meth:`task` for the
    quantities, :meth:`reward` for what is paid, :meth:`terminated` and
    :meth:`truncated` for when an episode ends, :meth:`action_to_command` for
    what an action means on the wire, :meth:`info` for what a logger sees - and
    leaves the simulator's side alone.
    """

    def task(self, world: World) -> TaskState:
        """Read the goal, the collision and the clock out of one world.

        A collision is the map's business when a grid is available: the cell the
        robot's centre falls in holds an obstacle. Without a grid it falls back
        to the lidar, and reports one when the closest beam is nearer than
        ``collision_distance``. Out of bounds is the robot's centre leaving the
        grid altogether.
        """
        nearest = world.nearest_agent()
        return TaskState(
            distance_to_goal=world.distance_to_goal,
            bearing_to_goal=world.bearing_to_goal,
            goal_reached=(
                world.distance_to_goal is not None
                and world.distance_to_goal <= self.goal_radius
            ),
            collision=self.is_collision(world),
            out_of_bounds=self.is_out_of_bounds(world),
            min_lidar=None if world.scan is None else world.scan.min_range(),
            nearest_agent_distance=None if nearest is None else nearest.distance,
            elapsed_seconds=self._elapsed_seconds,
            wall_seconds=self._wall_seconds,
            steps=self._steps,
        )

    def is_collision(self, world: World) -> bool:
        """Whether the robot is inside an obstacle, by the map, else by the lidar."""
        if world.pose is None:
            return False
        if world.map is not None:
            return world.map.is_occupied(
                world.pose.x, world.pose.y, self.occupancy_threshold
            )
        if world.scan is not None:
            return world.scan.min_range() < self.collision_distance
        return False

    def is_out_of_bounds(self, world: World) -> bool:
        """Whether the robot left the published grid."""
        if world.pose is None or world.map is None:
            return False
        return not world.map.contains(world.pose.x, world.pose.y)

    def reward(
        self,
        previous: TaskState,
        current: TaskState,
        action: tuple[float, float],
        world: World,
    ) -> float:
        """What the default task pays, per step.

        Progress toward the goal, minus a small price on the time spent, plus a
        bonus for arriving, minus the two failures. Progress is measured between
        two reads rather than from the start, so it cannot reward a policy for
        standing still, and the terminal terms are paid once because the episode
        ends with them.
        """
        value = 0.0
        if previous.distance_to_goal is not None and current.distance_to_goal is not None:
            value += self.progress_reward * (
                previous.distance_to_goal - current.distance_to_goal
            )
        value -= self.time_penalty * max(
            0.0, current.elapsed_seconds - previous.elapsed_seconds
        )
        if current.collision:
            value -= self.collision_penalty
        if current.out_of_bounds:
            value -= self.out_of_bounds_penalty
        if current.goal_reached:
            value += self.goal_reward
        return float(value)

    def action_to_command(self, action: Any) -> tuple[float, float]:
        """Turn an action into ``(linear_x, angular_z)`` for ``/cmd_vel``.

        The action is clipped to the limits the controller enforces anyway, so a
        policy that explores outside them sees the command it actually got, in
        ``info["action"]``.
        """
        values = np.asarray(action, dtype=np.float32).reshape(-1)
        if values.size != 2:
            raise ValueError(
                f"the default action is [linear_x, angular_z], got {values.size} value(s)"
            )
        return (
            float(np.clip(values[0], -self.max_linear, self.max_linear)),
            float(np.clip(values[1], -self.max_angular, self.max_angular)),
        )

    def terminated(self, task: TaskState) -> bool:
        """The episode succeeded, crashed or left the map."""
        return bool(task.goal_reached or task.collision or task.out_of_bounds)

    def truncated(self, task: TaskState) -> bool:
        """The episode spent its budget of world time, or the world stopped moving altogether.

        The budget is compared with the rounding tolerance of :data:`_BUDGET_ROUNDING`, so a budget that
        falls on the grid of control periods ends on the step that reaches it rather than one period later.
        The tolerance only ever shortens the episode by that fraction of a second, never lengthens it, and
        the overshoot of a budget that is *not* on the grid stays what it always was: less than one control
        period, since a step is the smallest slice of world time an episode can end on.
        """
        return bool(
            task.elapsed_seconds >= self.max_episode_seconds - _BUDGET_ROUNDING
        ) or self.stalled()

    def stalled(self) -> bool:
        """Whether the world's own clock has stopped while the environment kept stepping.

        The episode is counted in the world's seconds, so a world that stopped - a session paused from the
        Unity UI, a simulator that went away - would otherwise never reach its budget and a training loop
        would step a still world for ever. This is what ends it instead, and ``info["stalled"]`` says so.
        Zero or less turns the guard off, for a caller that would rather block than truncate.
        """
        if self.stall_timeout <= 0.0 or self._last_sim_change is None:
            return False
        return time.monotonic() - self._last_sim_change >= self.stall_timeout

    def info(self, world: World, task: TaskState) -> dict[str, Any]:
        """The per-step dictionary handed to the caller.

        ``world`` is the whole typed read, so a callback or a subclass has the
        scan, the agents and the grid without reading the session again. The
        rest are the scalars a logger wants, and the three clocks: the
        simulator's own (``sim_time_seconds``), the episode as this environment
        counts it, and the wall.
        """
        return {
            "world": world,
            "observation_names": self.observation_names,
            "distance_to_goal": task.distance_to_goal,
            "bearing_to_goal": task.bearing_to_goal,
            "goal_reached": task.goal_reached,
            "collision": task.collision,
            "out_of_bounds": task.out_of_bounds,
            "min_lidar": task.min_lidar,
            "nearest_agent": task.nearest_agent_distance,
            "episode_seconds": task.elapsed_seconds,
            "wall_seconds": task.wall_seconds,
            "stalled": self.stalled(),
            "sim_time_seconds": world.sim_time_seconds,
            "scenario_seconds": self.client.scenario_time_seconds,
            "scenario_id": world.scenario_id,
            "steps": task.steps,
            "action": self._last_command,
        }
