"""The Gymnasium spaces and the assembled observation of a RobotSNAP episode.

This module owns the contract a learner sees: the action space it samples from,
the observation space it is checked against, and the one
:meth:`SpacesMixin.observation` call that stacks the configured parts. Which
parts appear, in which order and with which parameters is the ``observations``
argument of the environment, so nothing here is specific to one task.

A subclass that changes the action (a discrete set of velocities, a waypoint)
replaces :meth:`make_action_space` together with
:meth:`~robotsnap.envs.task.TaskMixin.action_to_command`, so the space and the
wire command cannot drift.
"""

from __future__ import annotations

import numpy as np

try:  # the package itself has no dependency on gymnasium; this module has one.
    import gymnasium
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover - exercised by the extras, not the suite
    raise ImportError(
        "robotsnap.envs needs gymnasium: pip install 'RobotSNAP[env]'"
    ) from exc

from robotsnap.envs.observation import ObservationContext
from robotsnap.envs.task import TaskState
from robotsnap.envs.world import World

__all__ = ["SpacesMixin"]


class SpacesMixin:
    """The Gymnasium spaces and observation of :class:`~robotsnap.envs.base.RobotSNAPEnv`."""

    def make_action_space(self) -> gymnasium.Space:
        """The action space, ``[linear_x, angular_z]`` by default.

        A subclass that drives another way - a discrete set of velocities, a
        waypoint, a goal pose - replaces this together with
        :meth:`action_to_command`, and the two are read from the same place so
        they cannot drift.
        """
        return spaces.Box(
            low=np.array([-self.max_linear, -self.max_angular], dtype=np.float32),
            high=np.array([self.max_linear, self.max_angular], dtype=np.float32),
            dtype=np.float32,
        )

    def make_observation_space(self) -> gymnasium.Space:
        """The space of :meth:`observation`, matching it feature for feature.

        It is built from the bounds the parts declare, so it follows whatever
        ``observations`` asked for; a structured observation gets a
        ``spaces.Dict`` of the same parts, in the same order.
        """
        if self.observation_structure == "dict":
            return spaces.Dict(
                {
                    part.name: spaces.Box(
                        low=part.low, high=part.high, dtype=np.float32
                    )
                    for part in self.observation_parts
                }
            )
        low, high = self.observation_bounds()
        return spaces.Box(
            low=low,
            high=high,
            dtype=np.float32,
        )

    def observation_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """``(low, high)`` of the observation that was configured, in its order.

        The bounds of the flat vector, whatever the structure: a caller that
        wants one part's own bounds reads them on the part itself.
        """
        if not self.observation_parts:
            return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
        low = np.concatenate([part.low for part in self.observation_parts])
        high = np.concatenate([part.high for part in self.observation_parts])
        return low.astype(np.float32), high.astype(np.float32)

    def observation(self, world: World, task: TaskState) -> np.ndarray:
        """What the agent sees this step, assembled from the configured parts.

        Flat and of a fixed size by default, so a plain MLP can take it, and one
        array per part when the environment was built with
        ``observation_structure="dict"``. Which parts appear, in which order and
        with which parameters, is the ``observations`` argument of the
        constructor. Every part is still built by the ``*_features`` method
        below, so a subclass replaces the method it cares about and leaves the
        rest alone.
        """
        context = ObservationContext(env=self, world=world, task=task)
        if self.observation_structure == "dict":
            return {part.name: part.value(context) for part in self.observation_parts}
        if not self.observation_parts:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(
            [part.value(context) for part in self.observation_parts]
        ).astype(np.float32)

    def _apply_observation_sizes(self) -> None:
        """Let a part's own ``max``/``bins`` parameter size the environment.

        The size of a part is read from the environment when it is resolved, so
        the parameter has to land on the attribute first; keeping them in step
        here is what stops ``observations={"lidar": {"bins": 16}}`` from
        declaring one size and building another.
        """
        agents = self._observation_params.get("agents", {})
        if "max" in agents:
            self.max_agents = int(agents["max"])
        humans = self._observation_params.get("humans", {})
        if "max" in humans:
            self.max_humans = int(humans["max"])
        lidar = self._observation_params.get("lidar", {})
        if "bins" in lidar:
            self.lidar_bins = int(lidar["bins"])
        if self.max_agents < 0 or self.max_humans < 0 or self.lidar_bins <= 0:
            raise ValueError(
                "an observation size must be positive: max and bins cannot be "
                "negative, bins cannot be zero"
            )

    def _observation_layout(self) -> dict[str, slice]:
        """Where each part sits in the flat vector, by name."""
        layout: dict[str, slice] = {}
        start = 0
        for part in self.observation_parts:
            layout[part.name] = slice(start, start + part.size)
            start += part.size
        return layout
