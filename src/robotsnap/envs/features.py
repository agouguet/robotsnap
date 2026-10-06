"""The pieces a RobotSNAP observation is assembled from.

Each ``*_features`` method here turns one read
:class:`~robotsnap.envs.world.World` into one block of numbers. The environment
composes them in the order its ``observations`` argument asked for - see
:mod:`robotsnap.envs.observation` - so this module only knows what a robot, a
goal, a neighbour or a sweep looks like to the agent.

A subclass that wants a different robot block, a different goal framing or a
different neighbour layout replaces the one method it cares about and leaves the
others alone.
"""

from __future__ import annotations

import math

import numpy as np

from robotsnap.envs.task import TaskState
from robotsnap.envs.world import Pose2D, World

__all__ = ["FeaturesMixin"]


class FeaturesMixin:
    """The observation feature builders of :class:`~robotsnap.envs.base.RobotSNAPEnv`."""

    def robot_features(self, world: World) -> np.ndarray:
        """Pose and body velocity: ``x, y, cos yaw, sin yaw, v, w``.

        The heading travels as its cosine and sine rather than as an angle, so
        nothing in it wraps; velocity and yaw rate are the ones ``/odom``
        reports, in the ROS frame.

        One measured caveat about the last one: the yaw rate of the twist carries
        the opposite sign of the yaw rate of the pose it is published with - held
        at ``angular.z = +0.5``, the live scene took its published yaw from 0.00
        to -0.63 rad while reporting ``+0.30`` rad/s. The pose is the one the rest
        of the observation agrees with, so a subclass that needs a consistent yaw
        rate should difference that yaw rather than take it from the twist.
        """
        pose = world.pose or Pose2D(0.0, 0.0, 0.0)
        return np.array(
            [
                pose.x,
                pose.y,
                math.cos(pose.yaw),
                math.sin(pose.yaw),
                world.linear_velocity,
                world.angular_velocity,
            ],
            dtype=np.float32,
        )

    def goal_features(self, world: World, task: TaskState) -> np.ndarray:
        """The goal in the robot frame: ``dx, dy, distance, cos bearing, sin bearing``.

        Zeros when the session has no goal to publish, which is what a scenario
        with no goal for that robot looks like.
        """
        distance = task.distance_to_goal
        bearing = task.bearing_to_goal
        if distance is None or bearing is None:
            return np.zeros(5, dtype=np.float32)
        return np.array(
            [
                distance * math.cos(bearing),
                distance * math.sin(bearing),
                distance,
                math.cos(bearing),
                math.sin(bearing),
            ],
            dtype=np.float32,
        )

    def agent_features(self, world: World) -> np.ndarray:
        """The nearest ``max_agents`` neighbours, in the robot frame.

        One row per agent, nearest first: ``dx, dy, vx, vy, visible``. Rows that
        no neighbour fills are zeros with a zero visibility flag, so a network
        can tell "nothing there" from "a neighbour at the origin". Only agents
        the simulator reports are considered, and ``visible`` is the lidar
        visibility the simulator flags them with.
        """
        rows = np.zeros((self.max_agents, 5), dtype=np.float32)
        if self.max_agents == 0:
            return rows.reshape(0)
        nearest = sorted(world.agents, key=lambda agent: agent.distance)
        for index, agent in enumerate(nearest[: self.max_agents]):
            rows[index] = (
                agent.x,
                agent.y,
                agent.vx,
                agent.vy,
                1.0 if agent.visible else 0.0,
            )
        return rows.reshape(-1)

    def lidar_features(self, world: World) -> np.ndarray:
        """The sweep reduced to ``lidar_bins`` ranges, normalised to ``[0, 1]``.

        One means "that direction is clear to the sensor's reach", zero means
        "touching": the scale is the sensor's own ``range_max``, and a session
        that has not published a scan yet reads as clear.
        """
        if world.scan is None:
            return np.ones(self.lidar_bins, dtype=np.float32)
        span = float(world.scan.range_max)
        ranges = world.scan.resample(self.lidar_bins)
        if span <= 0.0:
            return np.zeros(self.lidar_bins, dtype=np.float32)
        return np.clip(ranges / span, 0.0, 1.0).astype(np.float32)
