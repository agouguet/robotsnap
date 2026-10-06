"""The scripted controller a run uses, and the polar encoding it shares.

The controller reads the world, not the observation, so the same task can be
measured with it and with a learned policy.
"""

from __future__ import annotations

import math


#: The observation of the social task: goal distance, goal bearing as a unit
#: vector, the same for the closest neighbour, then the lidar.
_POLAR = 3


def _polar(distance, bearing, scale: float):
    """A distance and a bearing as three bounded numbers, or zeros if there is none."""
    import numpy as np

    if distance is None or bearing is None:
        return np.zeros(_POLAR, dtype=np.float32)
    return np.array(
        [
            min(float(distance), scale) / scale,
            math.cos(bearing),
            math.sin(bearing),
        ],
        dtype=np.float32,
    )


def go_to_goal(world, *, linear: float = 0.8, gain: float = 1.5):
    """A proportional controller: turn toward the goal, slow down when far off.

    Not a learning method, and that is the point: the same task measures both.
    It reads the world rather than the observation, so it is written against the
    typed read instead of against whatever vector a subclass hands out.

    The sign is the measured one: on the live scene, holding ``angular.z`` at
    +0.5 for two seconds took the published yaw from 0.00 to -0.63 rad, so a
    positive command turns the robot clockwise in the ROS frame the poses are
    published in. Turning toward a goal that sits at a positive bearing therefore
    means a negative command. (The odometry twist is the odd one out: it reported
    +0.30 rad/s for the same turn, the opposite sign of the yaw in its own pose.)
    """
    import numpy as np

    bearing = world.bearing_to_goal
    if bearing is None:
        return np.zeros(2, dtype=np.float32)
    angular = float(np.clip(-gain * bearing, -1.0, 1.0))
    forward = linear * max(0.0, math.cos(bearing))
    return np.array([forward, angular], dtype=np.float32)
