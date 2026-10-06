"""Gymnasium environments over a RobotSNAP session.

The package is an extra, not a requirement: ``pip install 'RobotSNAP[env]'``
brings ``gymnasium``, and importing this module without it says so. Nothing in
``robotsnap.bridge``, ``robotsnap.client`` or ``robotsnap.viewer`` imports it,
so a session that only drives or watches the simulator stays free of it.

:class:`~robotsnap.envs.base.RobotSNAPEnv` is the environment to inherit from;
:func:`~robotsnap.envs.base.RobotSNAPEnv` documents which methods a subclass
replaces to change the observation, the reward, the actions or the end of an
episode, and :mod:`robotsnap.envs.observation` holds the named parts an
observation is assembled from - picking them is an argument of the constructor,
and registering a new one never touches the environment class.
"""

from __future__ import annotations

from robotsnap.envs.base import RobotSNAPEnv, RobotSNAPEnvError, TaskState
from robotsnap.envs.observation import (
    DEFAULT_OBSERVATIONS,
    ObservationContext,
    ObservationPart,
    describe_observation_parts,
    register_observation_part,
    registered_observation_names,
)
from robotsnap.envs.world import (
    Agent,
    Human,
    LidarScan,
    OccupancyMap,
    Pose2D,
    World,
    read_world,
)

__all__ = [
    "RobotSNAPEnv",
    "RobotSNAPEnvError",
    "TaskState",
    "DEFAULT_OBSERVATIONS",
    "ObservationContext",
    "ObservationPart",
    "describe_observation_parts",
    "register_observation_part",
    "registered_observation_names",
    "Agent",
    "Human",
    "LidarScan",
    "OccupancyMap",
    "Pose2D",
    "World",
    "read_world",
    "ENV_ID",
    "register_envs",
]

#: Id the goal-navigation environment registers under.
ENV_ID = "RobotSNAP/GoalNavigation-v0"


def register_envs(force: bool = False) -> str:
    """Register the environments with gymnasium, and return the id.

    Registration keeps the environment out of a registry until a caller asks for
    it, since the default constructor needs a running simulator to be useful.
    The environment checker of ``gymnasium.make`` is switched off for that same
    reason: it would probe a live session.

    ``force`` registers over an existing entry, for a session that changed the
    class while the interpreter stayed up.
    """
    from gymnasium.envs import registration

    if ENV_ID not in registration.registry or force:
        registration.register(
            id=ENV_ID,
            entry_point="robotsnap.envs.base:RobotSNAPEnv",
            disable_env_checker=True,
            order_enforce=False,
            kwargs={},
        )
    return ENV_ID
