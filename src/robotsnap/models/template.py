"""A worked example: copy this file, rename it, and make it yours.

This is a complete model, not a sketch: its environment inherits
:class:`~robotsnap.envs.base.RobotSNAPEnv` and runs, its agent really learns,
and the catalogue registers it beside CADRL and SARL. Four things are yours to
change, and each is marked in the code below with what it is and why you would
want another one: the observation, the action set, the reward, and the
agent/policy.

To build your own:

1. copy this file to ``models/<your_method>.py`` and rename the three classes;
2. point ``ENVIRONMENT`` and ``AGENT`` at them, as the bottom of this file does;
3. add one entry to ``METHODS`` and ``METHOD_LEARNER`` in
   :mod:`robotsnap.models`, and one name to ``_METHOD_MODULES``;

then ``python -m robotsnap train --method <your_method>`` runs it, and
``--save`` writes the checkpoint ``play --method <your_method> --load`` reads
back. ``tests/robotsnap/models/test_template.py`` is what a test of your own
looks like.
"""

from __future__ import annotations

import math
from typing import Any

import gymnasium
import numpy as np
from gymnasium import spaces

from robotsnap.envs.base import RobotSNAPEnv, TaskState
from robotsnap.envs.world import World, wrap_angle
from robotsnap.models.social import _DQNAgent, pack_flat_observation

__all__ = [
    "AGENT",
    "ENVIRONMENT",
    "TEMPLATE_COMMANDS",
    "TEMPLATE_OBSERVATION_SIZE",
    "TemplateAgent",
    "TemplateEnv",
    "template_value",
]

#: How wide the reduced observation is: goal error ``dx, dy``, the distance, the
#: heading error and the forward speed. Five numbers, and no crowd.
TEMPLATE_OBSERVATION_SIZE = 5

#: The three commands of the tiny action set: stand still, go straight, turn.
TEMPLATE_COMMANDS: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),
    (0.5, 0.0),
    (0.0, 0.5),
)

#: This task keeps room for no neighbour at all, and says so where the run line
#: and :mod:`robotsnap.runs.spec` read it.
TEMPLATE_MAX_NEIGHBOURS = 0


class TemplateEnv(RobotSNAPEnv):
    """The four axes of a task, in the smallest environment that has all four.

    Inheriting the base class is the whole of the transport: launch the
    scenario, pace the steps, read the world, write ``/cmd_vel`` and end the
    episode on the goal, a crash or the time budget all come from
    :class:`~robotsnap.envs.base.RobotSNAPEnv`. Overriding the four methods
    below is what makes it *this* task.
    """

    #: No crowd, so no neighbour rows. A run reads this to describe itself, and
    #: :mod:`robotsnap.runs.spec` reads it to name the observation, which is why it
    #: is declared and not implied.
    max_neighbours = TEMPLATE_MAX_NEIGHBOURS

    def __init__(
        self,
        *,
        max_neighbours: int = TEMPLATE_MAX_NEIGHBOURS,
        social_parameters: Any = None,
        **kwargs: Any,
    ):
        # The shared method runner hands every method a neighbour room and a set
        # of social weights. This task has neither, so it accepts them and drops
        # them rather than making the command line know which method it drives.
        del max_neighbours, social_parameters
        kwargs.setdefault("observations", "none")
        # Two terms are paid below and nothing else. The shared method runner
        # hands every method the same three penalties, so they are fixed here
        # rather than defaulted: a run that printed weights this task does not
        # pay would be describing a reward it does not have. Delete these three
        # lines and read them in :meth:`reward` to make the shaping yours.
        kwargs["collision_penalty"] = 0.0
        kwargs["out_of_bounds_penalty"] = 0.0
        kwargs["time_penalty"] = 0.0
        super().__init__(**kwargs)

    # -- axis 1: what the robot sees --------------------------------------

    def make_observation_space(self) -> gymnasium.Space:
        """The reduced observation, as one flat vector.

        Change this when the policy needs something the base parts do not
        carry: a different frame, a history, a learned embedding. Keeping the
        space and :meth:`observation` in one class is what stops the two from
        disagreeing about the width.
        """
        return spaces.Box(
            low=-np.inf, high=np.inf, shape=(TEMPLATE_OBSERVATION_SIZE,), dtype=np.float32
        )

    def observation(self, world: World, task: TaskState) -> np.ndarray:
        """The goal in the robot frame, the heading error and the speed.

        Deliberately less than the structured social observation: no distance
        to a human, no neighbour rows. A study that wants a smaller state - to
        train faster, or to show what the crowd costs - starts by cutting here.
        """
        distance = task.distance_to_goal
        bearing = task.bearing_to_goal
        if distance is None or bearing is None:
            return np.zeros(TEMPLATE_OBSERVATION_SIZE, dtype=np.float32)
        return np.array(
            [
                float(distance) * math.cos(bearing),
                float(distance) * math.sin(bearing),
                float(distance),
                wrap_angle(bearing),
                float(world.linear_velocity),
            ],
            dtype=np.float32,
        )

    # -- axis 2: what the robot can do -------------------------------------

    def make_action_space(self) -> gymnasium.Space:
        """Three commands instead of the eighty of the CADRL grid.

        Change this when the platform's commands are not a grid of speeds and
        headings - a differential drive with no reverse, a holonomic base, a
        waypoint to follow - and the matching :meth:`action_to_command`.
        """
        return spaces.Discrete(len(TEMPLATE_COMMANDS))

    def action_to_command(self, action: Any) -> tuple[float, float]:
        """Turn an action index into a clipped ``(linear_x, angular_z)``."""
        index = int(action)
        if not 0 <= index < len(TEMPLATE_COMMANDS):
            raise ValueError(
                f"action {index} is outside the {len(TEMPLATE_COMMANDS)} template commands"
            )
        return super().action_to_command(TEMPLATE_COMMANDS[index])

    # -- axis 3: what the robot is paid ------------------------------------

    def reward(
        self,
        previous: TaskState,
        current: TaskState,
        action: tuple[float, float],
        world: World,
    ) -> float:
        """Two terms: the metres taken off the goal, and a prize for arriving.

        A study that wants a different shaping - a social charge, a comfort
        term, a penalty for turning in place - adds it here and leaves the
        reward family of the social methods untouched.
        """
        value = 0.0
        if previous.distance_to_goal is not None and current.distance_to_goal is not None:
            value += self.progress_reward * (
                float(previous.distance_to_goal) - float(current.distance_to_goal)
            )
        if current.goal_reached:
            value += self.goal_reward
        return float(value)


# -- axis 4: what learns to act --------------------------------------------

_VALUE_CLASS: type | None = None


def template_value(**kwargs: Any):
    """Build the template's policy: a plain MLP over the reduced observation.

    The class is built here rather than at module import so that copying this
    file does not put torch on the import path of whoever only reads the
    catalogue. Replace it with your own ``nn.Module`` when your observation or
    your action set is not this shape; the contract an agent needs is the two
    members below and nothing else.
    """
    global _VALUE_CLASS
    if _VALUE_CLASS is None:
        import torch
        import torch.nn as nn

        class TemplateValue(nn.Module):
            """One value per action, straight from the observation.

            Two members are the contract with :class:`_DQNAgent`:
            :meth:`set_action_table` fixes which index is which command, and
            ``forward`` returns ``(batch, actions)`` values. The neighbour
            arguments are accepted and ignored, because this task has no crowd
            and the shared DQN still moves the same triple.
            """

            def __init__(
                self,
                *,
                ego_features: int,
                actions: int,
                neighbour_features: int = 1,
                action_features: int = 2,
                hidden: int = 32,
            ):
                super().__init__()
                self.ego_features = int(ego_features)
                self.neighbour_features = int(neighbour_features)
                self.action_features = int(action_features)
                self.actions = int(actions)
                self.hidden = int(hidden)
                self.body = nn.Sequential(
                    nn.Linear(self.ego_features, self.hidden),
                    nn.ReLU(),
                    nn.Linear(self.hidden, self.hidden),
                    nn.ReLU(),
                )
                self.head = nn.Linear(self.hidden, self.actions)
                self.register_buffer("action_table", torch.zeros(0, self.action_features))

            def set_action_table(self, table: Any) -> "TemplateValue":
                """Record the candidate commands; the table travels in the file."""
                values = torch.as_tensor(np.asarray(table, dtype=np.float32))
                if values.ndim != 2 or values.shape[1] != self.action_features:
                    raise ValueError(
                        f"an action table is (actions, {self.action_features}), "
                        f"got shape {tuple(values.shape)}"
                    )
                if self.actions != values.shape[0]:
                    raise ValueError(
                        f"the table holds {values.shape[0]} actions but this policy was built "
                        f"for {self.actions}"
                    )
                self.action_table = values
                return self

            def forward(
                self, ego: Any, neighbours: Any = None, mask: Any = None, actions: Any = None
            ):
                reference = next(self.parameters())
                values = torch.as_tensor(
                    np.asarray(ego), dtype=reference.dtype, device=reference.device
                )
                return self.head(self.body(values))

        _VALUE_CLASS = TemplateValue
    return _VALUE_CLASS(**kwargs)


class TemplateAgent(_DQNAgent):
    """The same DQN as the social methods, over the template's own policy.

    Nothing in the learner changes when the observation or the network does:
    the buffer, the target network, epsilon and the checkpoint are the shared
    ones. Replace this class when the *learning rule* is what you want to
    change - a policy gradient, a tabular method, a model-based planner - and
    keep :meth:`_DQNAgent.act`, :meth:`_DQNAgent.observe`, :meth:`_DQNAgent.learn`
    and :meth:`_DQNAgent.save` as the interface the run loop drives.
    """

    name = "template"

    @staticmethod
    def size_from(environment: Any) -> dict[str, int]:
        """The widths this agent reads off the environment it will drive."""
        return {
            "ego_features": int(environment.observation_space.shape[0]),
            "neighbour_features": 1,
        }

    def _pack(self, observation: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """A flat observation as the triple the shared DQN moves."""
        return pack_flat_observation(observation)

    def _build_model(self):
        return template_value(
            ego_features=self.ego_features,
            neighbour_features=self.neighbour_features,
            actions=self.actions,
            action_features=self.action_features,
            hidden=self.hidden,
        )


#: What the catalogue registers for ``template``: the task and the agent on it.
ENVIRONMENT = TemplateEnv
AGENT = TemplateAgent
