"""What every social-navigation method in this catalogue shares.

A method of the literature is more than a learner: it is an observation, an
action set, a reward and a policy at once. This module holds the pieces CADRL
and SARL agree on, so neither of them repeats a line of them, and it is the
*shared* half of :mod:`robotsnap.models`: each named method lives in its own
module beside it and adds only what makes it that method.

Four things are shared, each as a single definition rather than a copy:

- :class:`SocialActionSpace` - the discrete ``(speed, heading offset)`` grid the
  papers plan in, plus the one rule that turns a heading offset into the angular
  velocity ``/cmd_vel`` wants;
- the *layout* of a structured observation - the keys, their sizes, and
  :func:`build_structured_observation`, which is the only place a world becomes
  those arrays;
- :class:`SocialReward` - the hand-written social cost of Chen et al. (2017),
  with every weight exposed, which both methods are rewarded by;
- :class:`SocialNavEnv`, :func:`state_value_base`, :class:`LookaheadAgent` and
  :class:`_DQNAgent` - the shared task, the value net that folds a varying crowd
  into one vector, and the two learners a method picks between: the V(s) learner
  with a one-step lookahead that CADRL and SARL use, and the plain DQN the
  template still inherits.

Nothing here imports torch: the value network is built on the first call that
needs it, and the environment needs only gymnasium. A command line that only
lists what is available, or a test that only checks an observation, therefore
pays for numpy and gymnasium and no more.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import gymnasium
import numpy as np
from gymnasium import spaces

from robotsnap.envs.base import DEFAULT_CONTROL_PERIOD, RobotSNAPEnv, TaskState
from robotsnap.envs.world import World, wrap_angle

__all__ = [
    "DEFAULT_HEADINGS",
    "DEFAULT_MAX_NEIGHBOURS",
    "DEFAULT_SOCIAL_PARAMETERS",
    "DEFAULT_SPEEDS",
    "EGO_KEY",
    "EGO_SIZE",
    "FORMAT",
    "GOAL_KEY",
    "GOAL_SIZE",
    "HISTORY_KEYS",
    "MASK_KEY",
    "NEIGHBOURS_KEY",
    "NEIGHBOUR_SIZE",
    "SocialActionSpace",
    "SocialNavEnv",
    "SocialReward",
    "as_social_reward",
    "build_structured_observation",
    "heading_offset_to_angular",
    "pack_flat_observation",
    "pack_observation",
    "play",
    "train",
    "value_net_base",
]


# ---------------------------------------------------------------------------
# The action set: the discrete command grid of the CADRL papers.
# ---------------------------------------------------------------------------

#: Speeds and heading offsets the classic CADRL set is sampled at: five speeds
#: up to the linear limit, sixteen offsets over a full turn.
DEFAULT_SPEEDS = 5
DEFAULT_HEADINGS = 16
#: Tolerance two commands are compared within, in m/s and rad/s.
DEFAULT_TOLERANCE = 1e-6


def heading_offset_to_angular(offset: float, control_period: float) -> float:
    """The angular velocity that turns ``offset`` radians in one control period.

    This is the whole of the frame the papers work in versus the frame the
    simulator takes: an action says how far to turn per step, and ``/cmd_vel``
    wants a speed. Dividing by the period the step covers is the only sensible
    bridge between the two, and keeping it in one function means the action set
    and any caller that reasons about an action cannot disagree.

    The value is *not* clipped here. An offset larger than the controller can
    turn in one period produces an angular velocity the environment will clip,
    which is the same treatment every other command gets.
    """
    period = float(control_period)
    if not math.isfinite(period) or period <= 0.0:
        raise ValueError("control_period must be a finite, positive number of seconds")
    return float(offset) / period


class SocialActionSpace:
    """A fixed grid of ``(speed, heading offset)`` actions, the zero action first.

    ``speeds`` is either the number of speeds or the speeds themselves; a count
    is spread from zero to ``max_linear``, so the default five are the
    ``0, 0.25, 0.5, 0.75, 1`` of the papers. ``headings`` is the number of
    heading offsets spread over a full turn, so the default sixteen cover
    ``[0, 2*pi)`` in equal steps.

    The grid is the cross product of the two, and its ``(0, 0)`` cell - no
    speed, no turn - *is* the do-nothing action the papers list on its own, so
    it appears once, at index ``0``, and the rest of the grid fills the indices
    after it. With the defaults the space therefore holds ``5 * 16`` actions,
    the size of the classic CADRL set, and every command in it is distinct,
    which is what lets :meth:`index_of` be an exact inverse of indexing.

    ``control_period`` is the simulated seconds one command covers and is only
    used for the heading conversion, so an action means the same thing whatever
    the session's clock runs at.
    """

    def __init__(
        self,
        *,
        speeds: int | Sequence[float] = DEFAULT_SPEEDS,
        headings: int = DEFAULT_HEADINGS,
        control_period: float = DEFAULT_CONTROL_PERIOD,
        max_linear: float = 1.0,
    ):
        count = int(headings)
        if count <= 0:
            raise ValueError("the number of headings must be positive")
        if isinstance(speeds, (int, np.integer)):
            if int(speeds) <= 0:
                raise ValueError("the number of speeds must be positive")
            speed_values = np.linspace(0.0, float(max_linear), int(speeds))
        else:
            speed_values = np.asarray(list(speeds), dtype=np.float64)
            if speed_values.size == 0:
                raise ValueError("at least one speed is needed")
        # Validate the period once, through the conversion that uses it.
        heading_offset_to_angular(0.0, control_period)

        self.speeds: tuple[float, ...] = tuple(float(value) for value in speed_values)
        self.heading_offsets: tuple[float, ...] = tuple(
            2.0 * math.pi * index / count for index in range(count)
        )
        self.headings = count
        self.control_period = float(control_period)
        self.max_linear = float(max_linear)

        zero = (0.0, 0.0)
        commands: list[tuple[float, float]] = [zero]
        for speed in self.speeds:
            for offset in self.heading_offsets:
                command = (
                    float(speed),
                    heading_offset_to_angular(offset, self.control_period),
                )
                # The (0, 0) cell of the grid is the do-nothing action already
                # listed first, so it is not listed a second time.
                if command == zero:
                    continue
                commands.append(command)
        self._commands: tuple[tuple[float, float], ...] = tuple(commands)

    def __len__(self) -> int:
        return len(self._commands)

    def __getitem__(self, index: int) -> tuple[float, float]:
        """The ``(linear_x, angular_z)`` command of one action."""
        return self._commands[int(index)]

    def __iter__(self) -> Iterator[tuple[float, float]]:
        return iter(self._commands)

    @property
    def commands(self) -> np.ndarray:
        """Every command, as an ``(actions, 2)`` float32 array.

        This is what a value network wants as its candidate-action block: one
        row per action, in index order.
        """
        return np.asarray(self._commands, dtype=np.float32).reshape(len(self), 2)

    def index_of(
        self, linear_x: float, angular_z: float, tolerance: float = DEFAULT_TOLERANCE
    ) -> int:
        """The index whose command is ``(linear_x, angular_z)``.

        A command that is not in the set raises :class:`ValueError` rather than
        returning ``-1``: a caller that asks for an action the space does not
        hold has made a mistake, and a silent miss would read as the wrong
        action being taken.
        """
        target = (float(linear_x), float(angular_z))
        for index, command in enumerate(self._commands):
            if (
                abs(command[0] - target[0]) <= tolerance
                and abs(command[1] - target[1]) <= tolerance
            ):
                return index
        raise ValueError(
            f"{target} is not one of the {len(self)} social actions; "
            "the set holds (speed, heading offset / control_period) pairs"
        )

    def random_index(self, rng, near_stop_probability: float = 0.0) -> int:
        """A random index, biased towards the do-nothing action.

        The papers explore by sampling an action uniformly and, with some
        probability, standing still instead - a robot that never tries the zero
        action learns late that stopping is often the safest thing it can do.
        ``rng`` is a :class:`numpy.random.Generator`; anything else with an
        ``integers`` or ``randrange`` method works too.
        """
        probability = float(near_stop_probability)
        if probability > 0.0 and float(rng.random()) < probability:
            return 0
        draw = getattr(rng, "integers", None)
        if draw is not None:
            return int(draw(len(self)))
        return int(rng.randrange(len(self)))

    def describe(self) -> str:
        """A short summary of the grid, for a log line or a help screen."""
        return "\n".join(
            [
                f"{len(self)} social actions: {len(self.speeds)} speed(s) x "
                f"{self.headings} heading(s)",
                f"speeds {min(self.speeds):g}..{max(self.speeds):g} m/s",
                "heading offsets cover a full turn",
                f"a heading offset becomes angular_z = offset / {self.control_period:g} s",
                "index 0 is the do-nothing action (0, 0)",
            ]
        )


# ---------------------------------------------------------------------------
# The observation: one layout, and the one function that builds it.
# ---------------------------------------------------------------------------

#: The keys of a structured social observation, and their sizes.
EGO_KEY = "ego"
GOAL_KEY = "goal"
NEIGHBOURS_KEY = "neighbours"
MASK_KEY = "mask"

#: The robot's own features: goal position error ``x, y`` in the robot frame,
#: the heading error, and the body velocity ``v, w``.
EGO_SIZE = 5
#: The goal in the robot frame: ``dx, dy, distance, cos bearing, sin bearing``.
GOAL_SIZE = 5
#: One neighbour: relative position ``dx, dy`` and velocity ``vx, vy``.
NEIGHBOUR_SIZE = 4


def pack_observation(
    structured: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(ego, neighbours, mask)`` arrays from one structured observation.

    The goal is folded into the ego vector here rather than fed to the network
    as a third input: both describe the robot's own situation, and a value
    function that had to keep them apart would have nothing to gain from it.
    The neighbours keep their ``(max_neighbours, 4)`` shape and the mask its
    ``(max_neighbours,)`` one, which is exactly what the value networks take.
    """
    ego = np.concatenate(
        [
            np.asarray(structured[EGO_KEY], dtype=np.float32).reshape(-1),
            np.asarray(structured[GOAL_KEY], dtype=np.float32).reshape(-1),
        ]
    )
    neighbours = np.asarray(structured[NEIGHBOURS_KEY], dtype=np.float32)
    mask = np.asarray(structured[MASK_KEY]).astype(bool)
    if neighbours.ndim != 2:
        raise ValueError(
            f"neighbours are a (count, {NEIGHBOUR_SIZE}) array, got shape {neighbours.shape}"
        )
    if mask.shape != (neighbours.shape[0],):
        raise ValueError(
            f"the mask covers {mask.shape} but there are {neighbours.shape[0]} neighbour rows"
        )
    return ego, neighbours, mask


def pack_flat_observation(
    observation: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(ego, neighbours, mask)`` arrays from one flat observation.

    A task with no crowd to encode hands the DQN the same triple the structured
    packer does, with the whole vector as the ego features and one neighbour row
    flagged absent. That is the same statement the structured packer makes for a
    robot that sees nobody, so the buffer, the update and the checkpoint of the
    DQN are unchanged - only the width of what goes in differs.
    """
    ego = np.asarray(observation, dtype=np.float32).reshape(-1)
    return ego, np.zeros((1, 1), dtype=np.float32), np.zeros(1, dtype=bool)


def build_structured_observation(
    world: World, task: TaskState, max_neighbours: int
) -> dict[str, np.ndarray]:
    """The structured observation of one step, from a world and a task state.

    The layout is the one this module declares, because the networks that
    consume it and this function that builds it must agree to the last index:

    ``ego``
        the robot's own situation - the goal position error in the robot frame,
        the heading error and the body velocity - so a network sees how far it
        is from succeeding without having to difference two poses;
    ``goal``
        the goal in the robot frame, distance and bearing included, which is
        the quantity a reward reads and a policy may want to reason about;
    ``neighbours``
        up to ``max_neighbours`` rows of ``dx, dy, vx, vy``, nearest first,
        which is exactly the frame the simulator publishes agents in;
    ``mask``
        one flag per row, true where a neighbour filled it. The rows past the
        crowd are zeros *and* flagged false, so a network can tell "a neighbour
        at the origin" from "no neighbour here" - the distinction the padding
        rule of the value networks rests on.

    A session publishing no goal leaves the goal fields at zero rather than
    inventing one, and an empty crowd leaves every neighbour row padded. Both
    are ordinary states of a half-built world, not errors.
    """
    distance = task.distance_to_goal
    bearing = task.bearing_to_goal
    if distance is None or bearing is None:
        error_x = error_y = heading_error = 0.0
        goal = np.zeros(GOAL_SIZE, dtype=np.float32)
    else:
        error_x = float(distance) * math.cos(bearing)
        error_y = float(distance) * math.sin(bearing)
        # The bearing to the goal *is* the heading error of a robot that has to
        # point at it, so it is reported unwrapped and then folded here.
        heading_error = wrap_angle(bearing)
        goal = np.array(
            [
                error_x,
                error_y,
                float(distance),
                math.cos(bearing),
                math.sin(bearing),
            ],
            dtype=np.float32,
        )

    ego = np.array(
        [
            error_x,
            error_y,
            heading_error,
            float(world.linear_velocity),
            float(world.angular_velocity),
        ],
        dtype=np.float32,
    )

    neighbours = np.zeros((int(max_neighbours), NEIGHBOUR_SIZE), dtype=np.float32)
    mask = np.zeros(int(max_neighbours), dtype=bool)
    # The agents of a world are already in the robot frame, so their distance is
    # the length of their own position and no pose is needed to sort them.
    nearest = sorted(world.agents, key=lambda agent: math.hypot(agent.x, agent.y))
    for index, agent in enumerate(nearest[: int(max_neighbours)]):
        neighbours[index] = (agent.x, agent.y, agent.vx, agent.vy)
        mask[index] = True

    return {
        EGO_KEY: ego,
        GOAL_KEY: goal,
        NEIGHBOURS_KEY: neighbours,
        MASK_KEY: mask,
    }


# ---------------------------------------------------------------------------
# The reward: the hand-written social cost of the CADRL papers.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SocialReward:
    """The per-step reward of the CADRL cost function, with every weight exposed.

    Chen et al. (2017) do not learn a reward: they write one down, turn its
    negation into the training signal of a value network, and let the network
    learn to predict how expensive a state is. ``progress_weight`` scales the
    metres taken off the distance to the goal, ``collision_penalty`` the flat
    price of a crash, ``goal_bonus`` the flat prize of arriving and
    ``time_penalty`` the price of a second spent. The social part is
    ``personal_space_weight`` times a Gaussian of width ``comfort_distance``,
    plus ``too_close_penalty`` once the human is nearer than
    ``too_close_distance``.
    """

    progress_weight: float = 1.0
    collision_penalty: float = 10.0
    goal_bonus: float = 10.0
    time_penalty: float = 0.1
    comfort_distance: float = 1.0
    personal_space_weight: float = 1.0
    too_close_distance: float = 0.5
    too_close_penalty: float = 5.0

    def social_penalty(self, min_human_distance: float | None) -> float:
        """How much it costs to be ``min_human_distance`` away from a human.

        A missing distance (nobody detected) costs nothing, and neither does an
        infinite one: the two are the same statement - no human is near enough
        to matter.
        """
        if min_human_distance is None:
            return 0.0
        distance = float(min_human_distance)
        if not math.isfinite(distance):
            return 0.0
        distance = max(0.0, distance)
        penalty = self.personal_space_weight * math.exp(
            -((distance / float(self.comfort_distance)) ** 2)
        )
        if distance < float(self.too_close_distance):
            # The Gaussian is bounded, so on its own it would price touching a
            # human at barely more than brushing past one.
            penalty += float(self.too_close_penalty)
        return float(penalty)

    def __call__(
        self,
        *,
        previous_distance_to_goal: float | None = None,
        distance_to_goal: float | None = None,
        min_human_distance: float | None = None,
        collided: bool = False,
        arrived: bool = False,
        dt: float = 0.0,
    ) -> float:
        """The reward of one step, from the numbers a step has already computed.

        ``previous_distance_to_goal`` and ``distance_to_goal`` are the two reads
        that bracket the step; progress is their difference, so a robot that did
        not move is paid nothing for it and a robot that backed away is charged.
        Either end may be ``None`` - a session publishing no goal - and then no
        progress is paid at all.
        """
        value = 0.0
        if previous_distance_to_goal is not None and distance_to_goal is not None:
            value += float(self.progress_weight) * (
                float(previous_distance_to_goal) - float(distance_to_goal)
            )
        value -= float(self.time_penalty) * max(0.0, float(dt))
        value -= self.social_penalty(min_human_distance)
        if collided:
            value -= float(self.collision_penalty)
        if arrived:
            value += float(self.goal_bonus)
        return float(value)

    def parameters(self) -> dict[str, float]:
        """The constants, as a plain mapping, for a config file or a checkpoint."""
        return {field.name: float(getattr(self, field.name)) for field in fields(self)}


#: What :func:`as_social_reward` builds when a caller passes nothing at all.
DEFAULT_SOCIAL_PARAMETERS: dict[str, float] = SocialReward().parameters()


def as_social_reward(parameters: Any = None) -> SocialReward:
    """A :class:`SocialReward` from ``None``, a ready-made one or a mapping.

    An environment takes its social constants as an argument that a config file
    or a command line fills in, so accepting a plain mapping is what keeps that
    argument usable from either. An unknown key is refused rather than ignored:
    a misspelled weight would otherwise read as a weight that did nothing.
    """
    if parameters is None:
        return SocialReward()
    if isinstance(parameters, SocialReward):
        return parameters
    if isinstance(parameters, Mapping):
        known = {field.name for field in fields(SocialReward)}
        unknown = sorted(set(parameters) - known)
        if unknown:
            raise ValueError(
                f"unknown social parameter(s): {', '.join(unknown)}; "
                f"known ones are {', '.join(sorted(known))}"
            )
        return SocialReward(**{str(name): float(value) for name, value in parameters.items()})
    raise TypeError(
        f"social parameters are a SocialReward, a mapping or None, got {type(parameters).__name__}"
    )


# ---------------------------------------------------------------------------
# The task: the environment every named method is a child of.
# ---------------------------------------------------------------------------

#: How many neighbours the structured observation keeps room for.
DEFAULT_MAX_NEIGHBOURS = 8


class SocialNavEnv(RobotSNAPEnv):
    """A robot navigating a crowd, with CADRL's action set, reward and state.

    :class:`~robotsnap.envs.base.RobotSNAPEnv` is the simulator's business -
    launch, pace, read, command - and this subclass replaces the three things
    that are the *task's* business: the action becomes an index into
    :class:`SocialActionSpace`, the reward becomes :class:`SocialReward`, and
    the observation becomes the structured dictionary
    :func:`build_structured_observation` builds.

    The constructor takes every argument the base class takes and three more:
    ``max_neighbours`` sizes the structured observation, ``social_parameters``
    carries the :class:`SocialReward` constants (a ready-made reward, a mapping
    of them, or nothing for the defaults), and ``speeds``/``headings`` size the
    discrete action set.

    A child that has its own shaping names it in :attr:`reward_parameters`
    rather than in the constructor: those constants are the task's starting
    point, and a caller that passes ``social_parameters`` still layers its own
    keys on top of them.

    ``control_period`` and ``max_linear`` are read from the arguments here as
    well as being handed to the base class, because the action set has to exist
    before ``super().__init__`` asks for the action space. They come from the
    same call, so the two readings cannot disagree.

    Nothing is touched at construction time but the configuration: no session is
    waited for, no scenario is loaded, and the client the base class builds is
    only used once an episode actually starts.
    """

    #: The constants this task trains against when a caller asks for none. The
    #: base names the shared social cost; a child whose paper shapes differently
    #: replaces this one attribute and touches no other line. It is a copy, so
    #: a subclass that edits it in place cannot reach the shared constant.
    reward_parameters: Mapping[str, Any] = dict(DEFAULT_SOCIAL_PARAMETERS)

    def __init__(
        self,
        *,
        max_neighbours: int = DEFAULT_MAX_NEIGHBOURS,
        social_parameters: SocialReward | Mapping[str, Any] | None = None,
        speeds: int | tuple[float, ...] = DEFAULT_SPEEDS,
        headings: int = DEFAULT_HEADINGS,
        **kwargs: Any,
    ):
        if int(max_neighbours) <= 0:
            raise ValueError("max_neighbours must be positive")
        self.max_neighbours = int(max_neighbours)
        self.social_actions = SocialActionSpace(
            speeds=speeds,
            headings=headings,
            control_period=float(kwargs.get("control_period", DEFAULT_CONTROL_PERIOD)),
            max_linear=float(kwargs.get("max_linear", 1.0)),
        )
        self.social_reward = self.build_reward(social_parameters)
        # The base class assembles a flat observation out of named parts; this
        # environment hands out its own structured one instead, so it asks for
        # no part at all and the two never disagree about what is published.
        kwargs.setdefault("observations", "none")
        super().__init__(**kwargs)

    def build_reward(
        self, social_parameters: SocialReward | Mapping[str, Any] | None
    ) -> SocialReward:
        """This task's reward, with a caller's constants layered over its own.

        The order matters and is the point of :attr:`reward_parameters`: a
        caller's mapping overrides the constants this environment names and
        leaves the rest of them alone, so changing one weight does not silently
        reset the whole cost to the base defaults. A ready-made
        :class:`SocialReward` is taken as it is, and a key nobody recognises is
        still refused by :func:`as_social_reward`.
        """
        if social_parameters is None:
            return as_social_reward(self.reward_parameters)
        if isinstance(social_parameters, SocialReward):
            return social_parameters
        merged: dict[str, Any] = dict(self.reward_parameters)
        merged.update(dict(social_parameters))
        return as_social_reward(merged)

    # -- what a subclass replaces ------------------------------------------

    def make_action_space(self) -> gymnasium.Space:
        """The discrete indices of the social action set."""
        return spaces.Discrete(len(self.social_actions))

    def action_to_command(self, action: Any) -> tuple[float, float]:
        """Turn an action index into a clipped ``(linear_x, angular_z)``."""
        index = int(action)
        if not 0 <= index < len(self.social_actions):
            raise ValueError(
                f"action {index} is outside the {len(self.social_actions)} social actions"
            )
        return super().action_to_command(self.social_actions[index])

    def make_observation_space(self) -> gymnasium.Space:
        """The structured observation, as a dictionary of arrays.

        The mask is a ``MultiBinary`` rather than a box of zeros and ones: it is
        a statement about which rows exist, so there is no third value for it to
        take and no reason for a network to treat it as a continuous feature.
        """
        return spaces.Dict(
            {
                EGO_KEY: spaces.Box(
                    low=-np.inf, high=np.inf, shape=(EGO_SIZE,), dtype=np.float32
                ),
                GOAL_KEY: spaces.Box(
                    low=-np.inf, high=np.inf, shape=(GOAL_SIZE,), dtype=np.float32
                ),
                NEIGHBOURS_KEY: spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.max_neighbours, NEIGHBOUR_SIZE),
                    dtype=np.float32,
                ),
                MASK_KEY: spaces.MultiBinary(self.max_neighbours),
            }
        )

    def observation(self, world: World, task: TaskState) -> dict[str, np.ndarray]:
        """The structured observation of one step."""
        return build_structured_observation(world, task, self.max_neighbours)

    def min_human_distance(self, world: World) -> float | None:
        """How close the nearest human is, or ``None`` when there is none.

        Both sources of a person count: the agents the robot's detector reports
        (already in the robot frame) and the crowd the session publishes in the
        world frame. A benchmark that wants to separate perception from social
        cost can override this and read one of the two.
        """
        distances = [math.hypot(agent.x, agent.y) for agent in world.agents]
        pose = world.pose
        if pose is not None:
            distances += [
                math.hypot(human.x - pose.x, human.y - pose.y) for human in world.humans
            ]
        return min(distances) if distances else None

    def reward(
        self,
        previous: TaskState,
        current: TaskState,
        action: tuple[float, float],
        world: World,
    ) -> float:
        """The social reward of one step, from the two task states it brackets.

        A crash and leaving the map are both priced as a collision: the base
        class ends the episode on either, and from the crowd's point of view
        there is no difference between hitting a wall and leaving the room.
        """
        return self.social_reward(
            previous_distance_to_goal=previous.distance_to_goal,
            distance_to_goal=current.distance_to_goal,
            min_human_distance=self.min_human_distance(world),
            collided=bool(current.collision or current.out_of_bounds),
            arrived=bool(current.goal_reached),
            dt=max(0.0, current.elapsed_seconds - previous.elapsed_seconds),
        )

    def lookahead_reward(self, propagated: Any, action: int) -> float:
        """The estimated immediate reward ``R_hat(S_t, a_t, S_hat_{t+1})`` of the lookahead.

        Chen et al. (2017) score a candidate command by the cost of the state it
        would lead to one model step later, Eq. (5)-(7): the lookahead sums that
        cost with the discounted value of the propagated state. This method is
        the first half of that sum - the same :attr:`social_reward` the
        environment pays for a real step, read from a state no simulator has
        visited yet, so the planner and the reward the learner trains against
        cannot disagree about their weights or their thresholds.

        ``propagated`` is the propagated state of one candidate: a mapping with
        the structured observation's ``ego``, ``goal``, ``neighbours`` and
        ``mask`` keys, plus the goal distance *before* the step under
        ``previous_distance_to_goal``. The three arrays a raw
        :func:`~robotsnap.models.lookahead.propagate` returns are also accepted,
        in which case the goal distance before the step is read back from the
        ego's goal error and, when the caller does not say otherwise, no
        progress is paid.

        ``distance_to_goal`` is the propagated goal's own distance,
        ``min_human_distance`` the nearest masked neighbour of the propagated
        state (``None`` when the crowd is empty), ``arrived`` is a propagated
        goal inside :attr:`goal_radius`, ``collided`` a propagated neighbour
        inside :attr:`collision_distance`, and ``dt`` is
        :attr:`control_period`. ``action`` is the candidate's index and does not
        change the value - the paper's cost is a function of the state a command
        reaches, not of the command itself - but it keeps the call in the
        ``(S_t, a_t, S_hat_{t+1})`` form the lookahead is written in.
        """
        goal: Any = None
        neighbours: Any = None
        mask: Any = None
        previous: float | None = None

        if isinstance(propagated, Mapping):
            ego = propagated.get(EGO_KEY)
            goal = propagated.get(GOAL_KEY)
            neighbours = propagated.get(NEIGHBOURS_KEY)
            mask = propagated.get(MASK_KEY)
            previous = propagated.get("previous_distance_to_goal")
            if goal is None and ego is not None:
                vector = np.asarray(ego, dtype=np.float64).reshape(-1)
                goal = vector[EGO_SIZE : EGO_SIZE + GOAL_SIZE]
        else:
            entries = list(propagated)
            if len(entries) >= 3:
                ego, neighbours, mask = entries[0], entries[1], entries[2]
                vector = np.asarray(ego, dtype=np.float64).reshape(-1)
                goal = vector[EGO_SIZE : EGO_SIZE + GOAL_SIZE]
            if len(entries) >= 4:
                previous = entries[3]

        distance_to_goal: float | None = None
        if goal is not None:
            vector = np.asarray(goal, dtype=np.float64).reshape(-1)
            if vector.size >= 3:
                distance_to_goal = float(vector[2])
            elif vector.size >= 2:
                distance_to_goal = float(math.hypot(vector[0], vector[1]))
        if previous is None:
            previous = distance_to_goal
        else:
            previous = float(previous)

        min_human_distance: float | None = None
        if neighbours is not None and mask is not None:
            rows = np.asarray(neighbours, dtype=np.float64).reshape(-1, NEIGHBOUR_SIZE)
            keep = np.asarray(mask).astype(bool).reshape(-1)
            if rows.shape[0] == keep.shape[0]:
                distances = [
                    math.hypot(float(row[0]), float(row[1]))
                    for row, valid in zip(rows, keep)
                    if valid
                ]
                if distances:
                    min_human_distance = min(distances)

        collided = (
            min_human_distance is not None
            and min_human_distance < float(self.collision_distance)
        )
        arrived = (
            distance_to_goal is not None and distance_to_goal < float(self.goal_radius)
        )
        return self.social_reward(
            previous_distance_to_goal=previous,
            distance_to_goal=distance_to_goal,
            min_human_distance=min_human_distance,
            collided=collided,
            arrived=arrived,
            dt=float(self.control_period),
        )


# ---------------------------------------------------------------------------
# The value network: one base, one method per way of folding the neighbours.
# ---------------------------------------------------------------------------

_BASE_VALUE_NET: type | None = None


def value_net_base() -> type:
    """The shared value network, built on the first call that needs it.

    Ego, neighbours, mask and candidate actions in; one value per candidate
    out. The per-neighbour encoder is shared over the neighbours and does not
    see the candidate action, so it is computed once per state rather than once
    per (state, action): the action joins the pooled neighbour vector and the
    ego features at the head, where the value is read.

    Every method's policy is this class with :meth:`pool` replaced - CADRL's
    LSTM over the neighbours and SARL's attention over them are exactly that -
    and a method whose policy is not this shape at all, as the template's is
    not, is free to build its own. Building the class here rather than at module
    import is what keeps torch out of an interpreter that only reads the
    catalogue.
    """
    global _BASE_VALUE_NET
    if _BASE_VALUE_NET is None:
        import torch
        import torch.nn as nn

        class ValueNet(nn.Module):
            """Ego, neighbours, mask and candidate actions in; one value out.

            A subclass replaces :meth:`pool` only: everything else - the input
            contract, the action table and the head - is the same for every
            method, which is what makes two encoders comparable.
            """

            def __init__(
                self,
                *,
                ego_features: int,
                neighbour_features: int,
                actions: int = 0,
                action_features: int = 2,
                hidden: int = 64,
            ):
                super().__init__()
                if int(ego_features) <= 0 or int(neighbour_features) <= 0:
                    raise ValueError("ego_features and neighbour_features must be positive")
                self.ego_features = int(ego_features)
                self.neighbour_features = int(neighbour_features)
                self.action_features = int(action_features)
                self.actions = int(actions)
                self.hidden = int(hidden)

                self.encoder = nn.Sequential(
                    nn.Linear(self.neighbour_features, self.hidden),
                    nn.ReLU(),
                    nn.Linear(self.hidden, self.hidden),
                    nn.ReLU(),
                )
                self.head = nn.Sequential(
                    nn.Linear(self.ego_features + self.hidden + self.action_features, self.hidden),
                    nn.ReLU(),
                    nn.Linear(self.hidden, self.hidden),
                    nn.ReLU(),
                    nn.Linear(self.hidden, 1),
                )
                # The candidate commands travel with the weights, so a checkpoint
                # carries the action set it was trained on and not just the shape.
                self.register_buffer("action_table", torch.zeros(0, self.action_features))

            def set_action_table(self, table: Any) -> "ValueNet":
                """Fix the candidate commands, as an ``(actions, features)`` array."""
                values = torch.as_tensor(np.asarray(table, dtype=np.float32))
                if values.ndim != 2 or values.shape[1] != self.action_features:
                    raise ValueError(
                        f"an action table is (actions, {self.action_features}), "
                        f"got shape {tuple(values.shape)}"
                    )
                if self.actions and self.actions != values.shape[0]:
                    raise ValueError(
                        f"the table holds {values.shape[0]} actions but the network was built "
                        f"for {self.actions}"
                    )
                self.action_table = values
                self.actions = int(values.shape[0])
                return self

            # -- the input contract ----------------------------------------

            def _tensor(self, value: Any):
                """``value`` as a float tensor on the module's own device."""
                if not torch.is_tensor(value):
                    value = torch.as_tensor(np.asarray(value))
                reference = next(self.parameters())
                return value.to(device=reference.device, dtype=reference.dtype)

            def _candidates(self, actions: Any):
                """The candidate commands, from the argument or the action table."""
                if actions is None:
                    if self.action_table.numel() == 0:
                        raise ValueError(
                            "no candidate actions: pass `actions` or call set_action_table first"
                        )
                    return self.action_table
                table = self._tensor(actions)
                if table.dim() != 2 or table.shape[1] != self.action_features:
                    raise ValueError(
                        f"candidate actions are (actions, {self.action_features}), "
                        f"got shape {tuple(table.shape)}"
                    )
                return table

            def forward(self, ego: Any, neighbours: Any, mask: Any, actions: Any = None):
                """``(batch, actions)`` values for a batch of states.

                ``neighbours`` is ``(batch, count, features)`` and ``mask`` is
                ``(batch, count)``; a row whose mask is false is padding and is
                ignored everywhere, whatever it holds. A false entry is zeroed
                before the encoder - rather than after it - so a padded row of
                infinities cannot turn into a ``nan`` the pooling then spreads.
                """
                ego = self._tensor(ego)
                neighbours = self._tensor(neighbours)
                if ego.dim() != 2:
                    raise ValueError(f"ego is (batch, features), got shape {tuple(ego.shape)}")
                if neighbours.dim() != 3:
                    raise ValueError(
                        f"neighbours are (batch, count, features), got shape "
                        f"{tuple(neighbours.shape)}"
                    )
                batch, count = neighbours.shape[0], neighbours.shape[1]
                if ego.shape[0] != batch or ego.shape[1] != self.ego_features:
                    raise ValueError(
                        f"ego is ({batch}, {self.ego_features}) to match {batch} neighbour row(s)"
                    )
                if neighbours.shape[2] != self.neighbour_features:
                    raise ValueError(
                        f"neighbour features are {self.neighbour_features}, got "
                        f"{neighbours.shape[2]}"
                    )
                keep = self._tensor(mask).bool()
                if tuple(keep.shape) != (batch, count):
                    raise ValueError(
                        f"the mask is ({batch}, {count}), got shape {tuple(keep.shape)}"
                    )

                safe = torch.where(keep.unsqueeze(-1), neighbours, torch.zeros_like(neighbours))
                pooled = self.pool(self.encoder(safe), keep)

                candidates = self._candidates(actions).to(device=ego.device, dtype=ego.dtype)
                state = torch.cat([ego, pooled], dim=1).unsqueeze(1)
                state = state.expand(batch, candidates.shape[0], state.shape[-1])
                candidates = candidates.unsqueeze(0).expand(batch, -1, -1)
                return self.head(torch.cat([state, candidates], dim=2)).squeeze(-1)

            def pool(self, encoded, keep):
                """Fold ``(batch, count, hidden)`` neighbours into ``(batch, hidden)``."""
                raise NotImplementedError

        _BASE_VALUE_NET = ValueNet
    return _BASE_VALUE_NET


# ---------------------------------------------------------------------------
# The learners: the plain DQN the template inherits, and the checkpoint format
# and running normaliser every method's learner shares. The value learner CADRL
# and SARL actually use - V(s) with a one-step lookahead - lives in
# :mod:`robotsnap.models.lookahead`, beside the two of them.
# ---------------------------------------------------------------------------

#: The tag a checkpoint carries, so a reader knows at once what it is holding.
FORMAT = "robotsnap-social-agent-1"


class _Moments:
    """A running mean and variance of one vector, in the style of ``robotsnap.rl.policy``.

    Welford's recurrence, so a value never needs the history to be updated and
    a long run does not lose precision the way a running sum of squares does.
    The count starts just above zero so the first observation does not divide
    by it.
    """

    def __init__(self, size: int):
        self.size = int(size)
        self.count = 1e-4
        self.mean = np.zeros(self.size, dtype=np.float64)
        self.square = np.zeros(self.size, dtype=np.float64)

    def update(self, value: Any) -> None:
        sample = np.asarray(value, dtype=np.float64).reshape(self.size)
        self.count += 1.0
        delta = sample - self.mean
        self.mean += delta / self.count
        self.square += delta * (sample - self.mean)

    def apply(self, value: Any) -> np.ndarray:
        """Normalise ``value`` with the statistics collected so far."""
        sample = np.asarray(value, dtype=np.float64)
        scale = np.sqrt(np.maximum(self.square / self.count, 1e-8))
        return ((sample - self.mean) / scale).astype(np.float32)

    def state(self) -> dict[str, Any]:
        return {
            "mean": [float(entry) for entry in self.mean],
            "square": [float(entry) for entry in self.square],
            "count": float(self.count),
        }

    @classmethod
    def from_state(cls, document: Mapping[str, Any]) -> "_Moments":
        moments = cls(len(document["mean"]))
        moments.mean = np.asarray(document["mean"], dtype=np.float64)
        moments.square = np.asarray(document["square"], dtype=np.float64)
        moments.count = float(document["count"])
        return moments


class _RunningNorm:
    """The two sets of statistics a structured social observation is normalised by.

    The ego vector moves once per step; the neighbours move once per *valid*
    row, so a padded row never drags the mean towards its zeros. A checkpoint
    that did not carry these would be replayed at a different scale than the
    one the network learned on, which is the failure ``robotsnap.rl.policy`` guards
    against for its flat observation.
    """

    def __init__(self, ego_size: int, neighbour_size: int):
        self.ego = _Moments(ego_size)
        self.neighbours = _Moments(neighbour_size)

    def observe(self, ego: Any, neighbours: Any, mask: Any) -> None:
        self.ego.update(ego)
        for row, valid in zip(np.asarray(neighbours), np.asarray(mask).astype(bool)):
            if valid:
                self.neighbours.update(row)

    def look(self, ego: Any, neighbours: Any) -> tuple[np.ndarray, np.ndarray]:
        """Normalise a batch, without moving the statistics."""
        return (
            self.ego.apply(ego).reshape(np.shape(ego)),
            self.neighbours.apply(neighbours).reshape(np.shape(neighbours)),
        )

    def state(self) -> dict[str, Any]:
        return {"ego": self.ego.state(), "neighbours": self.neighbours.state()}

    @classmethod
    def from_state(cls, document: Mapping[str, Any]) -> "_RunningNorm":
        norm = cls(
            len(document["ego"]["mean"]), len(document["neighbours"]["mean"])
        )
        norm.ego = _Moments.from_state(document["ego"])
        norm.neighbours = _Moments.from_state(document["neighbours"])
        return norm


class _DQNAgent:
    """A replay-buffer DQN over the observation a method publishes.

    A value network that predicts the cost-to-go of a ``(state, action)`` pair
    is only half of an agent. The other half is the loop around it: a replay
    buffer so one step of experience is used many times, a target network so the
    regression target does not move with the weights it is grading, and an
    epsilon that decays so the agent explores widely at first and then settles.
    One piece of bookkeeping the papers add is here too - a share of the
    exploration draws is spent on the do-nothing action, because a robot that
    never tries stopping learns late that stopping is often the safest thing it
    can do.

    What goes into the network is the ``(ego, neighbours, mask)`` triple
    :meth:`_pack` returns; the shared task packs its structured observation, and
    a method whose observation is a flat vector overrides :meth:`_pack` and
    leaves the rest of the DQN alone. The candidate commands travel in the
    network, so :meth:`act` only has to pick an index and :meth:`update` only has
    to regress one.

    The interface is deliberately framework-light: an agent reads a mapping of
    numpy arrays, keeps numpy in its buffer, and only the forward and backward
    passes of :meth:`update` touch torch. Nothing here imports torch at module
    import, so a caller can build an agent, size its buffer and read its epsilon
    without a framework loaded.

    Save and load follow :mod:`robotsnap.rl.policy`, and for the same reason: a
    checkpoint that carried the weights alone could not be replayed, because it
    would not say how wide the network is, which commands its discrete action set
    held, or what statistics the observations were normalised by. All three
    travel with the weights here.
    """

    #: The name the algorithm registers under, and the one a checkpoint carries.
    name = "dqn"

    def __init__(
        self,
        *,
        ego_features: int = EGO_SIZE + GOAL_SIZE,
        neighbour_features: int = NEIGHBOUR_SIZE,
        actions: int,
        action_features: int = 2,
        action_table: Any = None,
        hidden: int = 64,
        learning_rate: float = 1e-3,
        gamma: float = 0.95,
        batch_size: int = 32,
        buffer_size: int = 20000,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        near_stop_probability: float = 0.1,
        target_update: int = 200,
        normalise: bool = True,
        seed: int | None = None,
        device: str = "cpu",
    ):
        import torch

        self.ego_features = int(ego_features)
        self.neighbour_features = int(neighbour_features)
        self.actions = int(actions)
        self.action_features = int(action_features)
        self.hidden = int(hidden)
        self.learning_rate = float(learning_rate)
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.buffer_size = int(buffer_size)
        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay = float(epsilon_decay)
        self.near_stop_probability = float(near_stop_probability)
        self.target_update = max(1, int(target_update))
        self.normalise = bool(normalise)
        self.seed = seed
        self.device = str(device)
        if self.actions <= 0:
            raise ValueError("an agent needs at least one action")

        if action_table is None:
            # A table that is not the commands of a real action set still has to
            # say which action is which, or every candidate would look the same
            # to the network; an index table does that for a synthetic model.
            table = np.zeros((self.actions, self.action_features), dtype=np.float32)
            table[:, 0] = np.arange(self.actions)
        else:
            table = np.asarray(action_table, dtype=np.float32)
        table = table.reshape(self.actions, self.action_features)

        self.online = self._build_model().to(self.device)
        self.online.set_action_table(table)
        self.target = self._build_model().to(self.device)
        self.target.set_action_table(table)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimiser = torch.optim.Adam(self.online.parameters(), lr=self.learning_rate)

        self.buffer: deque[tuple[Any, ...]] = deque(maxlen=self.buffer_size)
        self.epsilon = self.epsilon_start
        self.steps = 0
        self.rng = np.random.default_rng(seed)
        self.norm = (
            _RunningNorm(self.ego_features, self.neighbour_features)
            if self.normalise
            else None
        )

    # -- the model and the packing, supplied by the subclass -----------------

    def _build_model(self):
        """A fresh value network for this agent's shape."""
        raise NotImplementedError

    def _pack(self, observation: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The ``(ego, neighbours, mask)`` triple the network takes.

        The shared task's observation is the structured dictionary; a method
        whose observation is a flat vector overrides this and returns the same
        three arrays, so the buffer, the update and the checkpoint never have to
        know which of the two they are carrying.
        """
        return pack_observation(observation)

    def set_action_table(self, table: Any) -> None:
        """Replace the candidate commands of both networks."""
        self.online.set_action_table(table)
        self.target.set_action_table(table)

    # -- acting and learning ----------------------------------------------

    def act(self, observation: Any, greedy: bool = False) -> int:
        """The action to take from one observation.

        Greedy plays the best action the network knows; not greedy draws an
        exploration move with probability ``epsilon`` - the do-nothing action
        with probability ``near_stop_probability`` of its own, a uniform index
        otherwise - and plays the best one any other time.
        """
        if not greedy and float(self.rng.random()) < self.epsilon:
            if float(self.rng.random()) < self.near_stop_probability:
                return 0
            return int(self.rng.integers(self.actions))
        ego, neighbours, mask = self._prepare(observation)
        values = self._values(ego[None], neighbours[None], mask[None])
        return int(values.argmax(dim=1).item())

    def _values(self, ego: Any, neighbours: Any, mask: Any):
        """The network's values for a batch, without building a graph."""
        import torch

        with torch.no_grad():
            return self.online(ego, neighbours, mask)

    def observe(
        self,
        observation: Any,
        action: int,
        reward: float,
        next_observation: Any,
        done: bool,
    ) -> None:
        """Record one transition and take the next epsilon step.

        The arrays are copied into the buffer rather than referenced, so a
        caller that reuses its observation buffers between steps cannot corrupt
        what the agent learned from.
        """
        current = [entry.copy() for entry in self._pack(observation)]
        following = [entry.copy() for entry in self._pack(next_observation)]
        transition = (
            *current,
            int(action),
            float(reward),
            *following,
            bool(done),
        )
        self.buffer.append(transition)
        if self.norm is not None:
            self.norm.observe(current[0], current[1], current[2])
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def learn(self) -> float | None:
        """One gradient step on a sampled batch, or ``None`` while the buffer fills."""
        if len(self.buffer) < self.batch_size:
            return None
        indices = self.rng.integers(0, len(self.buffer), size=self.batch_size)
        return self.update([self.buffer[int(index)] for index in indices])

    def update(self, batch: Sequence[tuple[Any, ...]]) -> float:
        """One gradient step on the transitions ``batch``.

        Each transition is the tuple :meth:`observe` writes:
        ``(ego, neighbours, mask, action, reward, next_ego, next_neighbours,
        next_mask, done)``. The target network is refreshed every
        ``target_update`` steps, which is what keeps the regression target from
        chasing the weights it is grading.
        """
        import torch
        import torch.nn.functional as functional

        transitions = list(batch)
        if not transitions:
            raise ValueError("an update needs at least one transition")

        def stacked(index: int) -> np.ndarray:
            return np.stack([np.asarray(row[index]) for row in transitions])

        ego, neighbours, mask = stacked(0), stacked(1), stacked(2)
        actions = np.asarray([row[3] for row in transitions], dtype=np.int64)
        rewards = np.asarray([row[4] for row in transitions], dtype=np.float32)
        next_ego, next_neighbours, next_mask = stacked(5), stacked(6), stacked(7)
        dones = np.asarray([row[8] for row in transitions], dtype=np.float32)
        if self.norm is not None:
            ego, neighbours = self.norm.look(ego, neighbours)
            next_ego, next_neighbours = self.norm.look(next_ego, next_neighbours)

        device = next(self.online.parameters()).device

        def to_tensor(value: Any, dtype=None):
            return torch.as_tensor(value, dtype=dtype, device=device)

        ego_t = to_tensor(ego, torch.float32)
        neighbours_t = to_tensor(neighbours, torch.float32)
        mask_t = to_tensor(mask, torch.bool)
        next_trio = (
            to_tensor(next_ego, torch.float32),
            to_tensor(next_neighbours, torch.float32),
            to_tensor(next_mask, torch.bool),
        )
        actions_t = to_tensor(actions, torch.int64)
        rewards_t = to_tensor(rewards, torch.float32)
        dones_t = to_tensor(dones, torch.float32)

        self.online.train()
        values = self.online(ego_t, neighbours_t, mask_t)
        chosen = values.gather(1, actions_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            future = self.target(*next_trio).max(dim=1).values
            target = rewards_t + self.gamma * (1.0 - dones_t) * future
        loss = functional.mse_loss(chosen, target)

        self.optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimiser.step()

        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())

    # -- carrying a trained agent ------------------------------------------

    def config(self) -> dict[str, Any]:
        """The constructor arguments this agent was built with."""
        return {
            "ego_features": self.ego_features,
            "neighbour_features": self.neighbour_features,
            "actions": self.actions,
            "action_features": self.action_features,
            "action_table": self.online.action_table.detach().cpu().numpy().tolist(),
            "hidden": self.hidden,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "batch_size": self.batch_size,
            "buffer_size": self.buffer_size,
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay": self.epsilon_decay,
            "near_stop_probability": self.near_stop_probability,
            "target_update": self.target_update,
            "normalise": self.normalise,
            "seed": self.seed,
            "device": self.device,
        }

    def save(self, path: str | Path) -> str:
        """Write the weights, the configuration and the statistics to ``path``."""
        import torch

        document = {
            "format": FORMAT,
            "algorithm": self.name,
            "config": self.config(),
            "state_dict": self.online.state_dict(),
            "epsilon": float(self.epsilon),
            "steps": int(self.steps),
            "normaliser": None if self.norm is None else self.norm.state(),
        }
        torch.save(document, path)
        return str(path)

    @classmethod
    def load(cls, path: str | Path, *, device: str | None = None) -> "_DQNAgent":
        """Rebuild an agent of this class from the checkpoint at ``path``."""
        import torch

        document = torch.load(Path(path), map_location="cpu")
        if not isinstance(document, Mapping) or document.get("format") != FORMAT:
            raise ValueError(f"{path} is not a {FORMAT} checkpoint")
        if document.get("algorithm") != cls.name:
            raise ValueError(
                f"{path} holds a {document.get('algorithm')!r} agent, not a {cls.name!r} one"
            )
        return cls._from_document(document, device=device)

    @classmethod
    def _from_document(cls, document: Mapping[str, Any], *, device: str | None = None):
        config = dict(document["config"])
        if device is not None:
            config["device"] = str(device)
        agent = cls(**config)
        state = document["state_dict"]
        agent.online.load_state_dict(state)
        agent.target.load_state_dict(state)
        agent.online.eval()
        agent.target.eval()
        agent.epsilon = float(document.get("epsilon", agent.epsilon))
        agent.steps = int(document.get("steps", 0))
        if agent.norm is not None and document.get("normaliser"):
            agent.norm = _RunningNorm.from_state(document["normaliser"])
        return agent

    def _prepare(self, observation: Any):
        """``(ego, neighbours, mask)`` on the way into the network."""
        ego, neighbours, mask = self._pack(observation)
        if self.norm is not None:
            ego, neighbours = self.norm.look(ego, neighbours)
        return ego, neighbours, mask


# ---------------------------------------------------------------------------
# The loop: one trainer small enough to read, used twice.
# ---------------------------------------------------------------------------

#: The columns both loops fill in, in the order a log line prints them.
HISTORY_KEYS = ("episode", "reward", "steps", "epsilon", "loss", "outcome")


def _empty_history() -> dict[str, list]:
    return {key: [] for key in HISTORY_KEYS}


def _outcome(info: dict[str, Any]) -> str:
    """Why the episode stopped, from the step info dictionary."""
    if info.get("goal_reached"):
        return "goal"
    if info.get("collision"):
        return "collision"
    if info.get("out_of_bounds"):
        return "out_of_bounds"
    return "timeout"


def _run_episode(
    env,
    agent: _DQNAgent,
    *,
    greedy: bool,
    learn: bool,
    seed: int | None,
    max_steps: int | None,
    render: bool = False,
) -> tuple[float, int, list[float], str]:
    """One episode; ``(total reward, steps, losses, outcome)``.

    ``greedy`` and ``learn`` are separate on purpose: an evaluation episode is
    greedy and silent, while a training episode explores and writes to the
    replay buffer. The two travel through one loop so a change to the loop -
    where the freshness flag is read, what counts as a step - cannot make the
    evaluation of a policy differ from the training of it. ``render`` is that
    same kind of framing: it draws the run's window one frame per step, and is
    false for a run that asked for none.
    """
    observation, _ = env.reset(seed=seed)
    total = 0.0
    steps = 0
    losses: list[float] = []
    info: dict[str, Any] = {}
    terminated = truncated = False
    while not (terminated or truncated) and (max_steps is None or steps < max_steps):
        action = agent.act(observation, greedy=greedy)
        next_observation, reward, terminated, truncated, info = env.step(action)
        if learn:
            agent.observe(
                observation,
                action,
                reward,
                next_observation,
                bool(terminated or truncated),
            )
            loss = agent.learn()
            if loss is not None:
                losses.append(float(loss))
        observation = next_observation
        total += float(reward)
        steps += 1
        if render:
            env.render()
    return total, steps, losses, _outcome(info)


def train(
    env,
    agent: _DQNAgent,
    *,
    episodes: int = 100,
    max_steps: int | None = None,
    seed: int | None = None,
    explore: bool = True,
    render: bool = False,
    log: Callable[[str], Any] | None = print,
) -> dict[str, list]:
    """Run ``episodes`` training episodes and return what each one did.

    ``seed`` is spread over the episodes - the n-th episode resets with
    ``seed + n`` - so two runs of the same trainer with the same seed see the
    same sequence of episode starts, which is what makes a comparison between
    two agents reproducible.
    """
    history = _empty_history()
    for episode in range(int(episodes)):
        total, steps, losses, outcome = _run_episode(
            env,
            agent,
            greedy=not explore,
            learn=True,
            seed=None if seed is None else int(seed) + episode,
            max_steps=max_steps,
            render=render,
        )
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        history["episode"].append(episode)
        history["reward"].append(total)
        history["steps"].append(steps)
        history["epsilon"].append(float(agent.epsilon))
        history["loss"].append(mean_loss)
        history["outcome"].append(outcome)
        if log is not None:
            log(
                f"episode {episode + 1:4d} reward {total:8.2f} steps {steps:4d} "
                f"epsilon {float(agent.epsilon):.3f} loss {mean_loss:8.4f} outcome {outcome}"
            )
    return history


def play(
    path: str | Path,
    env,
    *,
    episodes: int = 1,
    max_steps: int | None = None,
    seed: int | None = None,
    device: str | None = None,
    render: bool = False,
    log: Callable[[str], Any] | None = print,
) -> dict[str, list]:
    """Load the agent saved at ``path`` and run greedy episodes with it.

    Nothing is written back to the agent - no replay, no gradient step, no
    epsilon decay - so playing a checkpoint never changes the policy it is
    playing, and the same file gives the same behaviour next time. The loader
    is read from :mod:`robotsnap.models` here rather than at import, because the
    registry is what knows which method a checkpoint names.
    """
    from robotsnap.models import load_agent

    agent = load_agent(path, device=device)
    history = _empty_history()
    for episode in range(int(episodes)):
        total, steps, _, outcome = _run_episode(
            env,
            agent,
            greedy=True,
            learn=False,
            seed=None if seed is None else int(seed) + episode,
            max_steps=max_steps,
            render=render,
        )
        history["episode"].append(episode)
        history["reward"].append(total)
        history["steps"].append(steps)
        history["epsilon"].append(float(agent.epsilon))
        history["loss"].append(float("nan"))
        history["outcome"].append(outcome)
        if log is not None:
            log(
                f"episode {episode + 1:4d} reward {total:8.2f} steps {steps:4d} outcome {outcome}"
            )
    return history
