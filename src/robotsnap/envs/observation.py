"""Composable observation components for :class:`~robotsnap.envs.base.RobotSNAPEnv`.

The default observation of the environment is one flat vector made of four
parts - the robot, the goal, the neighbours and the lidar - and a task that
wants another view of the same world had to override :meth:`RobotSNAPEnv.observation`
and rebuild the whole vector. This module turns that vector into a set of named
*parts* a caller picks and orders with the ``observations`` argument::

    RobotSNAPEnv(observations=("pose", "goal", "lidar_stats"))
    RobotSNAPEnv(observations="goal,agents", observation_params={"agents": {"max": 4}})
    RobotSNAPEnv(observations={"lidar": {"bins": 16}, "status": {}})
    RobotSNAPEnv(observations="none")          # an empty observation
    RobotSNAPEnv(observations=("pose", "goal"), observation_structure="dict")

A part declares its size, its bounds and how to build it, so the observation
space is derived from the same source as the value it must contain and cannot
drift from it. The four historical methods - ``robot_features``,
``goal_features``, ``agent_features``, ``lidar_features`` - stay the building
blocks of the corresponding parts: a subclass that overrides one of them keeps
working, whether the part was asked for by name or not.

A project with its own feature registers it once and asks for it by name::

    from robotsnap.envs.observation import ObservationPart, register_observation_part

    register_observation_part(
        "battery",
        ObservationPart(
            name="battery",
            size=1,
            low=np.zeros(1, dtype=np.float32),
            high=np.ones(1, dtype=np.float32),
            build=lambda context: np.array([context.world.linear_velocity / 10.0]),
            doc="a stand-in for a state the simulator will publish",
        ),
    )

Every part is built from an :class:`ObservationContext`, which carries the
environment, the :class:`~robotsnap.envs.world.World` read for that step, the
task state and the parameters the part was resolved with. Registering a custom
part never touches the environment class.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from robotsnap.envs.world import World

if TYPE_CHECKING:  # pragma: no cover - imported only for the annotations
    from robotsnap.envs.base import RobotSNAPEnv, TaskState

__all__ = [
    "DEFAULT_OBSERVATIONS",
    "MAX_NEIGHBOUR_SPEED",
    "ObservationContext",
    "ObservationPart",
    "register_observation_part",
    "registered_observation_names",
    "describe_observation_parts",
    "resolve_observation_parts",
    "env_observation_parts",
    "normalise_observation_names",
    "spec_parameters",
]

#: The parts the default observation is made of, in the order it stacks them.
DEFAULT_OBSERVATIONS: tuple[str, ...] = ("robot", "goal", "agents", "lidar")
#: Bound the observation space uses for the velocity of a neighbour or a human.
MAX_NEIGHBOUR_SPEED = 5.0
#: Names ``observations`` accepts for "say nothing about the world".
_EMPTY_NAMES = frozenset({"", "none", "empty", "nothing"})


@dataclass(frozen=True)
class ObservationContext:
    """Everything one observation part is built from.

    ``params`` are the parameters the part was resolved with, so a part that
    wants them at build time - rather than capturing them in its factory - reads
    them here. Built-in parts capture theirs, which keeps them plain functions.
    """

    env: "RobotSNAPEnv"
    world: World
    task: "TaskState"
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ObservationPart:
    """One named block of an observation, with the bounds of its own values.

    ``size`` is the number of scalars the part contributes. ``low`` and ``high``
    are its bounds, as arrays of ``size`` - a scalar is broadcast, which is how
    most parts declare theirs. ``build`` receives an :class:`ObservationContext`
    and returns ``size`` values; the environment flattens a part that returns a
    matrix, so a part can be written in whatever shape reads best.
    """

    name: str
    size: int
    low: np.ndarray
    high: np.ndarray
    build: Callable[[ObservationContext], np.ndarray]
    doc: str = ""

    def __post_init__(self) -> None:
        if self.size < 0:
            raise ValueError(f"observation part {self.name!r} has a negative size")
        for label, values in (("low", self.low), ("high", self.high)):
            array = np.asarray(values, dtype=np.float32).reshape(-1)
            if array.size != self.size:
                raise ValueError(
                    f"observation part {self.name!r} has size {self.size} but "
                    f"declares {array.size} {label} bound(s)"
                )
            object.__setattr__(self, label, array)
        if np.any(np.asarray(self.low) > np.asarray(self.high)):
            raise ValueError(f"observation part {self.name!r} has a low bound above its high")

    def value(self, context: ObservationContext) -> np.ndarray:
        """The part built for ``context``, as ``size`` float32 values.

        The size is checked here rather than trusted: a part that returns the
        wrong number of values would otherwise shift every part after it, which
        is the kind of bug that reads as "the policy learned nothing".
        """
        values = np.asarray(self.build(context), dtype=np.float32).reshape(-1)
        if values.size != self.size:
            raise ValueError(
                f"observation part {self.name!r} returned {values.size} value(s), "
                f"expected {self.size}"
            )
        return values


#: A factory builds a part for an environment and one part's parameters.
ObservationFactory = Callable[["RobotSNAPEnv", Mapping[str, Any]], ObservationPart]


def _bounds(low: Any, high: Any, size: int) -> tuple[np.ndarray, np.ndarray]:
    """``low``/``high`` broadcast to ``size`` float32 values, checked for order."""
    return (
        np.broadcast_to(np.asarray(low, dtype=np.float32), (size,)).astype(np.float32),
        np.broadcast_to(np.asarray(high, dtype=np.float32), (size,)).astype(np.float32),
    )


# -- the parts the package ships -------------------------------------------


def _robot_part(env: "RobotSNAPEnv", params: Mapping[str, Any]) -> ObservationPart:
    """Pose and body velocity together, the historical first block."""
    low, high = _bounds(
        [-np.inf, -np.inf, -1.0, -1.0, -env.max_linear, -env.max_angular],
        [np.inf, np.inf, 1.0, 1.0, env.max_linear, env.max_angular],
        6,
    )
    return ObservationPart(
        name="robot",
        size=6,
        low=low,
        high=high,
        build=lambda context: context.env.robot_features(context.world),
        doc="pose and body velocity: x, y, cos yaw, sin yaw, v, w",
    )


def _pose_part(env: "RobotSNAPEnv", params: Mapping[str, Any]) -> ObservationPart:
    """Where the robot is, without how fast it goes."""
    low, high = _bounds([-np.inf, -np.inf, -1.0, -1.0], [np.inf, np.inf, 1.0, 1.0], 4)
    return ObservationPart(
        name="pose",
        size=4,
        low=low,
        high=high,
        build=lambda context: context.env.robot_features(context.world)[:4],
        doc="pose alone: x, y, cos yaw, sin yaw",
    )


def _velocity_part(env: "RobotSNAPEnv", params: Mapping[str, Any]) -> ObservationPart:
    """How fast the robot goes, without where it is."""
    low, high = _bounds(
        [-env.max_linear, -env.max_angular], [env.max_linear, env.max_angular], 2
    )
    return ObservationPart(
        name="velocity",
        size=2,
        low=low,
        high=high,
        build=lambda context: context.env.robot_features(context.world)[4:],
        doc="body velocity alone: v, w",
    )


def _goal_part(env: "RobotSNAPEnv", params: Mapping[str, Any]) -> ObservationPart:
    """The goal in the robot frame, the historical second block."""
    low, high = _bounds(
        [-np.inf, -np.inf, 0.0, -1.0, -1.0], [np.inf, np.inf, np.inf, 1.0, 1.0], 5
    )
    return ObservationPart(
        name="goal",
        size=5,
        low=low,
        high=high,
        build=lambda context: context.env.goal_features(context.world, context.task),
        doc="goal in the robot frame: dx, dy, distance, cos bearing, sin bearing",
    )


def _goal_distance_part(env: "RobotSNAPEnv", params: Mapping[str, Any]) -> ObservationPart:
    """The single scalar a goal-reaching reward is already built on."""

    def build(context: ObservationContext) -> np.ndarray:
        distance = context.task.distance_to_goal
        return np.array([-1.0 if distance is None else distance], dtype=np.float32)

    low, high = _bounds([-1.0], [np.inf], 1)
    return ObservationPart(
        name="goal_distance",
        size=1,
        low=low,
        high=high,
        build=build,
        doc="distance to the goal, or -1 when the session publishes no goal",
    )


def _agents_part(env: "RobotSNAPEnv", params: Mapping[str, Any]) -> ObservationPart:
    """The neighbours the robot's own detector reports, nearest first."""
    if env.max_agents:
        low, high = _bounds(
            np.tile([-np.inf, -np.inf, -MAX_NEIGHBOUR_SPEED, -MAX_NEIGHBOUR_SPEED, 0.0], env.max_agents),
            np.tile([np.inf, np.inf, MAX_NEIGHBOUR_SPEED, MAX_NEIGHBOUR_SPEED, 1.0], env.max_agents),
            env.max_agents * 5,
        )
    else:
        low, high = _bounds([], [], 0)
    return ObservationPart(
        name="agents",
        size=env.max_agents * 5,
        low=low,
        high=high,
        build=lambda context: context.env.agent_features(context.world),
        doc=(
            "the nearest agents, one row each: dx, dy, vx, vy, visible"
            " (the count is the `max` parameter, `max_agents` by default)"
        ),
    )


def _lidar_part(env: "RobotSNAPEnv", params: Mapping[str, Any]) -> ObservationPart:
    """The sweep sampled into bins, the historical fourth block."""
    low, high = _bounds(np.zeros(env.lidar_bins), np.ones(env.lidar_bins), env.lidar_bins)
    return ObservationPart(
        name="lidar",
        size=env.lidar_bins,
        low=low,
        high=high,
        build=lambda context: context.env.lidar_features(context.world),
        doc="the scan resampled into `bins` ranges normalised to [0, 1] (`lidar_bins` by default)",
    )


def _lidar_stats_part(env: "RobotSNAPEnv", params: Mapping[str, Any]) -> ObservationPart:
    """Three numbers standing in for the whole sweep."""

    def build(context: ObservationContext) -> np.ndarray:
        scan = context.world.scan
        if scan is None:
            return np.array([1.0, 1.0, 1.0], dtype=np.float32)
        span = float(scan.range_max)
        ranges = np.asarray(scan.ranges, dtype=np.float64)
        finite = ranges[np.isfinite(ranges)]
        if span <= 0.0 or finite.size == 0:
            return np.array([1.0, 1.0, 1.0], dtype=np.float32)
        # A beam that came back at the sensor's reach is a beam that saw nothing.
        free = float(np.count_nonzero(~np.isfinite(ranges) | (ranges >= span))) / float(ranges.size)
        return np.array(
            [
                min(scan.min_range() / span, 1.0),
                min(float(finite.mean()) / span, 1.0),
                free,
            ],
            dtype=np.float32,
        )

    low, high = _bounds([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 3)
    return ObservationPart(
        name="lidar_stats",
        size=3,
        low=low,
        high=high,
        build=build,
        doc="the sweep summarised: closest range, mean range, fraction of free beams, all in [0, 1]",
    )


def _nearest_agent_part(env: "RobotSNAPEnv", params: Mapping[str, Any]) -> ObservationPart:
    """The single closest neighbour, as a fixed-size row."""

    def build(context: ObservationContext) -> np.ndarray:
        agent = context.world.nearest_agent()
        if agent is None:
            return np.zeros(5, dtype=np.float32)
        return np.array(
            [
                agent.distance,
                math.cos(agent.bearing),
                math.sin(agent.bearing),
                min(agent.speed, MAX_NEIGHBOUR_SPEED),
                1.0,
            ],
            dtype=np.float32,
        )

    low, high = _bounds([0.0, -1.0, -1.0, 0.0, 0.0], [np.inf, 1.0, 1.0, MAX_NEIGHBOUR_SPEED, 1.0], 5)
    return ObservationPart(
        name="nearest_agent",
        size=5,
        low=low,
        high=high,
        build=build,
        doc="the closest detected agent: distance, cos bearing, sin bearing, speed, present",
    )


def _humans_part(env: "RobotSNAPEnv", params: Mapping[str, Any]) -> ObservationPart:
    """The simulated humans of the snapshot, in the robot frame.

    This is the ground truth the session publishes about its own crowd, not what
    the robot's detector reports - ``agents`` is that one. The two answer
    different questions ("what is there" versus "what did the robot see"), and a
    benchmark that wants to separate perception from control takes both.
    """
    count = env.max_humans

    def build(context: ObservationContext) -> np.ndarray:
        rows = np.zeros((count, 5), dtype=np.float32)
        if count == 0:
            return rows.reshape(0)
        pose = context.world.pose
        if pose is None:
            return rows.reshape(-1)
        cos_yaw, sin_yaw = math.cos(pose.yaw), math.sin(pose.yaw)
        entries: list[tuple[float, float, float, float, float]] = []
        for human in context.world.humans:
            dx = human.x - pose.x
            dy = human.y - pose.y
            # World frame to robot frame: x forward, y left.
            rx = dx * cos_yaw + dy * sin_yaw
            ry = -dx * sin_yaw + dy * cos_yaw
            vx = human.vx * cos_yaw + human.vy * sin_yaw
            vy = -human.vx * sin_yaw + human.vy * cos_yaw
            entries.append((math.hypot(rx, ry), rx, ry, vx, vy))
        for index, (_, rx, ry, vx, vy) in enumerate(sorted(entries)[:count]):
            rows[index] = (rx, ry, vx, vy, 1.0)
        return rows.reshape(-1)

    if count:
        low, high = _bounds(
            np.tile([-np.inf, -np.inf, -MAX_NEIGHBOUR_SPEED, -MAX_NEIGHBOUR_SPEED, 0.0], count),
            np.tile([np.inf, np.inf, MAX_NEIGHBOUR_SPEED, MAX_NEIGHBOUR_SPEED, 1.0], count),
            count * 5,
        )
    else:
        low, high = _bounds([], [], 0)
    return ObservationPart(
        name="humans",
        size=count * 5,
        low=low,
        high=high,
        build=build,
        doc=(
            "the simulated humans, one row each, nearest first: dx, dy, vx, vy, present"
            " (the count is the `max` parameter, `max_humans` by default)"
        ),
    )


def _occupancy_part(env: "RobotSNAPEnv", params: Mapping[str, Any]) -> ObservationPart:
    """A square patch of the occupancy grid around the robot, as 0/1 values.

    The patch is world-aligned, not rotated with the robot: a row counts along
    ROS +y and a column along ROS +x, exactly the order of the grid it is cut
    from. A cell outside the published grid counts as occupied, since "the map
    ended here" is not something to plan through.
    """
    side = int(params.get("size", 21))
    if side <= 0:
        raise ValueError("the occupancy patch needs a positive size")
    if side % 2 == 0:  # centred on the robot, so an odd side
        side += 1
    threshold = int(params.get("threshold", env.occupancy_threshold))

    def build(context: ObservationContext) -> np.ndarray:
        patch = np.ones((side, side), dtype=np.float32)
        grid = context.world.map
        pose = context.world.pose
        if grid is None or pose is None:
            return patch.reshape(-1)
        resolution = float(grid.resolution) or 1.0
        column = int(math.floor((pose.x - grid.origin_x) / resolution))
        row = int(math.floor((pose.y - grid.origin_y) / resolution))
        half = side // 2
        columns = np.arange(column - half, column - half + side)
        rows = np.arange(row - half, row - half + side)
        inside_columns = (columns >= 0) & (columns < grid.width)
        inside_rows = (rows >= 0) & (rows < grid.height)
        if inside_columns.any() and inside_rows.any():
            cells = grid.cells[np.ix_(rows[inside_rows], columns[inside_columns])]
            patch[np.ix_(inside_rows, inside_columns)] = (cells >= threshold).astype(np.float32)
        return patch.reshape(-1)

    low, high = _bounds(np.zeros(side * side), np.ones(side * side), side * side)
    return ObservationPart(
        name="occupancy",
        size=side * side,
        low=low,
        high=high,
        build=build,
        doc="a `size`x`size` world-aligned patch of the grid around the robot, rows along +y",
    )


def _status_part(env: "RobotSNAPEnv", params: Mapping[str, Any]) -> ObservationPart:
    """The three flags the episode ends on."""

    def build(context: ObservationContext) -> np.ndarray:
        return np.array(
            [
                1.0 if context.task.goal_reached else 0.0,
                1.0 if context.task.collision else 0.0,
                1.0 if context.task.out_of_bounds else 0.0,
            ],
            dtype=np.float32,
        )

    low, high = _bounds([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 3)
    return ObservationPart(
        name="status",
        size=3,
        low=low,
        high=high,
        build=build,
        doc="the episode flags: goal reached, collision, out of bounds",
    )


def _time_part(env: "RobotSNAPEnv", params: Mapping[str, Any]) -> ObservationPart:
    """Where the episode stands in its own budget."""

    def build(context: ObservationContext) -> np.ndarray:
        budget = float(context.env.max_episode_seconds)
        elapsed = float(context.task.elapsed_seconds)
        if budget <= 0.0:
            return np.zeros(1, dtype=np.float32)
        return np.array([min(max(elapsed / budget, 0.0), 1.0)], dtype=np.float32)

    low, high = _bounds([0.0], [1.0], 1)
    return ObservationPart(
        name="time",
        size=1,
        low=low,
        high=high,
        build=build,
        doc="the fraction of the episode budget spent, in [0, 1]",
    )


_REGISTRY: dict[str, ObservationFactory] = {}


def register_observation_part(
    name: str,
    part: ObservationPart | ObservationFactory,
    *,
    replace: bool = False,
) -> None:
    """Make ``name`` usable in ``observations``, for this interpreter.

    ``part`` is either an :class:`ObservationPart` - used as it is, whatever
    parameters the spec carries - or a factory taking the environment and the
    part's parameters and returning the part to use. Registering over an
    existing name is refused unless ``replace`` says otherwise, so a collision
    between two projects is loud rather than silent.
    """
    key = str(name)
    if not key:
        raise ValueError("an observation part needs a name")
    if key in _REGISTRY and not replace:
        raise ValueError(f"observation part {key!r} is already registered")
    if isinstance(part, ObservationPart):
        if part.name != key:
            raise ValueError(
                f"registered name {key!r} does not match the part's own name {part.name!r}"
            )
        fixed = part
        _REGISTRY[key] = lambda env, params: fixed
    elif callable(part):
        _REGISTRY[key] = part
    else:  # pragma: no cover - a caller error the type checker already catches
        raise TypeError("an observation part must be an ObservationPart or a factory")


def registered_observation_names() -> tuple[str, ...]:
    """Every part that can be asked for, the built-ins first in their own order."""
    builtin = [name for name in DEFAULT_OBSERVATIONS if name in _REGISTRY]
    rest = sorted(name for name in _REGISTRY if name not in DEFAULT_OBSERVATIONS)
    return tuple(builtin + rest)


def describe_observation_parts() -> tuple[tuple[str, str], ...]:
    """``(name, doc)`` for every registered part, for a help screen."""
    described = []
    for name in registered_observation_names():
        part = _REGISTRY[name](_ProbeEnv(), {})
        described.append((name, part.doc or ""))
    return tuple(described)


class _ProbeEnv:
    """The smallest object a factory may read, to describe a part without an env.

    A factory only reads its configuration from the environment - the velocity
    limits, the number of bins, the number of neighbours - so describing one
    needs those defaults and nothing else. A custom factory that reaches for
    anything more is describing itself rather than the environment, and gets an
    ``AttributeError`` here rather than a wrong answer.
    """

    max_linear = 1.0
    max_angular = 1.0
    max_agents = 8
    max_humans = 8
    lidar_bins = 48
    occupancy_threshold = 50
    max_episode_seconds = 60.0


def _factory(name: str) -> ObservationFactory:
    try:
        return _REGISTRY[name]
    except KeyError:
        known = ", ".join(registered_observation_names())
        raise ValueError(f"unknown observation part {name!r}; known parts: {known}") from None


def normalise_observation_names(spec: Any) -> tuple[str, ...]:
    """The names ``spec`` asks for, without resolving anything.

    Accepts ``None`` (the default observation), a comma or space separated
    string, a sequence of names, or a mapping whose keys are the names. It is
    what a caller - a CLI, a config file - uses to show what was asked for, and
    it refuses a name nobody registered.
    """
    inline = {part.name for part in _inline_parts(spec)}
    names = tuple(name for name, _ in _explicit_spec(spec))
    for name in names:
        if name not in inline:
            _factory(name)
    return names


def _explicit_spec(spec: Any) -> list[tuple[str, Mapping[str, Any]]]:
    """``spec`` as an ordered list of ``(name, params)``, defaults filled in."""
    if spec is None:
        return [(name, {}) for name in DEFAULT_OBSERVATIONS]
    if isinstance(spec, ObservationPart):
        return [(spec.name, {})]
    if isinstance(spec, str):
        text = spec.strip()
        if text.lower() in _EMPTY_NAMES:
            return []
        chunks = text.replace(",", " ").split()
        return [(chunk, {}) for chunk in chunks]
    if isinstance(spec, Mapping):
        return [(str(name), dict(value or {})) for name, value in spec.items()]
    if isinstance(spec, Iterable):
        items: list[tuple[str, Mapping[str, Any]]] = []
        for entry in spec:
            if isinstance(entry, ObservationPart):
                items.append((entry.name, {}))
            elif isinstance(entry, str):
                items.append((entry, {}))
            elif isinstance(entry, tuple) and len(entry) == 2:
                name, params = entry
                items.append((str(name), dict(params or {})))
            else:
                raise TypeError(
                    "an observation spec holds part names, (name, params) pairs or "
                    f"ObservationPart instances, got {entry!r}"
                )
        return items
    raise TypeError(f"cannot read an observation spec from {spec!r}")


def resolve_observation_parts(
    env: "RobotSNAPEnv",
    spec: Any = None,
    params: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[ObservationPart, ...]:
    """The parts ``spec`` asks for, in order, resolved for ``env``.

    Parameters come from three places, each one winning over the next: the spec
    itself (``("lidar", {"lidar": {"bins": 16}})``), the ``params`` argument,
    and the part's own defaults. Parts passed in the spec as ready-made
    :class:`ObservationPart` instances are used as they are, which is how a
    caller adds one for a single environment without registering it.

    The parts the environment declares on itself - see
    :func:`env_observation_parts` - are reachable by name here too, so a task
    that writes one for itself does not have to make it global to be asked for.
    """
    inline = {part.name: part for part in env_observation_parts(env)}
    inline.update({part.name: part for part in _inline_parts(spec)})
    given = {str(name): dict(value or {}) for name, value in (params or {}).items()}
    resolved: list[ObservationPart] = []
    seen: set[str] = set()
    for name, own in _explicit_spec(spec):
        if name in seen:
            raise ValueError(f"observation part {name!r} is asked for twice")
        seen.add(name)
        if name in inline:
            resolved.append(inline[name])
            continue
        merged = dict(given.get(name, {}))
        merged.update(own)
        resolved.append(_factory(name)(env, merged))
    unknown = sorted(set(given) - seen)
    if unknown:
        # Refusing this is the point: a typo in a parameter block would
        # otherwise read as "the parameter did nothing".
        raise ValueError(
            "observation parameters were given for part(s) that were not asked "
            f"for: {', '.join(unknown)}"
        )
    return tuple(resolved)


def _inline_parts(spec: Any) -> tuple[ObservationPart, ...]:
    """The ready-made parts a sequence spec carries, if any."""
    if isinstance(spec, ObservationPart):
        return (spec,)
    if not isinstance(spec, Iterable) or isinstance(spec, (str, Mapping)):
        return ()
    return tuple(entry for entry in spec if isinstance(entry, ObservationPart))


def env_observation_parts(env: Any) -> tuple[ObservationPart, ...]:
    """The parts an environment carries itself, resolvable by name.

    A task that writes a part of its own - the polar goal of the social
    example, a battery the project adds - hands it to its environment as
    ``extra_observation_parts``. They are resolved like the registered ones, so
    a caller asks for them by name, but they stay local to that environment:
    nothing seeps into the registry another project shares.
    """
    parts = getattr(env, "extra_observation_parts", ()) or ()
    return tuple(part for part in parts if isinstance(part, ObservationPart))


def spec_parameters(spec: Any) -> dict[str, dict[str, Any]]:
    """The parameters a spec carries itself, by part name.

    The environment sizes its parts before it resolves them - the number of
    bins has to land on the environment before the ``lidar`` part is built - so
    it needs the spec's own blocks without the rest of the resolution.
    """
    return {name: dict(own) for name, own in _explicit_spec(spec)}


def _install_builtins() -> None:
    for name, factory in (
        ("robot", _robot_part),
        ("pose", _pose_part),
        ("velocity", _velocity_part),
        ("goal", _goal_part),
        ("goal_distance", _goal_distance_part),
        ("agents", _agents_part),
        ("lidar", _lidar_part),
        ("lidar_stats", _lidar_stats_part),
        ("nearest_agent", _nearest_agent_part),
        ("humans", _humans_part),
        ("occupancy", _occupancy_part),
        ("status", _status_part),
        ("time", _time_part),
    ):
        register_observation_part(name, factory)


_install_builtins()
