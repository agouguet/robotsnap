"""What a run actually drives, said in one line.

The command line has one ``--algo`` knob, and its values do not all name the
same kind of thing. ``ppo``, ``a2c``, ``sac`` and ``dqn`` name a *learning rule*:
they leave the task alone and say only how the parameters move. ``reinforce``
does the same through this package's own loop. A name the catalogue of
:mod:`robotsnap.models` registers names a *method* from the literature, and a
method brings more than a learner - its own value network, its own state
encoding, its own command set and its own reward, which is why it also brings
its own environment.

That difference was invisible: ``--observations`` was silently ignored by a
method, two runs both promised "discrete" while handing out two different
command tables, and nothing said which reward a run was paying.
:func:`resolve_run_spec` answers all of it by reading the environment that was
built rather than by re-declaring the mapping somewhere that could drift from
it, and :meth:`RunSpec.describe` is the line a run prints.

Nothing here imports gymnasium, numpy or torch, and nothing here names a method:
the module is imported by the command line, listing what a run is should not
cost a framework, and a task the catalogue grows tomorrow is described here
without an edit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "RunSpec",
    "describe_run",
    "resolve_run_spec",
]


@dataclass(frozen=True)
class RunSpec:
    """The five things a run can be asked about, resolved.

    ``algorithm`` is the learning rule - or ``"random"``, ``"scripted"`` or
    ``"none"`` for the runs that do not learn anything, which still need their
    task named. ``method`` is empty for every run that drives the plain
    environment, and names the method otherwise.
    """

    algorithm: str
    environment: str
    observations: str
    actions: str
    reward: str
    method: str = ""

    def describe(self) -> str:
        """The line a run prints: one row, one key per axis, stable enough to grep."""
        fields = []
        if self.method:
            fields.append(f"method={self.method}")
        fields.append(f"algorithm={self.algorithm}")
        fields.append(f"environment={self.environment}")
        fields.append(f"observations={self.observations}")
        fields.append(f"actions={self.actions}")
        fields.append(f"reward={self.reward}")
        return "run: " + " | ".join(fields)


def resolve_run_spec(
    environment: Any, *, algorithm: str, method: str | None = None
) -> RunSpec:
    """Read ``environment`` and say what it is, with ``algorithm`` and ``method``.

    Every field but the two names is read off the environment itself, so a
    subclass that changes its observation, its action space or its reward is
    described correctly without anyone remembering to update a table.
    """
    return RunSpec(
        algorithm=str(algorithm),
        environment=type(environment).__name__,
        observations=observation_label(environment),
        actions=action_label(environment),
        reward=reward_label(environment),
        method=str(method) if method else "",
    )


def describe_run(environment: Any, *, algorithm: str, method: str | None = None) -> str:
    """``resolve_run_spec(...).describe()``, for the run functions of :mod:`robotsnap.runs`."""
    return resolve_run_spec(environment, algorithm=algorithm, method=method).describe()


def observation_label(environment: Any) -> str:
    """What the environment hands a policy, named the way its caller named it.

    A *structured* observation is described by its shape rather than by its
    width, because that is what distinguishes it from a flat vector: an
    environment that publishes a dictionary of parts - the robot's features, the
    goal, one row per neighbour and a mask - asks the base class for no named
    part at all, so the two cannot disagree, and its size is not one number.

    Everything else is named after the parts it was asked for, or, when it
    publishes a vector without naming parts at all, by the width of that vector.
    """
    space = getattr(environment, "observation_space", None)
    names = tuple(getattr(environment, "observation_names", ()) or ())
    if names:
        joined = ",".join(str(name) for name in names)
        structure = str(getattr(environment, "observation_structure", "flat"))
        return joined if structure == "flat" else f"{joined} ({structure})"

    # No named part: the environment publishes an observation of its own. A
    # dictionary of arrays is a task that hands out ego, goal, neighbour rows
    # and a mask rather than a vector, and its own size is not one number.
    parts = getattr(space, "spaces", None)
    if parts is not None and len(parts) > 0:
        neighbours = int(getattr(environment, "max_neighbours", 0) or 0)
        return f"social(dict, {neighbours} neighbours)"

    width = getattr(space, "shape", None)
    if width is not None and len(width) == 1 and int(width[0]) > 0:
        return f"flat({int(width[0])})"
    return "none"


def action_label(environment: Any) -> str:
    """The action space in the shape a reader compares: ``Discrete(80)``, ``Box(2)``."""
    space = getattr(environment, "action_space", None)
    if space is None:
        return "?"

    kind = type(space).__name__
    # A finite set is asked for its size before its shape: gymnasium's Discrete
    # carries an empty shape, so reading the shape first would answer "Discrete"
    # and drop the one number that says what the run can do.
    count = getattr(space, "n", None)
    if count is not None:
        try:
            return f"{kind}({int(count)})"
        except (TypeError, ValueError):
            return kind

    shape = getattr(space, "shape", None)
    if shape is not None:
        try:
            first = int(shape[0])
        except (IndexError, TypeError, ValueError):
            return kind
        return f"{kind}({first})"
    return kind


def reward_label(environment: Any) -> str:
    """Which reward is paid, and with what weights.

    A method that brings its own cost function says so by carrying it, and the
    base environment pays its own weighted sum; printing the weights is what
    makes two runs comparable on paper as well as on screen.
    """
    if getattr(environment, "social_reward", None) is not None:
        return "social"

    weights = []
    for name, attribute in (
        ("goal", "goal_reward"),
        ("progress", "progress_reward"),
        ("collision", "collision_penalty"),
        ("oob", "out_of_bounds_penalty"),
        ("time", "time_penalty"),
    ):
        value = getattr(environment, attribute, None)
        if value is None:
            continue
        try:
            weights.append(f"{name} {float(value):g}")
        except (TypeError, ValueError):
            continue
    return " ".join(weights) if weights else "default"
