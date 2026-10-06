"""The options every sub-command shares.

One place a command module imports its shared options from: the bridge it dials,
how a step is paced, ``--method``, ``--stop``, and the argument types
``--json``-shaped flags use. The environment surface lives beside this module in
:mod:`robotsnap.cli.environment_options` and is re-exported at the bottom, so a
command still has a single import line.
"""

from __future__ import annotations

import argparse
import json

__all__ = []

#: What ``--algo`` means when it names a rule rather than a task. These leave
#: the task alone and say only how a policy's parameters move.
_LEARNING_RULES: tuple[str, ...] = ("reinforce", "ppo", "a2c", "sac", "dqn")


def _catalogue() -> tuple[dict[str, str], dict[str, str]]:
    """``(methods, learners)``, the method catalogue of :mod:`robotsnap.models`.

    Imported here rather than at module import on purpose: the catalogue's
    tables are plain data, so the names, the descriptions and the learner of
    each method can be read without a method's environment - or a framework -
    being loaded. A command that never names a method pays for nothing.
    """
    from robotsnap.models import METHOD_LEARNER, METHODS

    return METHODS, METHOD_LEARNER


def _algo_choices() -> tuple[str, ...]:
    """Every ``--algo`` spelling: the learning rules, plus the method aliases.

    Spelling a method in ``--algo`` is what a command line did before
    ``--method`` existed, so the method names stay accepted there. Reading them
    off the catalogue rather than repeating them keeps the two flags from ever
    disagreeing about which methods exist.
    """
    methods, _ = _catalogue()
    return (*_LEARNING_RULES, *methods)


def _json_object(text: str) -> dict:
    """``text`` as a JSON object, with a message a user can act on."""
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(
            f"not valid JSON ({exc.msg} at line {exc.lineno} column {exc.colno})"
        ) from exc
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError(
            "expected a JSON object, e.g. '{\"lidar\": {\"bins\": 24}}'"
        )
    return value


def _observation_spec(text: str):
    """An observation spec: ``"pose,goal,lidar"``, or a JSON object when written as one.

    The names on their own are the common case, so they need no quoting; the
    JSON form is there for the same reason ``--observation-params`` has one -
    a part and the parameters that size it belong together::

        --observations '{"lidar": {"bins": 24}, "pose": {}}'

    A string that starts with ``{`` but is malformed is reported as bad JSON
    rather than read as a part name nobody registered.
    """
    stripped = text.strip()
    if stripped.startswith("{"):
        return _json_object(stripped)
    return stripped


def _add_bridge_options(parser: argparse.ArgumentParser, *, scenario: str) -> None:
    parser.add_argument(
        "--port", type=int, default=10000, help="bridge port (default: %(default)s)"
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="bridge host (default: %(default)s)"
    )
    parser.add_argument(
        "--transport",
        choices=("tcp", "ros2"),
        default="tcp",
        help=(
            "how the session is reached: 'tcp' binds the bridge port Unity dials, "
            "'ros2' joins a graph the ROS2 endpoint is already serving, where "
            "--host/--port mean nothing (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--scenario", default=scenario, help="scenario id to write and run"
    )
    parser.add_argument(
        "--unity-project", default=None, help="path of the Unity project"
    )


def _add_pacing_options(parser: argparse.ArgumentParser, *, default: str = "free") -> None:
    parser.add_argument(
        "--control-period",
        type=float,
        default=0.2,
        help="simulated seconds per step (default: %(default)s)",
    )
    parser.add_argument(
        "--pacing",
        choices=("free", "lockstep"),
        default=default,
        help=(
            "how a step is timed: 'free' lets the world run at the time scale "
            "while the policy thinks, 'lockstep' has Unity spend exactly one "
            "control period per step and stop in between, which is what keeps "
            "an episode inside its budget and makes a slow model cost wall time "
            "rather than fidelity (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--time-scale",
        type=float,
        default=None,
        help=(
            "Unity time scale for the episode, up to 100 "
            "(a lockstep period holds at most control_period / physics step) "
            "(default: the scene's)"
        ),
    )


def _add_method_option(parser: argparse.ArgumentParser) -> None:
    """The named methods, on their own axis beside the learning rule.

    ``--algo`` names a learning rule: it says how the parameters of a policy
    move and leaves the task alone. A method is more than that: it brings its
    own observation, action set, reward and policy, which is why each one also
    brings the environment that defines those. Putting them on their own axis is
    what makes two methods comparable at all - the resolved line a run prints,
    and the tests that pin it, name each axis separately. The list itself comes
    from the catalogue, so a new method is described here by being registered
    there and nowhere else.
    """
    methods, learners = _catalogue()
    described = ", ".join(
        f"'{name}' ({learners[name]}) - {description}"
        for name, description in methods.items()
    )
    parser.add_argument(
        "--method",
        choices=tuple(methods),
        default=None,
        help=(
            "a named method, with its own observation, action set, reward and "
            f"learner: {described}. Left out, the run drives the plain "
            "environment with --algo. Spelling a method in --algo still means "
            "the method, so a command line written before this flag existed "
            "keeps working"
        ),
    )


def _add_curriculum_option(parser: argparse.ArgumentParser) -> None:
    """The schedule of scenarios a training run climbs, by name or by path.

    A curriculum is the answer to "which world does this episode happen in":
    a file under ``configs/curriculum`` names the stages in order, each with the
    scenario it runs and the rule that promotes the next one. Training without
    it runs one scenario for the whole session, which is what every run did
    before the flag existed.
    """
    parser.add_argument(
        "--curriculum",
        default=None,
        metavar="NAME|PATH",
        help=(
            "a curriculum under configs/curriculum (for example "
            "'social_navigation'), or a path to a YAML file: its stages pick the "
            "scenario each block of episodes runs"
        ),
    )


def _resolve_method(
    args: argparse.Namespace, *, default_algorithm: str | None
) -> tuple[str | None, str | None]:
    """``(algorithm, method)`` from ``--algo`` and ``--method``, or a refusal.

    ``--algo`` still accepts the method names, because it did before ``--method``
    existed; spelling one there is the same as asking for the method. What is
    refused is asking for both at once with two different values, or pairing a
    method with a learning rule: a method already carries its own learner, its
    observation, its action set and its reward, so ``--method <a method> --algo
    <a rule>`` would be silently ignoring one of the two asks. A refusal that
    names both flags is worth more than a run that quietly picks one.
    """
    methods, learners = _catalogue()
    asked = getattr(args, "method", None)
    algorithm = getattr(args, "algo", None)
    method = None

    if algorithm in methods:
        if asked is not None and asked != algorithm:
            raise SystemExit(
                f"--method {asked} and --algo {algorithm} name two different methods: "
                "pass one of them, or spell the same one twice"
            )
        method = asked or algorithm
    elif asked is not None:
        method = asked

    if method is not None and algorithm not in (None, "auto") and algorithm not in methods:
        raise SystemExit(
            f"--method {method} brings its own learner ({learners[method]}), its own "
            f"observation, action set and reward; --algo {algorithm} cannot be combined with "
            f"it. Drop --method to train the plain environment with {algorithm}."
        )

    if method is not None:
        return None, method
    return (algorithm if algorithm is not None else default_algorithm), None


def _add_stop_option(parser: argparse.ArgumentParser, *, default: bool) -> None:
    """The ``--stop``/``--no-stop`` pair, shared by every run with a session."""
    parser.add_argument(
        "--stop",
        action=argparse.BooleanOptionalAction,
        default=default,
        help=(
            "stop the simulation in Unity when the run ends, as the red stop "
            "button does: the scene is left in standby"
        ),
    )


def _add_config_option(parser: argparse.ArgumentParser) -> None:
    """The option that starts a run from a file of hyperparameters.

    A file supplies *defaults*, never overrides: an option typed on the command
    line still wins, which is why :func:`robotsnap.cli.main.main` reads the file
    and only then parses the line. ``NAME`` is resolved under the repository's
    ``configs/`` directory - ``algorithms/ppo``, ``methods/cadrl``, and so on -
    and a path is read where it lies.
    """
    parser.add_argument(
        "--config",
        default=None,
        metavar="NAME|PATH",
        help=(
            "hyperparameters to start from: a name under configs/ "
            "(algorithms/ppo, methods/cadrl, ...) or a path to a YAML/JSON file; "
            "an option given on the command line overrides the file"
        ),
    )


def _add_render_option(parser: argparse.ArgumentParser) -> None:
    """The window a run may draw: ``--viewer``, whose older spelling is ``--render``.

    Both spellings set the same destination, so a script written with
    ``--render`` keeps working and a caller who thinks of the package's 2D
    window as the viewer has the name in front of them.
    """
    parser.add_argument(
        "--viewer",
        "--render",
        dest="render",
        action="store_true",
        help="open the package's 2D window on the session while the run goes on",
    )


# Re-exported so a command imports its shared options from one module. The
# definitions live in ``environment_options`` because the environment surface is
# long enough to deserve its own file; that module imports the two argument
# types above lazily, so this import is safe whichever module loads first.
from robotsnap.cli.environment_options import (  # noqa: E402
    _add_environment_options,
    _environment_kwargs,
    _session_client,
)
