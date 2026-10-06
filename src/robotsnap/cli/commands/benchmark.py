"""The ``benchmark`` command: a suite of scenarios, a policy, and its numbers."""

from __future__ import annotations

import argparse

from robotsnap import runs
from robotsnap.cli.options import (
    _add_bridge_options,
    _add_config_option,
    _add_environment_options,
    _add_method_option,
    _add_pacing_options,
    _add_render_option,
    _add_stop_option,
    _environment_kwargs,
)

NAME = "benchmark"
ALIASES = ("campaign",)
HELP = "run a suite of scenarios and compare the campaigns"
DESCRIPTION = (
    "Play a policy over a suite of scenarios, N episodes each, average the "
    "metrics Unity reports, save the campaign, and compare saved campaigns. "
    "'bench' times the session; this command scores it."
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    _add_bridge_options(parser, scenario="python_benchmark_demo")
    _add_config_option(parser)
    # A campaign is a measurement, so the world is held between two decisions by
    # default: at a single times speed a free-running session lets the world move
    # while a learned policy thinks, and the episode it measures is then not the
    # one the policy was written for.
    _add_pacing_options(parser, default="lockstep")
    parser.add_argument(
        "--suite",
        default="basic",
        metavar="NAME|PATH|LIST",
        help=(
            "scenarios to run: a suite under configs/benchmarks, a path to a "
            "suite file, or ids separated by commas (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="episodes per scenario, so the campaign runs suite x episodes (default: %(default)s)",
    )
    parser.add_argument(
        "--policy",
        choices=("scripted", "random", "external", "ros2"),
        default="scripted",
        help=(
            "what drives the robot when no checkpoint is given: this package's "
            "go-to-goal controller, actions drawn from the space, or - with "
            "'external' (alias 'ros2') - a process outside Python that drives the "
            "robot through the bridge (the ROS2 graph), Python only pacing the "
            "world, resetting scenarios and measuring "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--load",
        default=None,
        help="checkpoint to benchmark, written by train (sb3, policy, or a method)",
    )
    _add_method_option(parser)
    parser.add_argument(
        "--algo",
        default="auto",
        choices=("auto", "ppo", "a2c", "sac", "dqn"),
        help="which learning rule wrote an sb3 checkpoint (default: %(default)s)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="where to write the campaign, so a later run can be compared to it",
    )
    parser.add_argument(
        "--compare",
        nargs="+",
        default=None,
        metavar="CAMPAIGN",
        help="compare saved campaigns instead of running one (needs at least two)",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="cap the steps of one episode (default: %(default)s)",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=120.0,
        help="budget of one episode, in world seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--name", default=None, help="name the campaign (default: the suite's name)"
    )
    _add_render_option(parser)
    parser.add_argument("--keep", action="store_true")
    _add_stop_option(parser, default=True)
    _add_environment_options(parser)


def run(args: argparse.Namespace) -> int:
    if args.compare:
        return runs.compare(args.compare)

    # A checkpoint decides its own environment, so it is its own policy kind;
    # what the flag names is only the two checkpoint-free floors.
    policy = "checkpoint" if args.load else args.policy
    return runs.run_scenarios(
        suite=args.suite,
        episodes=args.episodes,
        policy=policy,
        load=args.load,
        method=args.method,
        algo=args.algo,
        name=args.name,
        out=args.out,
        seed=args.seed,
        max_steps=args.max_steps,
        seconds=args.seconds,
        scenario_id=args.scenario,
        port=args.port,
        host=args.host,
        robot=args.robot,
        unity_project=args.unity_project,
        control_period=args.control_period,
        time_scale=args.time_scale,
        pacing=args.pacing,
        render=args.render,
        keep=args.keep,
        stop=args.stop,
        **_environment_kwargs(args),
    )
