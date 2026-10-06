"""The ``goal`` command: run the social task with the scripted go-to-goal controller."""

from __future__ import annotations

import argparse

from robotsnap import runs
from robotsnap.cli.options import (
    _add_bridge_options,
    _add_environment_options,
    _add_pacing_options,
    _add_render_option,
    _add_stop_option,
    _environment_kwargs,
)

NAME = "goal"
ALIASES = ()
HELP = "run the social task with a go-to-goal controller"
DESCRIPTION = (
    "Reach the goal while keeping a social distance from the crowd, with a "
    "scripted controller."
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    _add_bridge_options(parser, scenario="python_social_demo")
    _add_pacing_options(parser)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument(
        "--seconds", type=float, default=45.0, help="budget of one episode"
    )
    parser.add_argument("--social-distance", type=float, default=1.2)
    parser.add_argument(
        "--random", action="store_true", help="random actions, as a baseline"
    )
    _add_render_option(parser)
    parser.add_argument("--keep", action="store_true")
    _add_stop_option(parser, default=False)
    _add_environment_options(parser)


def run(args: argparse.Namespace) -> int:
    return runs.run_goal_episode(
        scenario_id=args.scenario,
        port=args.port,
        host=args.host,
        unity_project=args.unity_project,
        control_period=args.control_period,
        time_scale=args.time_scale,
        pacing=args.pacing,
        seconds=args.seconds,
        episodes=args.episodes,
        social_distance=args.social_distance,
        random=args.random,
        render=args.render,
        keep=args.keep,
        stop=args.stop,
        **_environment_kwargs(args),
    )
