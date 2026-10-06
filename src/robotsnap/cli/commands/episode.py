"""The ``episode`` command: one random-action episode of goal navigation."""

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

NAME = "episode"
ALIASES = ()
HELP = "run one episode of goal navigation with random actions"
DESCRIPTION = "Run one episode of the goal-navigation environment with random actions."


def add_arguments(parser: argparse.ArgumentParser) -> None:
    _add_bridge_options(parser, scenario="python_env_demo")
    _add_pacing_options(parser)
    parser.add_argument(
        "--seconds", type=float, default=20.0, help="episode budget, in world seconds"
    )
    parser.add_argument(
        "--steps", type=int, default=0, help="stop after this many steps (0: the budget)"
    )
    parser.add_argument("--seed", type=int, default=0, help="seed of the episode")
    _add_render_option(parser)
    parser.add_argument(
        "--keep", action="store_true", help="keep the scenario file on disk"
    )
    _add_stop_option(parser, default=False)
    _add_environment_options(parser)


def run(args: argparse.Namespace) -> int:
    return runs.run_random_episode(
        scenario_id=args.scenario,
        port=args.port,
        host=args.host,
        unity_project=args.unity_project,
        control_period=args.control_period,
        time_scale=args.time_scale,
        pacing=args.pacing,
        seconds=args.seconds,
        steps=args.steps,
        seed=args.seed,
        render=args.render,
        keep=args.keep,
        stop=args.stop,
        **_environment_kwargs(args),
    )
