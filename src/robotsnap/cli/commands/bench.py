"""The ``bench`` command: time N control steps, simulated seconds against wall seconds."""

from __future__ import annotations

import argparse

from robotsnap import runs
from robotsnap.cli.options import (
    _add_bridge_options,
    _add_config_option,
    _add_environment_options,
    _add_pacing_options,
    _add_render_option,
    _add_stop_option,
    _environment_kwargs,
)

NAME = "bench"
ALIASES = ()
HELP = "time N steps, simulated seconds against wall seconds"
DESCRIPTION = "Run N control steps and print simulated seconds against wall seconds."


def add_arguments(parser: argparse.ArgumentParser) -> None:
    _add_bridge_options(parser, scenario="python_bench_demo")
    _add_config_option(parser)
    _add_pacing_options(parser)
    parser.add_argument(
        "--steps", type=int, default=100, help="number of control steps to run"
    )
    parser.add_argument("--keep", action="store_true", help="keep the scenario file on disk")
    _add_stop_option(parser, default=False)
    _add_render_option(parser)
    _add_environment_options(parser)


def run(args: argparse.Namespace) -> int:
    return runs.run_benchmark(
        steps=args.steps,
        scenario_id=args.scenario,
        port=args.port,
        host=args.host,
        unity_project=args.unity_project,
        control_period=args.control_period,
        time_scale=args.time_scale,
        pacing=args.pacing,
        render=args.render,
        keep=args.keep,
        stop=args.stop,
        **_environment_kwargs(args),
    )
