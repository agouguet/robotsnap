"""The ``scenario`` command: write, launch, drive and freeze a scenario."""

from __future__ import annotations

import argparse

from robotsnap import runs
from robotsnap.cli.options import _add_bridge_options, _add_render_option, _session_client

NAME = "scenario"
ALIASES = ()
HELP = "write, launch, drive and freeze a scenario"
DESCRIPTION = "Drive a whole scenario from Python: write it, launch it, stop it."


def add_arguments(parser: argparse.ArgumentParser) -> None:
    _add_bridge_options(parser, scenario="python_demo")
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="seconds to wait for Unity, then for the world",
    )
    parser.add_argument(
        "--seconds", type=float, default=6.0, help="how long to drive the robot"
    )
    parser.add_argument(
        "--speed", type=float, default=0.4, help="linear speed of the drive, in m/s"
    )
    parser.add_argument(
        "--no-drive", action="store_true", help="only build and freeze the scenario"
    )
    _add_render_option(parser)
    parser.add_argument("--keep", action="store_true", help="keep the scenario file on disk")


def run(args: argparse.Namespace) -> int:
    return runs.run_scenario(
        scenario_id=args.scenario,
        client=_session_client(args),
        port=args.port,
        host=args.host,
        unity_project=args.unity_project,
        timeout=args.timeout,
        seconds=args.seconds,
        speed=args.speed,
        no_drive=args.no_drive,
        render=args.render,
        keep=args.keep,
    )
