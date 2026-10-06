"""Drive a whole scenario from Python: write it, launch it, stop it.

This is the reverse of the usual direction. Instead of pressing Play and
reading what Unity does, a script writes a scenario file into the project,
asks the session to build it, moves the robot inside it and freezes it again.
The round trip lives in :func:`robotsnap.runs.run_scenario`, which the
``scenario`` sub-command of ``python -m robotsnap`` calls too; this file is its
command-line wrapper.

The bridge has to be listening before the scene dials it, so the order is:

    # 1. this script, from the repository root
    python examples/scenario_control.py --port 10000
    # 2. then press Play in Unity, on a scene whose connector points at
    #    127.0.0.1:10000

The script waits for the connection, writes ``python_demo.yaml`` into
``Assets/StreamingAssets/Scenarios`` of the Unity project, loads it, drives the
primary robot for a few seconds and stops the scenario. The file is removed
again unless ``--keep`` is given.

What it prints on purpose: ``sim_time_seconds`` next to ``scenario_time``. The
first is the simulator's own clock and counts from the moment the session
started, not from the scenario; the second is measured from the launch, which
is the number an episode is made of.
"""

from __future__ import annotations

import argparse

from robotsnap.runs import run_scenario


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "The scenario is written to "
            "<project>/Assets/StreamingAssets/Scenarios/<name>.yaml; pass "
            "--unity-project or set ROBOTSNAP_UNITY_PROJECT to point at the "
            "project when it is not located automatically."
        ),
    )
    parser.add_argument("--port", type=int, default=10000, help="bridge port (default: 10000)")
    parser.add_argument("--host", default="0.0.0.0", help="bridge host (default: 0.0.0.0)")
    parser.add_argument("--scenario", default="python_demo", help="scenario id to write and load")
    parser.add_argument("--unity-project", default=None, help="path of the Unity project")
    parser.add_argument("--timeout", type=float, default=120.0, help="seconds to wait for Unity, then for the world")
    parser.add_argument("--seconds", type=float, default=6.0, help="how long to drive the robot")
    parser.add_argument("--speed", type=float, default=0.4, help="linear speed of the drive, in m/s")
    parser.add_argument("--no-drive", action="store_true", help="only build and freeze the scenario")
    parser.add_argument("--keep", action="store_true", help="keep the scenario file on disk")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_scenario(
        scenario_id=args.scenario,
        port=args.port,
        host=args.host,
        unity_project=args.unity_project,
        timeout=args.timeout,
        seconds=args.seconds,
        speed=args.speed,
        no_drive=args.no_drive,
        keep=args.keep,
    )


if __name__ == "__main__":
    raise SystemExit(main())
