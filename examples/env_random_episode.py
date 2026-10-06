"""Run one episode of the goal-navigation environment with random actions.

The smallest complete use of :class:`robotsnap.envs.RobotSNAPEnv`: build an
environment, reset it, step it until the episode ends, and print what happened.
It is also the smoke test to run first - if this works, the bridge, the scene
and the streams are wired the way the environment expects.

The bridge has to be listening before the scene dials it, so the order is:

    # 1. this script, from the repository root
    python examples/env_random_episode.py --port 10000
    # 2. then press Play in Unity, on a scene whose connector points at
    #    127.0.0.1:10000

The run itself lives in :func:`robotsnap.runs.run_random_episode`, which the
``episode`` sub-command of ``python -m robotsnap`` calls too. The scenario is
written by the environment itself, out of the fields of that module, so nothing
has to exist in the project beforehand. It is removed on the way out unless
``--keep`` is given.
"""

from __future__ import annotations

import argparse

from robotsnap.runs import run_random_episode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--port", type=int, default=10000, help="bridge port (default: 10000)")
    parser.add_argument("--host", default="0.0.0.0", help="bridge host (default: 0.0.0.0)")
    parser.add_argument("--scenario", default="python_env_demo", help="scenario id to write and run")
    parser.add_argument("--unity-project", default=None, help="path of the Unity project")
    parser.add_argument("--seconds", type=float, default=20.0, help="episode budget, in world seconds")
    parser.add_argument("--control-period", type=float, default=0.2, help="simulated seconds per step")
    parser.add_argument(
        "--time-scale",
        type=float,
        default=None,
        help="Unity time scale for the episode (default: the scene's)",
    )
    parser.add_argument("--steps", type=int, default=0, help="stop after this many steps (0: the budget)")
    parser.add_argument("--seed", type=int, default=0, help="seed of the episode")
    parser.add_argument("--render", action="store_true", help="draw the session in a pygame window")
    parser.add_argument("--keep", action="store_true", help="keep the scenario file on disk")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_random_episode(
        scenario_id=args.scenario,
        port=args.port,
        host=args.host,
        unity_project=args.unity_project,
        control_period=args.control_period,
        time_scale=args.time_scale,
        seconds=args.seconds,
        steps=args.steps,
        seed=args.seed,
        render=args.render,
        keep=args.keep,
    )


if __name__ == "__main__":
    raise SystemExit(main())
