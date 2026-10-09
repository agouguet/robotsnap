"""Train a policy on the RobotSNAP task, with a network of your own.

The environment hands out numpy observations and takes numpy actions, so it is
agnostic about the framework: this script is one way to plug a network in, and
nothing in ``robotsnap.envs`` knows it exists. Swap the policy for your own, or
the loop for the one of a library - the two calls it makes are ``reset()`` and
``step()``.

What runs here is REINFORCE with a return baseline and an entropy bonus: few
lines, no value network, enough to show the wiring end to end. It is not a good
algorithm for this task - a run of a few dozen episodes is a smoke test, not a
result. The loop itself lives in :func:`robotsnap.runs.run_training`, which the
``train`` sub-command of ``python -m robotsnap`` calls too, and PyTorch is
imported only when that function runs.

It needs PyTorch, which the package does not::

    pip install -e ".[env,train]"          # gymnasium and torch
    python examples/env_train_reinforce.py --episodes 20

An interpreter that already has torch works too, with the source tree on the
path::

    PYTHONPATH=src /usr/bin/python3 examples/env_train_reinforce.py --episodes 20

As with the other examples, start this first and press Play in Unity after.
"""

from __future__ import annotations

import argparse

from robotsnap.runs import run_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--entropy", type=float, default=0.01)
    parser.add_argument("--max-steps", type=int, default=1000, help="cap the steps of one episode")
    parser.add_argument("--seconds", type=float, default=120.0, help="budget of one episode, in world seconds")
    parser.add_argument("--control-period", type=float, default=0.2, help="simulated seconds per step")
    parser.add_argument(
        "--time-scale",
        type=float,
        default=None,
        help="Unity time scale for the episode (default: the scene's)",
    )
    parser.add_argument("--port", type=int, default=10000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--scenario", default="default")
    parser.add_argument("--unity-project", default=None)
    parser.add_argument("--save", default=None, help="path to write the weights to")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--keep", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_training(
        episodes=args.episodes,
        gamma=args.gamma,
        lr=args.lr,
        hidden=args.hidden,
        entropy=args.entropy,
        max_steps=args.max_steps,
        seconds=args.seconds,
        control_period=args.control_period,
        time_scale=args.time_scale,
        scenario_id=args.scenario,
        port=args.port,
        host=args.host,
        unity_project=args.unity_project,
        save=args.save,
        render=args.render,
        keep=args.keep,
    )


if __name__ == "__main__":
    raise SystemExit(main())
