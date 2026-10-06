"""An environment of your own: another observation, another reward, another action.

A custom task inherits :class:`robotsnap.envs.RobotSNAPEnv` and changes a few
methods. The task that used to live in this file - the polar observation, the
social reward, the go-to-goal controller - now lives in
:func:`robotsnap.runs.run_goal_episode`, together with the world it runs in, and
this script is its command-line wrapper; the ``goal`` sub-command of
``python -m robotsnap`` calls the same function. Start from there when you write
your own.

Two things change against the base class, which is what a custom task usually
is:

- the **observation** - the goal and the closest person in polar form instead of
  the default flat vector, written as two components of its own
  (``extra_observation_parts``) and named in its default spec, so a caller can
  still ask for other components;
- the **reward** - progress toward the goal, minus a price for walking too close
  to someone;
- nothing else: the launching, the pacing, the reading of the streams and the
  reset stay in the base class.

It then runs the task with a plain go-to-goal controller, which is the other
half of what this environment is for: measuring a navigation method you already
have, not only training one. Use ``--random`` to see the same task with random
actions for comparison.

    python examples/my_env.py --episodes 3                 # scripted controller
    python examples/my_env.py --episodes 3 --random         # random baseline
    python examples/my_env.py --render                      # watch it in pygame

As with the other examples, start this first and press Play in Unity after.
"""

from __future__ import annotations

import argparse

from robotsnap.runs import run_goal_episode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--port", type=int, default=10000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--scenario", default="python_social_demo")
    parser.add_argument("--unity-project", default=None)
    parser.add_argument("--seconds", type=float, default=45.0, help="budget of one episode")
    parser.add_argument("--control-period", type=float, default=0.2, help="simulated seconds per step")
    parser.add_argument(
        "--time-scale",
        type=float,
        default=None,
        help="Unity time scale for the episode (default: the scene's)",
    )
    parser.add_argument("--social-distance", type=float, default=1.2)
    parser.add_argument("--random", action="store_true", help="random actions, as a baseline")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--keep", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_goal_episode(
        episodes=args.episodes,
        scenario_id=args.scenario,
        port=args.port,
        host=args.host,
        unity_project=args.unity_project,
        seconds=args.seconds,
        control_period=args.control_period,
        time_scale=args.time_scale,
        social_distance=args.social_distance,
        random=args.random,
        render=args.render,
        keep=args.keep,
    )


if __name__ == "__main__":
    raise SystemExit(main())
