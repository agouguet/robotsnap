"""Score a policy over a suite of scenarios, then compare two campaigns.

The command line already has this as ``python -m robotsnap benchmark``; what
this file shows is the same thing as four library calls, for a script that wants
to loop over policies, seeds or suites of its own:

    runs.run_scenarios(...)                  # play a suite, write the campaign
    campaign.load_campaign(path)             # read one back
    campaign.compare_campaigns(first, second)
    campaign.format_comparison(comparison)   # one aligned table

Start it before pressing Play in Unity, exactly like the other examples. The
``--compare`` half needs no session and no simulator at all:

    python examples/benchmark_suite.py --suite basic_short --episodes 3 \
        --out results/goal.json
    python examples/benchmark_suite.py --compare results/goal.json results/ppo.json
"""

from __future__ import annotations

import argparse

from robotsnap.runs import compare, run_scenarios


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--suite",
        default="basic_short",
        help="a suite under configs/benchmarks, a suite file, or ids separated by commas",
    )
    parser.add_argument("--episodes", type=int, default=3, help="episodes per scenario")
    parser.add_argument(
        "--policy",
        choices=("scripted", "random"),
        default="scripted",
        help="what drives the robot when no checkpoint is given",
    )
    parser.add_argument("--load", default=None, help="a checkpoint to benchmark")
    parser.add_argument(
        "--method", default=None, help="the method that wrote the checkpoint, if it is a method's"
    )
    parser.add_argument("--out", default=None, help="where to write the campaign")
    parser.add_argument(
        "--compare",
        nargs="+",
        default=None,
        metavar="CAMPAIGN",
        help="compare saved campaigns instead of running one",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--seconds", type=float, default=120.0)
    parser.add_argument("--port", type=int, default=10000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--unity-project", default=None)
    parser.add_argument("--time-scale", type=float, default=None)
    parser.add_argument(
        "--pacing",
        choices=("free", "lockstep"),
        default="lockstep",
        help="hold the world between two decisions, as a measurement wants",
    )
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--no-stop", action="store_true", help="leave Unity running")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.compare:
        # Two files in, one table out: no session, no Unity, no policy.
        return compare(args.compare)

    return run_scenarios(
        suite=args.suite,
        episodes=args.episodes,
        policy="checkpoint" if args.load else args.policy,
        load=args.load,
        method=args.method,
        out=args.out,
        seed=args.seed,
        max_steps=args.max_steps,
        seconds=args.seconds,
        port=args.port,
        host=args.host,
        unity_project=args.unity_project,
        time_scale=args.time_scale,
        pacing=args.pacing,
        render=args.render,
        keep=args.keep,
        stop=not args.no_stop,
    )


if __name__ == "__main__":
    raise SystemExit(main())
