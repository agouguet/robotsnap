"""The ``play`` command: play a saved policy back, without training."""

from __future__ import annotations

import argparse

from robotsnap import runs
from robotsnap.cli.options import (
    _LEARNING_RULES,
    _add_bridge_options,
    _add_config_option,
    _add_environment_options,
    _add_method_option,
    _add_pacing_options,
    _add_render_option,
    _add_stop_option,
    _catalogue,
    _environment_kwargs,
    _resolve_method,
)

NAME = "play"
ALIASES = ("infer",)
HELP = "play a saved policy (needs torch)"
DESCRIPTION = "Load a saved policy and run it on the RobotSNAP task, without training."


def add_arguments(parser: argparse.ArgumentParser) -> None:
    _add_bridge_options(parser, scenario="python_policy_demo")
    _add_config_option(parser)
    # Free running, unlike training: playing a policy back is how a run is
    # *watched*, and lockstep is the mode that stops the world between two
    # decisions - which is what keeps a step to one control period whatever the
    # policy costs, and what makes the scene pause for as long as the policy
    # takes to answer. At a single times speed that pause is what a watcher sees
    # as lag, so the smooth mode is the default here and the exact one stays a
    # flag away. See --pacing.
    _add_pacing_options(parser, default="free")
    parser.add_argument(
        "--load",
        required=True,
        help="checkpoint written by `train --save` to play back",
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument(
        "--algo",
        default="auto",
        choices=("auto", *_LEARNING_RULES[1:], *_catalogue()[0]),
        help=(
            "which learning rule wrote the checkpoint; 'auto' reads the file "
            "itself, and a method name is the spelling --method had before it "
            "existed (default: %(default)s)"
        ),
    )
    _add_method_option(parser)
    parser.add_argument(
        "--seconds", type=float, default=45.0, help="budget of one episode"
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="draw the action from the policy instead of taking its mean",
    )
    _add_render_option(parser)
    parser.add_argument("--keep", action="store_true")
    _add_stop_option(parser, default=True)
    _add_environment_options(parser)


def run(args: argparse.Namespace) -> int:
    algorithm, method = _resolve_method(args, default_algorithm=None)
    if method is not None:
        kind = "social"
    elif algorithm in ("ppo", "a2c", "sac", "dqn"):
        kind = "sb3"
    else:
        kind = runs._checkpoint_kind(args.load)
    if kind == "social":
        return runs.run_social_policy(
            load=args.load,
            algo=method or runs.social_checkpoint_algorithm(args.load),
            episodes=args.episodes,
            control_period=args.control_period,
            time_scale=args.time_scale,
            pacing=args.pacing,
            seconds=args.seconds,
            scenario_id=args.scenario,
            port=args.port,
            host=args.host,
            unity_project=args.unity_project,
            render=args.render,
            keep=args.keep,
            stop=args.stop,
            **_environment_kwargs(args),
        )
    if kind == "sb3":
        # A Stable-Baselines3 checkpoint carries its own algorithm and its own
        # observation shape; a PyTorch file written by --algo reinforce does not
        # look like this at all, so the file says which reader to use.
        return runs.run_sb3_policy(
            load=args.load,
            algo=args.algo,
            episodes=args.episodes,
            control_period=args.control_period,
            time_scale=args.time_scale,
            pacing=args.pacing,
            seconds=args.seconds,
            scenario_id=args.scenario,
            port=args.port,
            host=args.host,
            unity_project=args.unity_project,
            render=args.render,
            keep=args.keep,
            stop=args.stop,
            **_environment_kwargs(args),
        )
    return runs.run_policy_episodes(
        load=args.load,
        scenario_id=args.scenario,
        port=args.port,
        host=args.host,
        unity_project=args.unity_project,
        control_period=args.control_period,
        time_scale=args.time_scale,
        pacing=args.pacing,
        seconds=args.seconds,
        episodes=args.episodes,
        sample=args.sample,
        render=args.render,
        keep=args.keep,
        stop=args.stop,
        **_environment_kwargs(args),
    )
