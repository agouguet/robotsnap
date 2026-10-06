"""The ``train`` command: train a policy, with a learning rule or a named method."""

from __future__ import annotations

import argparse

from robotsnap import runs
from robotsnap.cli.options import (
    _add_bridge_options,
    _add_config_option,
    _add_curriculum_option,
    _add_environment_options,
    _add_method_option,
    _add_pacing_options,
    _add_render_option,
    _add_stop_option,
    _algo_choices,
    _environment_kwargs,
    _json_object,
    _resolve_method,
)

NAME = "train"
ALIASES = ()
HELP = "train a policy (needs torch)"
DESCRIPTION = (
    "Train a policy on the RobotSNAP task: --algo picks the learning rule, "
    "--method a named method of the literature that brings its own task."
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    _add_bridge_options(parser, scenario="python_train_demo")
    _add_config_option(parser)
    # Training and inference are the two runs where a control period has to mean
    # something: the episode budget is counted in the world's seconds, and a
    # policy that takes its time must cost wall time rather than change the
    # task. Lockstep is what holds that, so it is the default here and not in
    # the demos.
    _add_pacing_options(parser, default="lockstep")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument(
        "--algo",
        default=None,
        choices=_algo_choices(),
        help=(
            "learning rule: 'reinforce' is the loop of this package and the "
            "rest are stable-baselines3's (default: reinforce). A method name "
            "is accepted here too, as the spelling --method had before it "
            "existed; a method is not a rule and brings its own observation, "
            "action set and reward - see --method"
        ),
    )
    _add_method_option(parser)
    _add_curriculum_option(parser)
    parser.add_argument(
        "--timesteps",
        type=int,
        default=20_000,
        help=(
            "control steps to train for, with a stable-baselines3 algorithm: "
            "steps x control_period is the world time a run covers "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--policy-kwargs",
        type=_json_object,
        default=None,
        metavar="JSON",
        help="network options handed to stable-baselines3, e.g. '{\"net_arch\": [128, 128]}'",
    )
    parser.add_argument(
        "--algo-kwargs",
        type=_json_object,
        default=None,
        metavar="JSON",
        help="algorithm options handed to stable-baselines3, e.g. '{\"n_steps\": 512}'",
    )
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--entropy", type=float, default=0.01)
    parser.add_argument(
        "--max-steps", type=int, default=1000, help="cap the steps of one episode"
    )
    parser.add_argument(
        "--seconds", type=float, default=120.0, help="budget of one episode, in world seconds"
    )
    parser.add_argument("--save", default=None, help="path to write the weights to")
    parser.add_argument(
        "--load", default=None, help="checkpoint to resume training from"
    )
    _add_render_option(parser)
    parser.add_argument("--keep", action="store_true")
    _add_stop_option(parser, default=True)
    _add_environment_options(parser)


def run(args: argparse.Namespace) -> int:
    algorithm, method = _resolve_method(args, default_algorithm="reinforce")
    if method is not None:
        return runs.run_social_training(
            algo=method,
            episodes=args.episodes,
            max_steps=args.max_steps,
            seconds=args.seconds,
            control_period=args.control_period,
            time_scale=args.time_scale,
            pacing=args.pacing,
            scenario_id=args.scenario,
            port=args.port,
            host=args.host,
            unity_project=args.unity_project,
            save=args.save,
            load=args.load,
            render=args.render,
            keep=args.keep,
            stop=args.stop,
            curriculum=args.curriculum,
            **_environment_kwargs(args),
        )
    if algorithm != "reinforce":
        # Everything but the historical REINFORCE loop goes through
        # stable-baselines3, which brings its own training loop and its own
        # checkpoint format; the environment and the flags are the same.
        return runs.run_sb3_training(
            algo=algorithm,
            timesteps=args.timesteps,
            gamma=args.gamma,
            lr=args.lr,
            hidden=args.hidden,
            entropy=args.entropy,
            seconds=args.seconds,
            control_period=args.control_period,
            time_scale=args.time_scale,
            pacing=args.pacing,
            scenario_id=args.scenario,
            port=args.port,
            host=args.host,
            unity_project=args.unity_project,
            save=args.save,
            load=args.load,
            render=args.render,
            keep=args.keep,
            stop=args.stop,
            curriculum=args.curriculum,
            policy_kwargs=args.policy_kwargs,
            algo_kwargs=args.algo_kwargs,
            **_environment_kwargs(args),
        )
    return runs.run_training(
        scenario_id=args.scenario,
        port=args.port,
        host=args.host,
        unity_project=args.unity_project,
        control_period=args.control_period,
        time_scale=args.time_scale,
        pacing=args.pacing,
        seconds=args.seconds,
        episodes=args.episodes,
        gamma=args.gamma,
        lr=args.lr,
        hidden=args.hidden,
        entropy=args.entropy,
        max_steps=args.max_steps,
        save=args.save,
        load=args.load,
        render=args.render,
        keep=args.keep,
        stop=args.stop,
        curriculum=args.curriculum,
        **_environment_kwargs(args),
    )
