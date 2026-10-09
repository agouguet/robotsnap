"""Playing this package's own policy, and reading what wrote a checkpoint.

Three writers put a policy on disk and they do not share a format, so the file
decides which reader a run uses.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from robotsnap.rl import policy

from robotsnap.runs.episodes import _outcome
from robotsnap.runs.presets import training_fields
from robotsnap.runs.session import (
    _announce,
    _announce_session,
    _environment_options,
    _remove_scratch_scenario,
    _scenario_directory,
    _stop_session,
)


def run_policy_episodes(
    load: str,
    *,
    episodes: int = 1,
    sample: bool = False,
    stop: bool = True,
    scenario_id: str = "python_policy_demo",
    scenario_fields: Mapping[str, Any] | None = None,
    client: Any = None,
    port: int = 10000,
    host: str = "0.0.0.0",
    unity_project: str | None = None,
    seconds: float = 45.0,
    robot: str | None = None,
    control_period: float = 0.2,
    wait_timeout: float = 30.0,
    time_scale: float | None = None,
    pacing: str = "free",
    render: bool = False,
    keep: bool = False,
    hidden: int | None = None,
    observations: Any = None,
    observation_params: Mapping[str, Any] | None = None,
    observation_structure: str = "flat",
    max_agents: int = 8,
    max_humans: int = 8,
    lidar_bins: int = 48,
    goal_radius: float = 0.5,
    collision_distance: float = 0.25,
    occupancy_threshold: int = 50,
    max_linear: float = 1.0,
    max_angular: float = 1.0,
    reload_on_reset: bool = False,
    control_mode: str = "ros",
    goal_reward: float = 10.0,
    progress_reward: float = 1.0,
    collision_penalty: float = 10.0,
    out_of_bounds_penalty: float = 5.0,
    time_penalty: float = 0.1,
) -> int:
    """Play a saved policy on the task it was trained for, without learning.

    The checkpoint carries the network, the statistics the observations were
    normalised by and the width of the observation, so the run only checks that
    the environment hands out the same vector and then steps. The action is the
    mean of the Gaussian - deterministic - unless ``sample`` asks for a draw
    from it; either way no gradient is taken and the normalisation never moves.
    A checkpoint with no saved normalisation is played with a warning: its
    observations reach the network raw.
    """
    import torch

    from robotsnap.envs import RobotSNAPEnv

    if scenario_fields is None:
        scenario_fields = training_fields()
    directory = _scenario_directory(unity_project)
    if directory is None:
        return 2

    environment = RobotSNAPEnv(
        **_environment_options(
            scenario_id=scenario_id,
            scenario_fields=scenario_fields,
            scenario_directory=directory,
            client=client,
            port=port,
            host=host,
            robot=robot,
            control_period=control_period,
            wait_timeout=wait_timeout,
            time_scale=time_scale,
            pacing=pacing,
            seconds=seconds,
            render=render,
            observations=observations,
            observation_params=observation_params,
            observation_structure=observation_structure,
            max_agents=max_agents,
            max_humans=max_humans,
            lidar_bins=lidar_bins,
            goal_radius=goal_radius,
            collision_distance=collision_distance,
            occupancy_threshold=occupancy_threshold,
            max_linear=max_linear,
            max_angular=max_angular,
            reload_on_reset=reload_on_reset,
            control_mode=control_mode,
            goal_reward=goal_reward,
            progress_reward=progress_reward,
            collision_penalty=collision_penalty,
            out_of_bounds_penalty=out_of_bounds_penalty,
            time_penalty=time_penalty,
        )
    )
    _announce(environment, algorithm="reinforce")
    size = int(environment.observation_space.shape[0])
    outcomes: list[str] = []
    started = False
    try:
        try:
            network, norm, meta = policy.load_policy(
                load, observation_size=size, hidden=hidden
            )
        except ValueError as exc:
            print(f"cannot play {load}: {exc}", file=sys.stderr)
            return 2
        if size != int(meta["observation_size"]):
            print(
                f"cannot play {load}: the checkpoint was trained on a "
                f"{meta['observation_size']}-value observation, but this "
                f"environment hands out {size}",
                file=sys.stderr,
            )
            return 2

        _announce_session(environment.client, host=host, port=port)
        if meta["legacy"] or meta["normaliser"] is None:
            print(
                f"warning: {load} carries no normalisation statistics; the "
                "observations reach the network raw",
                file=sys.stderr,
            )
        started = True
        with torch.no_grad():
            for episode in range(episodes):
                observation, info = environment.reset(seed=episode)
                total = 0.0
                steps = 0
                while True:
                    tensor = torch.from_numpy(
                        norm.normalise(observation)
                    ).unsqueeze(0)
                    mean, std = network(tensor)
                    action = (
                        torch.distributions.Normal(mean, std).sample()
                        if sample
                        else mean
                    )
                    observation, reward, terminated, truncated, info = environment.step(
                        action.detach().numpy()[0]
                    )
                    total += reward
                    steps += 1
                    if render:
                        environment.render()
                    if terminated or truncated:
                        break
                outcome = _outcome(info)
                outcomes.append(outcome)
                print(
                    f"episode {episode + 1}: {outcome:>13} | reward {total:+7.2f}"
                    f" | steps {steps:4d} | {info['episode_seconds']:5.1f} s"
                )
    finally:
        _stop_session(environment, stop and started)
        environment.close()
        if not keep:
            _remove_scratch_scenario(environment, scenario_id, directory)

    if outcomes:
        reached = outcomes.count("goal reached")
        print(f"{reached}/{len(outcomes)} episodes reached the goal")
    return 0


def _checkpoint_kind(path: str | Path) -> str:
    """What wrote a checkpoint, read from the file rather than from its name.

    Three writers put a policy on disk here and they do not share a format: a
    Stable-Baselines3 archive is a zip, and both this package's policy and the
    social agents write a torch file whose own ``format`` field says which. The
    file decides, because a name can be anything and loading a checkpoint with
    the wrong reader is how a good policy looks broken.
    """
    try:
        import torch

        document = torch.load(str(path), map_location="cpu", weights_only=False)
    except Exception:
        return "sb3"

    if isinstance(document, Mapping):
        from robotsnap.models.social import FORMAT as SOCIAL_FORMAT

        if document.get("format") == SOCIAL_FORMAT:
            return "social"
    return "policy"


def _action_count(environment) -> int:
    """How many actions a method's environment hands out.

    A social method publishes its own command grid; a method built on the base
    class publishes a Gymnasium space instead. Both are one number to a log
    line, and reading whichever is there keeps the run line honest for a
    method that has no crowd and no grid.
    """
    space = getattr(environment, "social_actions", None)
    if space is not None:
        return len(space)
    return int(environment.action_space.n)
