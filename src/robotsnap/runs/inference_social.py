"""Playing a social method's checkpoint, and reading which method wrote it."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from robotsnap import scenario

from robotsnap.runs.presets import training_fields
from robotsnap.runs.session import (
    _announce,
    _environment_options,
    _scenario_directory,
    _stop_session,
)


def run_social_policy(
    *,
    load: str,
    algo: str,
    episodes: int = 1,
    control_period: float = 0.2,
    time_scale: float | None = None,
    pacing: str = "free",
    seconds: float = 45.0,
    scenario_id: str = "python_social_play",
    scenario_fields: Mapping[str, Any] | None = None,
    client: Any = None,
    port: int = 10000,
    host: str = "0.0.0.0",
    robot: str | None = None,
    wait_timeout: float = 30.0,
    unity_project: str | None = None,
    render: bool = False,
    keep: bool = False,
    stop: bool = False,
    max_neighbours: int = 8,
    social_parameters: Mapping[str, Any] | None = None,
    observations: Any = None,
    observation_params: Mapping[str, Mapping[str, Any]] | None = None,
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
    """Play a method's checkpoint on the live session, without training."""
    from robotsnap.models import ALGORITHMS as SOCIAL_ALGORITHMS
    from robotsnap.models import METHOD_LEARNER
    from robotsnap.models.social import play as social_play

    key = str(algo).strip().lower() if algo else next(iter(SOCIAL_ALGORITHMS))
    if key not in SOCIAL_ALGORITHMS:
        print(
            f"unknown method {algo!r}: {', '.join(sorted(SOCIAL_ALGORITHMS))} "
            "are what robotsnap.models registers"
        )
        return 2

    environment_class = SOCIAL_ALGORITHMS[key][0]
    if scenario_fields is None:
        scenario_fields = training_fields()
    directory = _scenario_directory(unity_project)
    if directory is None:
        return 2

    environment = environment_class(
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
        ),
        max_neighbours=int(max_neighbours),
        social_parameters=social_parameters,
    )
    # The learner the checkpoint was trained with, read from the catalogue:
    # CADRL and SARL are state-value methods with a lookahead, GA3C-CADRL is an
    # actor-critic, and the one line a run prints has to say which.
    _announce(environment, algorithm=METHOD_LEARNER.get(key, "dqn"), method=key)

    started = False
    try:
        environment.reset()
        started = True
        history = social_play(load, environment, episodes=int(episodes), log=print)
        for index, outcome in enumerate(history.get("outcome", [])):
            print(f"episode {index + 1:3d} | outcome {outcome}")
        return 0
    finally:
        _stop_session(environment, stop and started)
        environment.close()
        if not keep:
            scenario.delete(scenario_id, directory=directory)


def social_checkpoint_algorithm(path: str | Path) -> str:
    """Which method's agent wrote a checkpoint, read from the agent's own document.

    A method's agent writes its own name beside its weights, so which one to
    rebuild is a property of the file: reading it is cheaper and truer than
    asking the user to repeat on the command line what the file already says.
    """
    import torch

    from robotsnap.models import METHODS as KNOWN_METHODS

    document = torch.load(str(path), map_location="cpu", weights_only=False)
    name = document.get("algorithm") if isinstance(document, Mapping) else None
    if name not in KNOWN_METHODS:
        raise ValueError(
            f"{path} does not name a method of robotsnap.models; it says {name!r}"
        )
    return str(name)
