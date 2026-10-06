"""Playing a Stable-Baselines3 checkpoint on the live session, without training."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from robotsnap import scenario

from robotsnap.runs.presets import training_fields
from robotsnap.runs.session import (
    _announce,
    _environment_options,
    _scenario_directory,
    _stop_session,
)


def run_sb3_policy(
    *,
    load: str,
    algo: str = "auto",
    episodes: int = 1,
    control_period: float = 0.2,
    time_scale: float | None = None,
    pacing: str = "free",
    seconds: float = 45.0,
    scenario_id: str = "python_sb3_play",
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
    """Play a Stable-Baselines3 checkpoint on the live session, without training."""
    from robotsnap.rl import sb3 as sb3_support

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
    _announce(environment, algorithm=str(algo))

    started = False
    try:
        environment.reset()
        started = True
        history = sb3_support.play(
            load,
            environment,
            episodes=int(episodes),
            algo=algo,
        )
        for record in history:
            print(
                f"episode {record['episode'] + 1:3d} | reward {record['reward']:+8.2f}"
                f" | steps {record['steps']:4d} | {record['seconds']:5.1f} s"
                f" | {'arrived' if record['terminated'] else 'ran out'}"
                f"{' | stalled' if record['stalled'] else ''}"
            )
        return 0
    finally:
        _stop_session(environment, stop and started)
        environment.close()
        if not keep:
            scenario.delete(scenario_id, directory=directory)
