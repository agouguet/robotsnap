"""The two short runs: a random episode, and the scripted controller.

Their formatters live here too, so the lines the two runs print are written
beside the runs that print them.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from robotsnap import scenario

from robotsnap.runs.controllers import go_to_goal
from robotsnap.runs.presets import random_episode_fields, social_fields
from robotsnap.runs.session import (
    _announce,
    _announce_session,
    _environment_options,
    _scenario_directory,
    _stop_session,
)


def run_random_episode(
    *,
    scenario_id: str = "python_env_demo",
    scenario_fields: Mapping[str, Any] | None = None,
    client: Any = None,
    port: int = 10000,
    host: str = "0.0.0.0",
    robot: str | None = None,
    control_period: float = 0.2,
    wait_timeout: float = 30.0,
    time_scale: float | None = None,
    pacing: str = "free",
    seconds: float = 20.0,
    steps: int = 0,
    seed: int = 0,
    render: bool = False,
    keep: bool = False,
    stop: bool = False,
    unity_project: str | None = None,
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
    """One episode of goal navigation with random actions, and what happened.

    The smallest complete use of :class:`robotsnap.envs.RobotSNAPEnv`: build the
    environment, reset it, step it until the episode ends, and print. The
    scenario is written by the environment itself, out of ``scenario_fields``.
    """
    import numpy as np

    from robotsnap.envs import RobotSNAPEnv

    if scenario_fields is None:
        scenario_fields = random_episode_fields()
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
    _announce(environment, algorithm="random")
    rng = np.random.default_rng(seed)
    total = 0.0
    try:
        _announce_session(environment.client, host=host, port=port)
        observation, info = environment.reset(seed=seed)
        print(f"episode started  : scenario {info['scenario_id']}")
        print(
            f"goal at          : {info['distance_to_goal']:.2f} m,"
            f" bearing {info['bearing_to_goal']:+.2f} rad"
        )
        print(f"observation      : {_observation_summary(observation)}")

        step = 0
        while True:
            action = rng.uniform(
                environment.action_space.low, environment.action_space.high
            )
            observation, reward, terminated, truncated, info = environment.step(action)
            total += reward
            step += 1
            if render:
                environment.render()
            if step % 10 == 0 or terminated or truncated:
                print(
                    f"step {step:4d} | goal {_metres(info['distance_to_goal'])}"
                    f" | closest {_metres(info['min_lidar'])}"
                    f" | reward {total:+7.2f} | {info['episode_seconds']:5.1f} s"
                )
            if terminated or truncated:
                print(f"episode ended    : {'truncated' if truncated else _why(info)}")
                break
            if steps and step >= steps:
                print("stopped by --steps")
                break
        return 0
    finally:
        _stop_session(environment, stop)
        environment.close()
        if not keep:
            removed = scenario.delete(scenario_id, directory=directory)
            print(f"scenario removed : {removed}")


def _metres(value) -> str:
    return "  n/a " if value is None else f"{value:5.2f}m"


def _observation_summary(observation) -> str:
    """``shape dtype`` for a flat array, the named shapes for a dict one.

    The flat form is the one the environment has always handed out and what the
    historical line printed; a structured observation prints the parts instead,
    in the order they were asked for.
    """
    if hasattr(observation, "shape") and hasattr(observation, "dtype"):
        return f"{observation.shape} {observation.dtype}"
    if isinstance(observation, Mapping):
        parts = ", ".join(
            f"{name}{getattr(value, 'shape', '')}" for name, value in observation.items()
        )
        return f"dict({parts})"
    return repr(observation)


def _why(info) -> str:
    if info["goal_reached"]:
        return "goal reached"
    if info["collision"]:
        return "collision"
    if info["out_of_bounds"]:
        return "left the map"
    return "unknown"


def run_goal_episode(
    *,
    episodes: int = 1,
    scenario_id: str = "python_social_demo",
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
    social_distance: float = 1.2,
    random: bool = False,
    render: bool = False,
    keep: bool = False,
    stop: bool = False,
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
    """The social task with a plain go-to-goal controller, or random actions.

    Measuring a navigation method you already have is half of what this
    environment is for; ``random`` is the baseline it is measured against.
    """
    import numpy as np

    from robotsnap.runs.training import _social_nav_class

    environment_class = _social_nav_class()

    if scenario_fields is None:
        scenario_fields = social_fields()
    directory = _scenario_directory(unity_project)
    if directory is None:
        return 2

    environment = environment_class(
        social_distance=social_distance,
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
    )
    _announce(environment, algorithm="scripted")
    rng = np.random.default_rng(0)
    outcomes = []
    try:
        _announce_session(environment.client, host=host, port=port)
        for episode in range(episodes):
            _, info = environment.reset(seed=episode)
            total = 0.0
            while True:
                action = (
                    rng.uniform(
                        environment.action_space.low, environment.action_space.high
                    )
                    if random
                    else go_to_goal(info["world"])
                )
                _, reward, terminated, truncated, info = environment.step(action)
                total += reward
                if render:
                    environment.render()
                if terminated or truncated:
                    break
            outcome = _outcome(info)
            outcomes.append(outcome)
            print(
                f"episode {episode + 1}: {outcome:>13} | reward {total:+7.2f}"
                f" | {info['episode_seconds']:5.1f} s | closest {_closest(info['min_lidar'])}"
            )
    finally:
        _stop_session(environment, stop)
        environment.close()
        if not keep:
            scenario.delete(scenario_id, directory=directory)

    if outcomes:
        reached = outcomes.count("goal reached")
        print(f"{reached}/{len(outcomes)} episodes reached the goal")
    return 0


def _outcome(info) -> str:
    if info["goal_reached"]:
        return "goal reached"
    if info["collision"]:
        return "collision"
    if info["out_of_bounds"]:
        return "left the map"
    return "out of time"


def _closest(value) -> str:
    return "n/a" if value is None else f"{value:.2f}m"
