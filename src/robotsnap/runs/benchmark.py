"""The timed run: simulated seconds against wall seconds."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from typing import Any

from robotsnap import scenario

from robotsnap.runs.presets import bench_fields
from robotsnap.runs.session import (
    _announce,
    _announce_session,
    _environment_options,
    _scenario_directory,
    _stop_session,
)


def run_benchmark(
    *,
    steps: int = 100,
    scenario_id: str = "python_bench_demo",
    scenario_fields: Mapping[str, Any] | None = None,
    client: Any = None,
    port: int = 10000,
    host: str = "0.0.0.0",
    robot: str | None = None,
    control_period: float = 0.2,
    wait_timeout: float = 30.0,
    time_scale: float | None = None,
    pacing: str = "free",
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
    """Run N control steps and print simulated seconds against wall seconds.

    The ratio is the measurement the pacing exists for: at a session running at
    ``time_scale``, one wall second is that many simulated seconds, so a run that
    prints ``4.0x`` covers four seconds of the world per second of the wall. The
    episode budget is not a stop condition here - the step count is - so the
    simulated clock is left free to run.
    """
    from robotsnap.envs import RobotSNAPEnv

    if scenario_fields is None:
        scenario_fields = bench_fields()
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
            seconds=math.inf,
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
    _announce(environment, algorithm="none")
    try:
        _announce_session(environment.client, host=host, port=port)
        environment.reset()
        started = time.monotonic()
        taken = 0
        info: dict[str, Any] = {}
        for _ in range(max(0, steps)):
            _, _, terminated, truncated, info = environment.step([0.0, 0.0])
            if render:
                environment.render()
            taken += 1
            if terminated or truncated:
                break
        wall = time.monotonic() - started
        simulated = float(info.get("episode_seconds") or 0.0)
        ratio = simulated / wall if wall > 0.0 else 0.0
        print(f"steps            : {taken}")
        print(f"sim time         : {simulated:.3f} s")
        print(f"wall time        : {wall:.3f} s")
        print(f"sim/wall         : {ratio:.2f}x")
        return 0
    finally:
        _stop_session(environment, stop)
        environment.close()
        if not keep:
            removed = scenario.delete(scenario_id, directory=directory)
            print(f"scenario removed : {removed}")
