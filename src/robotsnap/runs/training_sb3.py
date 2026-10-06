"""The Stable-Baselines3 training run.

The environment is a Gymnasium environment already, so the algorithm drives the
live simulation directly; what this run owns is the environment's own options,
the translation of the shared flags onto the algorithm, and the two ends of the
run.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from robotsnap import scenario

from robotsnap.runs.presets import training_fields
from robotsnap.runs.session import (
    _apply_curriculum,
    _build_environment,
    _curriculum,
    _environment_options,
    _scenario_directory,
    _stop_session,
)


def _curriculum_callback(plan: Any) -> Any:
    """A Stable-Baselines3 callback that prints each curriculum promotion.

    A curriculum that advances silently is indistinguishable from one that never
    does, and the training loop of Stable-Baselines3 has no other seam for the
    package to write into. The callback reads the plan the wrapper also reads,
    so it can never disagree with the stage the environment is really running.
    """
    if plan is None:
        return None

    from stable_baselines3.common.callbacks import BaseCallback

    class _LogCurriculum(BaseCallback):
        def __init__(self) -> None:
            super().__init__()
            self._seen = -1

        def _on_step(self) -> bool:
            if plan.index != self._seen:
                self._seen = plan.index
                print(f"curriculum: {plan.summary()}", flush=True)
            return True

    return _LogCurriculum()


def run_sb3_training(
    *,
    algo: str = "ppo",
    timesteps: int = 20_000,
    gamma: float = 0.99,
    lr: float = 3e-4,
    hidden: int = 64,
    entropy: float = 0.01,
    seconds: float = 120.0,
    control_period: float = 0.2,
    time_scale: float | None = None,
    pacing: str = "free",
    scenario_id: str = "python_sb3_demo",
    scenario_fields: Mapping[str, Any] | None = None,
    client: Any = None,
    port: int = 10000,
    host: str = "0.0.0.0",
    robot: str | None = None,
    wait_timeout: float = 30.0,
    unity_project: str | None = None,
    save: str | None = None,
    load: str | None = None,
    render: bool = False,
    keep: bool = False,
    stop: bool = False,
    policy_kwargs: Mapping[str, Any] | None = None,
    algo_kwargs: Mapping[str, Any] | None = None,
    curriculum: str | None = None,
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
    """Train a Stable-Baselines3 algorithm on the RobotSNAP task.

    The environment is a Gymnasium environment already, so the algorithm drives
    the live simulation directly and nothing here re-implements a training loop:
    what this function owns is the environment's own options, the translation of
    the shared flags onto the algorithm (``--lr`` on the learning rate,
    ``--hidden`` on the network width, ``--gamma`` on the discount), and the two
    ends of the run - the session it starts and the session it stops.

    ``timesteps`` is in control steps, so the world seconds a run covers are
    ``timesteps * control_period``: a hundred thousand steps at a fifth of a
    second is five and a half hours of the world, at whatever speed the session
    manages to deliver them.
    """
    from robotsnap.rl import sb3 as sb3_support

    from robotsnap.envs import RobotSNAPEnv

    key = str(algo).strip().lower()
    if key not in sb3_support.ALGORITHMS:
        print(
            f"unknown algorithm {algo!r}: {', '.join(sorted(sb3_support.ALGORITHMS))} "
            "are the ones stable-baselines3 brings, or use --algo reinforce"
        )
        return 2

    if scenario_fields is None:
        scenario_fields = training_fields()
    directory = _scenario_directory(unity_project)
    if directory is None:
        return 2

    plan = _curriculum(curriculum)

    options = _environment_options(
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

    tuned = dict(policy_kwargs or {})
    if hidden:
        tuned.setdefault("net_arch", [int(hidden), int(hidden)])
    algorithm_options: dict[str, Any] = {
        "gamma": float(gamma),
        "learning_rate": float(lr),
    }
    if key in ("ppo", "a2c"):
        algorithm_options["ent_coef"] = float(entropy)
    # A value-based run does not pay for an entropy term, and DQN of SB3 takes
    # neither ent_coef nor a policy network wider than its net_arch.
    algorithm_options.update(dict(algo_kwargs or {}))

    def wrap(environment: Any) -> Any:
        """The curriculum first, then the action set a value-based run needs."""
        environment = _apply_curriculum(environment, plan)
        if sb3_support.ALGORITHMS[key]:
            environment = sb3_support.with_discrete_actions(environment)
        return environment

    # The environment is wrapped before it is announced: a value-based run takes
    # a finite command set, and the line a caller reads should name the action
    # space the run really uses rather than the one the class hands out.
    environment = _build_environment(
        RobotSNAPEnv,
        algorithm=key,
        wrap=wrap if (plan is not None or sb3_support.ALGORITHMS[key]) else None,
        **options,
    )
    started = False
    try:
        environment.reset()
        started = True
        if plan is not None:
            print(f"curriculum: {plan.summary()}", flush=True)
        print(
            f"{scenario_id}: {timesteps} steps of {control_period:g} s of world time"
        )
        model = sb3_support.train(
            algo=key,
            timesteps=int(timesteps),
            save=save,
            env=environment,
            policy_kwargs=tuned,
            algo_kwargs=algorithm_options,
            callback=_curriculum_callback(plan),
        )
        if save:
            print(f"model written to {save}")
        return 0
    finally:
        _stop_session(environment, stop and started)
        environment.close()
        if not keep:
            scenario.delete(scenario_id, directory=directory)
