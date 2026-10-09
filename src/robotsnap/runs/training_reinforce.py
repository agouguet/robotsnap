"""The REINFORCE training run, and the pieces of its loop.

The package's own learner: a numpy-in, numpy-out loop with PyTorch imported
inside the call. A run of a few dozen episodes is a smoke test of the wiring,
not a result.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from typing import Any

from robotsnap.rl import policy

from robotsnap.runs.presets import training_fields
from robotsnap.runs.session import (
    _announce,
    _announce_session,
    _apply_curriculum,
    _curriculum,
    _environment_options,
    _remove_scratch_scenario,
    _scenario_directory,
    _stop_session,
)


def run_training(
    *,
    episodes: int = 20,
    gamma: float = 0.99,
    lr: float = 3e-4,
    hidden: int = 64,
    entropy: float = 0.01,
    max_steps: int = 1000,
    seconds: float = 120.0,
    curriculum: str | None = None,
    control_period: float = 0.2,
    time_scale: float | None = None,
    pacing: str = "free",
    scenario_id: str = "default",
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
    """Train a REINFORCE policy on the RobotSNAP task.

    The environment hands out numpy observations and takes numpy actions, so the
    loop is agnostic about the framework; PyTorch is imported here, inside the
    call, and only this run needs it. A run of a few dozen episodes is a smoke
    test of the wiring, not a result.
    """
    from robotsnap.envs import RobotSNAPEnv

    torch, run_episode = _reinforce_components()
    Policy = policy.policy_class()
    RunningNorm = policy.running_norm_class()

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
    plan = _curriculum(curriculum)
    environment = _apply_curriculum(environment, plan)
    _announce(environment, algorithm="reinforce")
    observation_size = environment.observation_space.shape[0]
    action_size = environment.action_space.shape[0]
    # Whether the run ever reached the session: a checkpoint that cannot be
    # resumed leaves the scene where it was found, it does not freeze it.
    started = False
    try:
        try:
            if load:
                # A continuation picks up both the weights and the normalisation
                # the network was trained with; the checkpoint carries the
                # width, so the environment's observation is checked against it.
                network, norm, meta = policy.load_policy(
                    load, observation_size=observation_size
                )
                depth = meta["hidden"]
            else:
                network = Policy(observation_size, action_size, hidden)
                norm = RunningNorm(observation_size)
                depth = hidden
            optimizer = torch.optim.Adam(network.parameters(), lr=lr)
        except ValueError as exc:
            print(f"cannot start the training run: {exc}", file=sys.stderr)
            return 2
        reached = 0
        started = True
        _announce_session(environment.client, host=host, port=port)
        print(
            f"observation {environment.observation_space.shape}"
            f" -> action {environment.action_space.shape}"
        )
        if plan is not None:
            print(f"curriculum: {plan.summary()}", flush=True)
        for episode in range(1, episodes + 1):
            steps, rewards, log_probs, entropies, returns, info = run_episode(
                environment,
                network,
                norm,
                gamma=gamma,
                max_steps=max_steps,
                render=render,
            )
            # The returns are centred on their own mean: that is the baseline,
            # and it costs one line instead of a critic network.
            advantage = returns - returns.mean()
            if returns.numel() > 1 and float(returns.std()) > 0.0:
                advantage = advantage / (returns.std() + 1e-8)
            loss = -(
                torch.stack(
                    [log_prob * value for log_prob, value in zip(log_probs, advantage)]
                ).sum()
                + entropy * torch.stack(entropies).sum()
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            optimizer.step()

            reached += 1 if info["goal_reached"] else 0
            print(
                f"episode {episode:3d} | reward {sum(rewards):+8.2f}"
                f" | steps {steps:4d} | {info['episode_seconds']:5.1f} s"
                f" | goal {info['distance_to_goal'] if info['distance_to_goal'] is None else round(info['distance_to_goal'], 2)}"
                f" | loss {float(loss.detach()):+7.3f}"
                + ("" if plan is None else f" | curriculum {plan.summary()}")
            )
        print(f"{reached}/{episodes} episodes reached the goal")
        if save:
            policy.save_policy(
                save,
                network,
                norm,
                observation_size=observation_size,
                hidden=depth,
                observation_names=environment.observation_names,
                actions=action_size,
            )
            print(f"weights written to {save}")
        return 0
    finally:
        _stop_session(environment, stop and started)
        environment.close()
        if not keep:
            _remove_scratch_scenario(environment, scenario_id, directory)


def _reinforce_components():
    """The training-loop pieces of :func:`run_training`, built on first use.

    Imported and defined here rather than at module import so that ``runs``,
    and the command line over it, stays usable without PyTorch: only the
    training run calls this. The network and the normaliser now live in
    :mod:`robotsnap.rl.policy`, so a checkpoint and a run agree on their shape.
    """
    import numpy as np
    import torch

    def run_episode(
        environment, policy, norm, *, gamma: float, max_steps: int, render: bool
    ):
        """One episode: collect a trajectory, and the returns of each step."""
        observation, info = environment.reset()
        log_probs: list[Any] = []
        entropies: list[Any] = []
        rewards: list[float] = []
        steps = 0

        while True:
            tensor = torch.from_numpy(norm(observation))
            distribution = policy.distribution(tensor)
            action = distribution.sample()
            log_probs.append(distribution.log_prob(action).sum())
            entropies.append(distribution.entropy().sum())

            observation, reward, terminated, truncated, info = environment.step(
                action.detach().numpy()
            )
            rewards.append(float(reward))
            steps += 1
            if render:
                environment.render()
            if terminated or truncated or steps >= max_steps:
                break

        returns = []
        running = 0.0
        for reward in reversed(rewards):
            running = reward + gamma * running
            returns.append(running)
        returns.reverse()
        return (
            steps,
            rewards,
            log_probs,
            entropies,
            torch.tensor(returns, dtype=torch.float32),
            info,
        )

    return torch, run_episode
