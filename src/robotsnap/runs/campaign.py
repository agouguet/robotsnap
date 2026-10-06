"""The ``benchmark`` run: a suite of scenarios, a policy, and the numbers it left.

What this run owns is the *binding*: which environment class a policy needs,
how a checkpoint becomes a callable that answers one action at a time, and how
a campaign is written where a later comparison can find it. The loop itself
lives in :mod:`robotsnap.analysis.campaign`, because a campaign is a reading of
the session rather than a way of driving it, and the two must not drift apart.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from robotsnap import scenario

from robotsnap.runs.presets import training_fields
from robotsnap.runs.session import (
    _announce,
    _environment_options,
    _scenario_directory,
    _stop_session,
)


def _scripted_action(environment: Any) -> Callable[[Any, Any], Any]:
    """The go-to-goal controller of this package, as a benchmark policy.

    It reads the world rather than the observation, so the same run measures a
    suite without a checkpoint and without pretending the baseline is a learned
    policy: a benchmark needs a floor to compare against.
    """
    from robotsnap.runs.controllers import go_to_goal

    def act(observation: Any, info: Any) -> Any:
        return go_to_goal(environment.world())

    return act


def _random_action(environment: Any) -> Callable[[Any, Any], Any]:
    """Actions drawn from the environment's own space, the other floor."""

    def act(observation: Any, info: Any) -> Any:
        return environment.action_space.sample()

    return act


def _neutral_action(environment: Any) -> Any:
    """A no-op action of the environment's own space, derived from it.

    An externally driven episode ignores the value - the velocity comes from the
    other process - but the campaign loop still calls ``step(action)``, so the
    action has to convert without raising. Zeros shaped like a continuous
    command, and the first index for a discrete one.
    """
    from gymnasium import spaces

    space = environment.action_space
    if isinstance(space, spaces.Discrete):
        return 0
    shape = getattr(space, "shape", None)
    if shape is None:
        return 0

    import numpy as np

    return np.zeros(shape, dtype=getattr(space, "dtype", np.float32))


def _external_action(environment: Any) -> Callable[[Any, Any], Any]:
    """The action a campaign hands an externally driven episode.

    The environment writes no command in this mode, so the action is a value of
    the right shape that ``step`` accepts and the measured world ignores.
    """
    neutral = _neutral_action(environment)

    def act(observation: Any, info: Any) -> Any:
        return neutral

    return act


@dataclass(frozen=True)
class _Policy:
    """What a campaign needs to know before it builds the environment.

    A checkpoint names its own writer, and the writer decides everything else:
    which environment class the run must build (a method brings its own
    observation and action space), whether the session has to hand out the
    discrete command set, and what the saved campaign says it measured. All of
    that has to be known *before* the environment exists, which is why it is
    resolved here and the action callable - the one part that needs a built
    environment - is a builder rather than a function.
    """

    environment_class: type
    make_act: Callable[[Any], Callable[[Any, Any], Any]] | None
    label: str
    discrete: bool = False
    external_control: bool = False


def _policy_and_environment(
    *,
    policy: str,
    load: str | None,
    method: str | None,
    algo: str,
) -> _Policy:
    """Resolve ``--policy``/``--load``/``--method`` into what a campaign runs."""
    from robotsnap.envs import RobotSNAPEnv

    key = str(policy).strip().lower()
    if key not in ("scripted", "random", "checkpoint", "external", "ros2"):
        raise ValueError(
            f"unknown policy {policy!r}: use 'scripted', 'random', 'external' "
            "(alias 'ros2'), or --load with a checkpoint"
        )
    if key == "random":
        return _Policy(RobotSNAPEnv, None, "random")
    if key == "scripted":
        return _Policy(RobotSNAPEnv, None, "scripted")
    if key in ("external", "ros2"):
        return _Policy(RobotSNAPEnv, _external_action, "external", external_control=True)

    if not load:
        raise ValueError("--policy checkpoint needs --load <file>")

    from robotsnap.runs.inference_torch import _checkpoint_kind

    kind = _checkpoint_kind(load)
    if method is not None or kind == "social":
        from robotsnap.models import ALGORITHMS, load_agent
        from robotsnap.runs.inference_social import social_checkpoint_algorithm

        name = (
            str(method).strip().lower()
            if method is not None
            else social_checkpoint_algorithm(load)
        )
        if name not in ALGORITHMS:
            raise ValueError(
                f"unknown method {method!r}: {', '.join(sorted(ALGORITHMS))} "
                "are what robotsnap.models registers"
            )

        agent = load_agent(load)

        def make_act(environment: Any) -> Callable[[Any, Any], Any]:
            def act(observation: Any, info: Any) -> Any:
                return agent.act(observation, greedy=True)

            return act

        return _Policy(ALGORITHMS[name][0], make_act, f"method:{name}:{load}")

    if kind == "policy":
        from robotsnap.rl import policy as policy_support

        network, norm, meta = policy_support.load_policy(load)

        def make_act(environment: Any) -> Callable[[Any, Any], Any]:
            import torch

            size = int(environment.observation_space.shape[0])
            if size != int(meta["observation_size"]):
                raise ValueError(
                    f"{load} was trained on a {meta['observation_size']}-value "
                    f"observation, but this run hands out {size}; check "
                    "--observations against the run that wrote the checkpoint"
                )

            def act(observation: Any, info: Any) -> Any:
                with torch.no_grad():
                    tensor = torch.from_numpy(norm.normalise(observation)).unsqueeze(0)
                    mean, _std = network(tensor)
                    return mean.detach().numpy()[0]

            return act

        return _Policy(RobotSNAPEnv, make_act, f"policy:{load}")

    from robotsnap.rl import sb3 as sb3_support

    reading = None if str(algo).strip().lower() in ("", "auto") else algo
    # The *rule* decides the action set, not the checkpoint's own space: that is
    # the table this package trains and plays by, and it keeps a value-based
    # run on the discrete grid even if the file is read without its marker.
    rule = sb3_support._algorithm_from_checkpoint(load, algo=reading)
    discrete = bool(sb3_support.ALGORITHMS[rule])
    model = sb3_support.load_model(load, algo=rule)

    def make_act(environment: Any) -> Callable[[Any, Any], Any]:
        def act(observation: Any, info: Any) -> Any:
            return model.predict(observation, deterministic=True)[0]

        return act

    return _Policy(RobotSNAPEnv, make_act, f"sb3:{load}", discrete=discrete)


def run_scenarios(
    *,
    suite: str = "basic",
    episodes: int = 5,
    policy: str = "scripted",
    load: str | None = None,
    method: str | None = None,
    algo: str = "auto",
    name: str | None = None,
    out: str | None = None,
    seed: int | None = None,
    max_steps: int = 1000,
    control_period: float = 0.2,
    seconds: float = 120.0,
    scenario_id: str = "python_benchmark_demo",
    scenario_fields: Mapping[str, Any] | None = None,
    client: Any = None,
    port: int = 10000,
    host: str = "0.0.0.0",
    robot: str | None = None,
    wait_timeout: float = 30.0,
    unity_project: str | None = None,
    time_scale: float | None = None,
    pacing: str = "lockstep",
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
    """Run a suite of scenarios with one policy and save what it measured.

    ``episodes`` is the number of episodes *per scenario*: a suite of seven
    scenarios with five episodes each is thirty-five episodes, and the campaign
    document keeps them apart per scenario as well as averaged over the whole
    run, because a method that fails one scenario and wins the others is not
    the same as one that is mediocre everywhere.
    """
    from robotsnap.analysis import campaign as campaign_support
    from robotsnap.analysis import suites as suite_support

    from robotsnap.runs.session import _build_environment

    if scenario_fields is None:
        scenario_fields = training_fields()
    directory = _scenario_directory(unity_project)
    if directory is None:
        return 2

    try:
        scenarios = suite_support.scenarios_for(suite)
    except suite_support.SuitesError as error:
        print(str(error), file=sys.stderr)
        return 2

    try:
        chosen = _policy_and_environment(
            policy=policy, load=load, method=method, algo=algo
        )
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2

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
        external_control=chosen.external_control,
    )

    def wrap(environment: Any) -> Any:
        """The discrete command set, when the checkpoint was trained on it."""
        if not chosen.discrete:
            return environment
        from robotsnap.rl import sb3 as sb3_support

        return sb3_support.with_discrete_actions(environment)

    environment = _build_environment(
        chosen.environment_class,
        algorithm="benchmark",
        method=method,
        wrap=wrap if chosen.discrete else None,
        **options,
    )
    started = False
    try:
        try:
            if chosen.make_act is None:
                if chosen.label == "random":
                    act = _random_action(environment)
                else:
                    act = _scripted_action(environment)
            else:
                act = chosen.make_act(environment)
        except ValueError as error:
            # A checkpoint that does not match the environment this run built:
            # said before an episode is played, not as a wrong-looking score.
            print(str(error), file=sys.stderr)
            return 2
        label = "scripted go-to-goal" if chosen.label == "scripted" else chosen.label

        def read_episode(scenario_name: str, index: int) -> Mapping[str, Any] | None:
            """The episode Unity just finished, the same list its dashboard shows."""
            episodes_of_session = environment.client.episodes
            episodes = episodes_of_session() if callable(episodes_of_session) else None
            if not episodes:
                return None
            return episodes[-1]

        print(
            f"benchmark {name or suite}: {len(scenarios)} scenarios x {episodes} "
            f"episodes, policy {label}"
        )
        document = campaign_support.run_campaign(
            environment=environment,
            scenarios=scenarios,
            episodes_per_scenario=episodes,
            act=act,
            name=name or suite,
            suite=suite,
            policy=label,
            seed=seed,
            max_steps=max_steps,
            episode_reader=read_episode,
            on_episode=_print_episode,
        )
        started = True
    except KeyboardInterrupt:
        print("benchmark interrupted", file=sys.stderr)
        return 130
    finally:
        _stop_session(environment, stop and started)
        environment.close()
        if not keep:
            scenario.delete(scenario_id, directory=directory)

    _print_campaign(document)
    if out:
        path = campaign_support.save_campaign(document, out)
        print(f"campaign written to {path}")
    return 0


def _print_episode(index: int, scenario_name: str, summary_line: str) -> None:
    """One line per finished episode, so a long campaign shows its progress."""
    print(f"[{scenario_name}] episode {index}: {summary_line}", flush=True)


def _print_campaign(campaign: Any) -> None:
    """The per-scenario and whole-suite numbers a campaign exists to report."""
    from robotsnap.analysis.campaign import format_summary

    for result in campaign.results:
        print(f"{result.scenario:>24}  {format_summary(result.summary)}")
    print(f"{'ALL':>24}  {format_summary(campaign.summary)}")


def compare(paths: Sequence[str]) -> int:
    """Print the comparison of two or more saved campaigns, without running one."""
    from robotsnap.analysis import campaign as campaign_support

    if len(paths) < 2:
        print("--compare needs at least two campaign files", file=sys.stderr)
        return 2

    campaigns = [campaign_support.load_campaign(path) for path in paths]
    for index in range(1, len(campaigns)):
        comparison = campaign_support.compare_campaigns(
            campaigns[index - 1], campaigns[index]
        )
        print(campaign_support.format_comparison(comparison))
    return 0
