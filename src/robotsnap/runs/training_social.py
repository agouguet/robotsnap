"""The social-method training run, and the task it builds on the base env.

The class is what ``examples/my_env.py`` used to define: it inherits
:class:`robotsnap.envs.RobotSNAPEnv` and changes the observation and the reward,
which is what a custom task usually is.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from robotsnap.runs.controllers import _POLAR, _polar
from robotsnap.runs.inference_torch import _action_count
from robotsnap.runs.presets import training_fields
from robotsnap.runs.session import (
    _announce,
    _apply_curriculum,
    _curriculum,
    _environment_options,
    _remove_scratch_scenario,
    _scenario_directory,
    _stop_session,
)


def _social_nav_class():
    """Build the social task on top of the environment extra, on first use.

    The class is what ``examples/my_env.py`` used to define: it inherits
    :class:`robotsnap.envs.RobotSNAPEnv` and changes two things, which is what a
    custom task usually is - the **observation** (the goal and the closest
    person in polar form, as two parts of its own next to the sweep) and the
    **reward** (progress toward the goal, minus a price for walking too close to
    someone). The launching, the pacing, the reading of the streams and the
    reset stay in the base class.

    The observation is written with the parts of
    :mod:`robotsnap.envs.observation` rather than by overriding
    ``observation``: the polar goal and the polar neighbour are components this
    task owns, declared on itself with ``extra_observation_parts`` and named in
    its default spec. A caller that asks for other parts - ``--observations
    pose,goal`` - gets them, so the social default is a starting point and not a
    cage.

    Everything the constructor adds is set *before* ``super().__init__`` so that
    :meth:`make_observation_space` can read it: the spaces are built in the base
    constructor, from the fields the subclass has already set.
    """
    import numpy as np

    from robotsnap.envs import RobotSNAPEnv
    from robotsnap.envs.observation import ObservationPart

    class SocialNavEnv(RobotSNAPEnv):
        """Reach the goal while keeping a social distance from the crowd."""

        #: Distances are divided by this one, so the observation stays bounded.
        MAX_DISTANCE = 20.0

        #: The parts this task observes by default, the two of its own first.
        DEFAULT_PARTS = ("goal_polar", "nearest_agent_polar", "lidar")

        def __init__(
            self,
            *,
            social_distance: float = 1.2,
            proximity_penalty: float = 2.0,
            observations: Any = None,
            **kwargs,
        ):
            self.social_distance = float(social_distance)
            self.proximity_penalty = float(proximity_penalty)
            # Declared before the base constructor, which resolves the parts.
            self.extra_observation_parts = self._own_parts()
            if observations is None:
                observations = self.DEFAULT_PARTS
            super().__init__(observations=observations, **kwargs)

        def _own_parts(self):
            """The two polar parts of this task, reachable by name in a spec."""

            def goal_polar(context):
                return _polar(
                    context.task.distance_to_goal,
                    context.task.bearing_to_goal,
                    self.MAX_DISTANCE,
                )

            def neighbour_polar(context):
                nearest = context.world.nearest_agent()
                if nearest is None:
                    return np.zeros(_POLAR, dtype=np.float32)
                return _polar(nearest.distance, nearest.bearing, self.MAX_DISTANCE)

            return tuple(
                ObservationPart(
                    name=name,
                    size=_POLAR,
                    low=np.array([0.0, -1.0, -1.0], dtype=np.float32),
                    high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
                    build=build,
                    doc=doc,
                )
                for name, build, doc in (
                    (
                        "goal_polar",
                        goal_polar,
                        "the goal in polar form: distance, cos bearing, sin bearing",
                    ),
                    (
                        "nearest_agent_polar",
                        neighbour_polar,
                        "the closest detected agent in polar form: distance, cos bearing, sin bearing",
                    ),
                )
            )

        def reward(self, previous, current, action, world) -> float:
            value = super().reward(previous, current, action, world)
            nearest = current.nearest_agent_distance
            if nearest is not None and nearest < self.social_distance:
                # Per step, so lingering beside someone costs more than passing.
                closeness = (self.social_distance - nearest) / self.social_distance
                value -= self.proximity_penalty * closeness
            return value

    return SocialNavEnv


def run_social_training(
    *,
    algo: str | None = None,
    episodes: int = 100,
    max_steps: int = 0,
    seed: int | None = None,
    max_neighbours: int = 8,
    social_parameters: Mapping[str, Any] | None = None,
    curriculum: str | None = None,
    seconds: float = 120.0,
    control_period: float = 0.2,
    time_scale: float | None = None,
    pacing: str = "free",
    scenario_id: str = "python_social_demo",
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
    """Train one named method of :mod:`robotsnap.models` on its own task.

    A method is not a learning rule: it brings its own observation, action set,
    reward and policy, and the catalogue is what says which environment and
    which agent a name means. Two methods of the literature can therefore be
    compared on the same simulator without either of them sharing a line with
    the other, which is the point of naming them separately.
    """
    from robotsnap.models import ALGORITHMS as SOCIAL_ALGORITHMS
    from robotsnap.models import METHOD_LEARNER, load_agent, make_agent
    from robotsnap.models.social import train as social_train

    # No method is named here on purpose: the catalogue is the one place that
    # knows which methods exist, so an omitted name means the first one it
    # registers rather than a spelling repeated in the core.
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

    plan = _curriculum(curriculum)

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
    environment = _apply_curriculum(environment, plan)
    # The learner is the method's own - a state value with a lookahead for
    # CADRL and SARL, an actor-critic for GA3C-CADRL - so the line a run prints
    # reads it off the catalogue instead of repeating one of them for all.
    _announce(environment, algorithm=METHOD_LEARNER.get(key, "dqn"), method=key)

    started = False
    try:
        environment.reset()
        started = True
        agent = load_agent(load) if load else make_agent(key, environment)
        print(
            f"{key} on {scenario_id}: {_action_count(environment)} actions, "
            f"up to {int(getattr(environment, 'max_neighbours', 0))} neighbours, "
            f"{episodes} episodes of {seconds:g} s of world time"
        )
        if plan is not None:
            print(f"curriculum: {plan.summary()}", flush=True)

        def log(line: str) -> None:
            """One episode line, with the stage it was played at."""
            print(line if plan is None else f"{line} | curriculum {plan.summary()}")

        history = social_train(
            environment,
            agent,
            episodes=int(episodes),
            max_steps=int(max_steps) or None,
            seed=seed,
            render=render,
            log=log,
        )
        arrived = sum(1 for outcome in history["outcome"] if outcome == "goal")
        print(f"{arrived}/{len(history['episode'])} episodes reached the goal")
        if save:
            print(f"agent written to {agent.save(save)}")
        return 0
    finally:
        _stop_session(environment, stop and started)
        environment.close()
        if not keep:
            _remove_scratch_scenario(environment, scenario_id, directory)
