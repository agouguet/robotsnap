"""How a command opens and closes a session with Unity.

Building the environment, the options every environment shares, where a
scenario file goes, and the stop button: the plumbing, not the runs.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from robotsnap import scenario
from robotsnap.runs.spec import describe_run


def _announce(environment: Any, *, algorithm: str, method: str | None = None) -> None:
    """Print the one line that says what this run really drives.

    The command line has a single ``--algo`` knob whose values do not all name
    the same kind of thing - see :mod:`robotsnap.runs.spec` - and a run that says out
    loud which environment, which observation, which action space and which
    reward it resolved cannot quietly differ from the one the caller thought
    they asked for. The line is read from the environment that was built, so it
    describes the run rather than a table that hopes to match it.
    """
    # Flushed on purpose: this is the line a redirected log has to show first, before whatever the session
    # says next, and it is the one a caller greps to know what a run actually drove.
    print(describe_run(environment, algorithm=algorithm, method=method), flush=True)


def _announce_session(client: Any, *, host: str, port: int) -> None:
    """Say where this run reaches the session, before anything waits on it.

    A run on the socket and a run on a ROS2 graph are the same code from here, so the line names the
    transport instead of printing a port a graph does not have. A client on a bridge says which one it
    holds; anything else is the socket this run would have bound.
    """
    label = getattr(getattr(client, "bridge", None), "session_label", None)
    where = label if label else f"bridge on {host}:{port}"
    print(f"{where} - press Play in Unity now")


def _build_environment(
    factory: Any,
    *,
    algorithm: str,
    method: str | None = None,
    wrap: Any = None,
    **options: Any,
) -> Any:
    """Build the environment, apply ``wrap`` if there is one, and announce the result.

    ``wrap`` is what an algorithm that needs a different action space does to
    the environment it was handed - ``dqn`` wants a finite set of commands where
    the class hands out a continuous box - and it is applied *before* the
    announcement so the line names the action space the run really uses.
    """
    environment = factory(**options)
    if wrap is not None:
        environment = wrap(environment)
    _announce(environment, algorithm=algorithm, method=method)
    return environment


def _environment_options(
    *,
    scenario_id: str,
    scenario_fields: Mapping[str, Any] | None,
    scenario_directory: Path | None,
    client: Any,
    port: int,
    host: str,
    robot: str | None,
    control_period: float,
    wait_timeout: float,
    time_scale: float | None,
    pacing: str,
    seconds: float,
    render: bool,
    observations: Any,
    observation_params: Mapping[str, Any] | None,
    observation_structure: str,
    max_agents: int,
    max_humans: int,
    lidar_bins: int,
    goal_radius: float,
    collision_distance: float,
    occupancy_threshold: int,
    max_linear: float,
    max_angular: float,
    reload_on_reset: bool,
    control_mode: str,
    goal_reward: float,
    progress_reward: float,
    collision_penalty: float,
    out_of_bounds_penalty: float,
    time_penalty: float,
    external_control: bool = False,
) -> dict[str, Any]:
    """The keyword arguments every environment of a run is built from, once.

    The three observation arguments are passed only when they were set: left at
    their defaults the constructor is asked for its own historical observation,
    rather than handed a value repeated here.

    ``scenario_directory`` is the one the run resolved from ``--unity-project``,
    so the environment writes the scenario where the run will delete it: a run
    that was pointed at another project must not write into the checkout the
    package would have found by itself.
    """
    options: dict[str, Any] = dict(
        scenario=scenario_id,
        scenario_fields=scenario_fields,
        scenario_directory=scenario_directory,
        client=client,
        port=port,
        host=host,
        robot=robot,
        control_period=control_period,
        wait_timeout=wait_timeout,
        time_scale=time_scale,
        pacing=pacing,
        max_episode_seconds=seconds,
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
        external_control=external_control,
        render_mode="human" if render else None,
    )
    if observations is not None:
        options["observations"] = observations
    if observation_params is not None:
        options["observation_params"] = observation_params
    if observation_structure != "flat":
        options["observation_structure"] = observation_structure
    return options


def _scenario_directory(unity_project: str | None) -> Path | None:
    """Where a scenario file goes, or ``None`` after saying what is missing."""
    try:
        return scenario.scenarios_dir(unity_project)
    except FileNotFoundError as exc:
        print(f"cannot find the Unity project: {exc}", file=sys.stderr)
        return None


def _stop_session(environment, stop: bool) -> None:
    """Stop the Unity session, when the run asked for it.

    Called before ``environment.close()``:
    :meth:`RobotSNAPClient.stop_simulation` is the application's red stop button
    - it pauses the clock, takes every agent out of the scene and leaves the
    application in standby (``Ready``), the map and the scenario kept, so a
    training run or an inference demo stops the simulation instead of carrying
    on once the Python side is gone. A run that does not ask for it is
    untouched.
    """
    if not stop:
        return
    stopped = environment.client.stop_simulation()
    print(f"simulation stopped : {stopped}")


def _remove_scratch_scenario(environment, scenario_id: str, directory) -> bool:
    """Remove the scenario file a run wrote, and only that one.

    A run that names a scenario the project already ships never wrote it: that
    file belongs to the project, and removing it on the way out would take a
    scenario away from the user. ``--keep`` and this guard are the two ways a
    file survives a run; the environment is what knows which happened.
    """
    if not getattr(environment, "wrote_scenario", False):
        return False
    return scenario.delete(scenario_id, directory=directory)


def _curriculum(spec: str | None) -> Any:
    """The curriculum a run asked for by name or by path, or ``None``.

    Kept here rather than in each training run because the three learners share
    one spelling for it, and the import is deliberately inside the call: a run
    that trains without a curriculum must not pay for the module.
    """
    if not spec:
        return None
    from robotsnap.rl.curriculum import Curriculum

    return Curriculum.load(spec)


def _apply_curriculum(environment: Any, curriculum: Any) -> Any:
    """Wrap ``environment`` so each reset starts the curriculum's current stage.

    A wrapper is all this is: the base environment keeps its scenario option -
    the curriculum only decides *which* scenario that is, and when to change it.
    """
    if curriculum is None:
        return environment
    from robotsnap.rl.curriculum_env import CurriculumEnv

    return CurriculumEnv(environment, curriculum)


def _open_viewer(client: Any):
    """Open the package's 2D window on ``client``, or ``None`` when pygame is missing.

    This is the same window a run draws through
    :meth:`~robotsnap.envs.RobotSNAPEnv.render`, for the one command that drives the client itself
    instead of stepping an environment. Returns a pair the caller pumps with :func:`_pump_viewer`
    and closes with :func:`_close_viewer`; ``None`` means the caller runs headless, which is what a
    machine without a display wants rather than a traceback.
    """
    try:
        import pygame
    except ImportError as exc:
        print(f"--viewer needs pygame ({exc}): pip install pygame", file=sys.stderr)
        return None

    from robotsnap.viewer import Viewer

    pygame.init()
    return pygame, Viewer(pygame, client)


def _pump_viewer(viewer: Any) -> bool:
    """Draw one frame of a window opened by :func:`_open_viewer`.

    Returns ``False`` once the window has been closed by hand, so a caller in a driving loop can
    decide whether to stop. A ``None`` viewer is headless and always ``True``.
    """
    if viewer is None:
        return True
    pygame, window = viewer
    if not window.handle_events():
        return False
    window.update()
    return True


def _close_viewer(viewer: Any) -> None:
    """Close the window opened by :func:`_open_viewer`, if there is one. Idempotent."""
    if viewer is None:
        return
    pygame, _window = viewer
    pygame.quit()
