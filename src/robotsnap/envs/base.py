"""A Gymnasium environment over a live RobotSNAP session.

:class:`RobotSNAPEnv` composes five short mixins, one file each:

- :mod:`robotsnap.envs.task` - ``TaskMixin``: the task hooks ``task``, ``is_collision``, ``reward``, ``terminated``, ``truncated``, ``stalled``, ``action_to_command`` and ``info``.
- :mod:`robotsnap.envs.features` - ``FeaturesMixin``: the ``*_features`` blocks an observation is assembled from.
- :mod:`robotsnap.envs.spaces` - ``SpacesMixin``: the action and observation spaces, and ``observation``.
- :mod:`robotsnap.envs.pacing` - ``PacingMixin``: holding a step to one control period of simulated time.
- :mod:`robotsnap.envs.session` - ``SessionMixin`` and ``RobotSNAPEnvError``: launching the scenario, waiting for the world, counting an episode's clocks.

To write your own environment, replace a method of ``TaskMixin`` or ``FeaturesMixin`` on a subclass; the simulator's side stays where it is. One episode is one navigation problem, and a ``step`` covers ``control_period`` seconds of simulated time - ``control_period / time_scale`` seconds of wall time.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:  # the package itself has no dependency on gymnasium; this module has one.
    import gymnasium
except ImportError as exc:  # pragma: no cover - exercised by the extras, not the suite
    raise ImportError(
        "robotsnap.envs needs gymnasium: pip install 'RobotSNAP[env]'"
    ) from exc

from robotsnap import topics
from robotsnap.client import RobotSNAPClient
from robotsnap.envs.features import FeaturesMixin
from robotsnap.envs.observation import MAX_NEIGHBOUR_SPEED, ObservationPart, resolve_observation_parts, spec_parameters
from robotsnap.envs.pacing import FREE, LOCKSTEP, PacingMixin, _PACING_MODES, _lockstep_grace
from robotsnap.envs.session import RobotSNAPEnvError, SessionMixin
from robotsnap.envs.spaces import SpacesMixin
from robotsnap.envs.task import TaskMixin, TaskState
from robotsnap.envs.world import World, read_world

__all__ = ["RobotSNAPEnv", "RobotSNAPEnvError", "TaskState", "DEFAULT_CONTROL_PERIOD", "MAX_AGENT_SPEED", "Observation"]

#: Seconds of simulated time one :meth:`RobotSNAPEnv.step` covers, by default.
DEFAULT_CONTROL_PERIOD = 0.1
#: Bound used for the velocity of a neighbour in the observation space.
MAX_AGENT_SPEED = MAX_NEIGHBOUR_SPEED

#: What :meth:`RobotSNAPEnv.reset` and :meth:`RobotSNAPEnv.step` hand back: one flat vector, or one array per part when the observation is structured.
Observation = Any


class RobotSNAPEnv(TaskMixin, FeaturesMixin, SpacesMixin, PacingMixin, SessionMixin, gymnasium.Env):
    """A single-robot navigation environment over the RobotSNAP bridge.

    The environment owns the bridge unless a ``client`` is given, in which case
    it uses that one and leaves its lifetime alone.

    ``scenario`` is the id of the scenario to run, and ``scenario_fields`` (a
    document for :func:`robotsnap.scenario.build`) makes the environment write it
    first, so one script can hold both the world and the task.

    ``robot`` names the robot to drive; left as ``None`` every command goes to
    the session's primary robot on the unprefixed topics it has always used.

    ``control_period`` is the simulated seconds one :meth:`step` covers: the
    environment holds for ``control_period / time_scale`` seconds of wall time.
    """

    metadata = {"render_modes": ["human"], "render_fps": 10}

    #: Parts this class adds to its own specs, resolvable by name in
    #: ``observations`` next to the registered ones. A subclass that writes a
    #: feature of its own sets it to a tuple of
    #: :class:`~robotsnap.envs.observation.ObservationPart` - before calling
    #: ``super().__init__`` when the part reads a field of the instance.
    extra_observation_parts: tuple[ObservationPart, ...] = ()

    def __init__(
        self,
        *,
        scenario: str | None = None,
        scenario_fields: Mapping[str, Any] | None = None,
        scenario_directory: str | Path | None = None,
        robot: str | None = None,
        client: RobotSNAPClient | None = None,
        host: str = "0.0.0.0",
        port: int = 10000,
        control_period: float = DEFAULT_CONTROL_PERIOD,
        wait_timeout: float = 30.0,
        max_episode_seconds: float = 60.0,
        goal_radius: float = 0.5,
        collision_distance: float = 0.25,
        occupancy_threshold: int = 50,
        lidar_bins: int = 48,
        max_agents: int = 8,
        max_humans: int = 8,
        max_linear: float = 1.0,
        max_angular: float = 1.0,
        time_scale: float | None = None,
        fixed_timestep: float | None = None,
        maximum_delta_time: float | None = None,
        pacing: str = FREE,
        external_control: bool = False,
        stall_timeout: float = 2.0,
        control_mode: str = "ros",
        goal_reward: float = 10.0,
        progress_reward: float = 1.0,
        collision_penalty: float = 10.0,
        out_of_bounds_penalty: float = 5.0,
        time_penalty: float = 0.1,
        reload_on_reset: bool = False,
        observations: Any = None,
        observation_params: Mapping[str, Mapping[str, Any]] | None = None,
        observation_structure: str = "flat",
        render_mode: str | None = None,
    ):
        """Build the environment; ``control_period`` is simulated seconds per step.

        ``time_scale``, when given, is the ``Time.timeScale`` the episode runs at
        and must be strictly positive; left as ``None`` the environment paces
        with the scale the session reports. ``fixed_timestep`` and
        ``maximum_delta_time`` are the physics step and the most simulation one
        drawn frame may catch up on, both left to the simulator when omitted.

        ``pacing`` is ``"free"`` or ``"lockstep"``: free running the world moves
        while the policy thinks, lockstep asks the simulator to spend exactly
        ``control_period`` of its own seconds per step and stop in between - see
        :meth:`_end_episode`.

        ``observations`` picks and orders the parts of the observation (``None``
        keeps ``robot, goal, agents, lidar``); ``observation_params`` carries
        their parameters separately and ``observation_structure`` is ``"flat"``
        or ``"dict"``. ``max_agents``, ``max_humans`` and ``lidar_bins`` stay the
        sizes of the parts that use them, and a part's own ``max``/``bins`` wins.
        ``scenario_directory`` is where ``scenario_fields`` is written.

        ``external_control`` is for an episode whose velocity is owned outside
        Python - a ROS2 node of its own publishing the bridge's ``cmd_vel`` - so
        a step releases the pacing and reads the world without writing a command
        that would race the one the external policy sends.
        """
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unsupported render_mode {render_mode!r}")
        if control_period <= 0.0:
            raise ValueError("control_period must be positive")
        if time_scale is not None:
            scale = float(time_scale)
            if not math.isfinite(scale) or scale <= 0.0:
                raise ValueError("time_scale must be positive")
        for name, value in (("fixed_timestep", fixed_timestep),
                            ("maximum_delta_time", maximum_delta_time)):
            if value is not None and (not math.isfinite(float(value)) or float(value) <= 0.0):
                raise ValueError(f"{name} must be positive")
        if pacing not in _PACING_MODES:
            raise ValueError(
                f"pacing is {FREE!r} or {LOCKSTEP!r}, got {pacing!r}"
            )
        if max_agents < 0 or max_humans < 0 or lidar_bins <= 0:
            raise ValueError(
                "max_agents and max_humans must not be negative, lidar_bins must be positive"
            )
        if observation_structure not in ("flat", "dict"):
            raise ValueError(
                f"observation_structure is 'flat' or 'dict', got {observation_structure!r}"
            )

        self.scenario = scenario
        self.scenario_fields = dict(scenario_fields or {})
        self.scenario_directory = scenario_directory
        self.robot = robot
        self.control_period = float(control_period)
        self.wait_timeout = float(wait_timeout)
        self.max_episode_seconds = float(max_episode_seconds)
        self.goal_radius = float(goal_radius)
        self.collision_distance = float(collision_distance)
        self.occupancy_threshold = int(occupancy_threshold)
        self.lidar_bins = int(lidar_bins)
        self.max_agents = int(max_agents)
        self.max_humans = int(max_humans)
        self.max_linear = float(max_linear)
        self.max_angular = float(max_angular)
        self.time_scale = None if time_scale is None else float(time_scale)
        self.fixed_timestep = None if fixed_timestep is None else float(fixed_timestep)
        self.maximum_delta_time = (
            None if maximum_delta_time is None else float(maximum_delta_time)
        )
        self.pacing = str(pacing)
        self.external_control = bool(external_control)
        self.stall_timeout = float(stall_timeout)
        self.control_mode = str(control_mode)
        self.goal_reward = float(goal_reward)
        self.progress_reward = float(progress_reward)
        self.collision_penalty = float(collision_penalty)
        self.out_of_bounds_penalty = float(out_of_bounds_penalty)
        self.time_penalty = float(time_penalty)
        self.reload_on_reset = bool(reload_on_reset)
        self.render_mode = render_mode

        self.observations = observations
        self.observation_structure = observation_structure
        self._observation_params = {
            str(name): dict(value or {}) for name, value in (observation_params or {}).items()
        }
        # The spec carries parameters too, and they win over the ones handed in
        # separately: a block written next to the part it sizes is the more
        # specific statement of the two. Both have to size the environment
        # before the parts are resolved, so they are merged here once.
        for name, own in spec_parameters(observations).items():
            self._observation_params.setdefault(name, {}).update(own)
        self._apply_observation_sizes()
        self.observation_parts: tuple[ObservationPart, ...] = resolve_observation_parts(
            self, observations, self._observation_params
        )
        self.observation_names = tuple(part.name for part in self.observation_parts)
        self.observation_slices = self._observation_layout()

        self._owns_client = client is None
        self.client = client if client is not None else RobotSNAPClient(host=host, port=port)
        self._episode_ready = False
        self._task = TaskState(
            distance_to_goal=None,
            bearing_to_goal=None,
            goal_reached=False,
            collision=False,
            out_of_bounds=False,
            min_lidar=None,
            nearest_agent_distance=None,
            elapsed_seconds=0.0,
            wall_seconds=0.0,
            steps=0,
        )
        self._elapsed_seconds = 0.0
        self._wall_seconds = 0.0
        # The simulator's own clock at the instant this episode began, read on the first world read and not
        # guessed: see _advance_episode_clock.
        self._episode_clock_start: float | None = None
        # The world's clock as the last read saw it, which is where a lockstep period starts counting.
        self._last_sim_time: float | None = None
        # When that clock last moved, so a world that stopped - a paused session, a disabled simulator - ends
        # the episode instead of letting a training loop step a world that is not there.
        self._last_sim_change: float | None = None
        # Wall seconds of the step just taken, the fallback the episode clock uses on a scene with no clock.
        self._last_wall_delta = 0.0
        self._steps = 0
        self._last_command: tuple[float, float] = (0.0, 0.0)
        # Wall seconds the pacing spent beyond the wait it asked for, measured
        # step by step; see :meth:`_wait_control_period`.
        self._pacing_latency = 0.0
        # Whether this environment holds the world on purpose because an episode ended: a lockstep gate
        # is left stopped at the end of a mission, and close() must not start it again. See _end_episode.
        self._world_held = False
        # Whether a lockstep episode has already been told it has no clock to verify its steps against.
        self._warned_no_clock = False
        # Time scale the session last reported, refreshed by every world read:
        # the fallback that paces a step when the caller set no scale of its own.
        self._scale_from_world: float | None = None
        self._scenario_written = False
        self._streams: dict[str, bool] = {"world": False, "pose": False}
        self._pygame = None
        self._viewer = None

        self.action_space = self.make_action_space()
        self.observation_space = self.make_observation_space()

    # -- the Gymnasium loop -------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: Mapping[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        """Start an episode: make sure the scenario runs, then read the world.

        With no scenario applied yet the configured one is launched; otherwise
        the current scenario is re-applied, which is the cheaper reset and the
        one that gives every episode the same start. ``options`` can carry
        ``scenario`` to run another one for this episode and ``launch`` to force
        a full reload.
        """
        super().reset(seed=seed)
        chosen = dict(options or {})
        if seed is not None:
            self.client.set_random_seed(int(seed))

        self._prepare_session(seed=seed, options=chosen)
        self._begin_episode()

        world = self.world()
        # A free-running episode's epoch is this read - the world as it stands once the setup has stopped
        # touching it, and before any action - so the world time between here and the first step counts
        # against the budget like the periods the steps spend. A lockstep episode already has its epoch, from
        # the held world _begin_episode waited for, and this read only confirms it.
        self._advance_episode_clock(world)
        self._task = self.task(world)
        self._episode_ready = True
        return self.observation(world, self._task), self.info(world, self._task)

    def step(
        self, action: Any
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        """Apply one action for one control period, then read the result.

        The period is ``control_period`` seconds of simulated time, not of wall
        time. The command goes out first, then the environment waits for the
        period and, for one more period at most, for a state and an odometry
        message newer than the command. A session that publishes slower than
        that is not waited for: the step returns with ``info["fresh"]`` false.

        The step that ends the episode - on the goal, on a failure or on the
        time budget - also stops the world, so a lockstep session is left held.
        See :meth:`_end_episode`.
        """
        if not self._episode_ready:
            raise RobotSNAPEnvError("call reset() before step()")

        command = self.action_to_command(action)
        if not self.external_control:
            if not self.client.send_cmd_vel(*command, robot=self.robot):
                raise RobotSNAPEnvError(
                    f"the command could not be sent: {self.client.last_error}"
                )
        self._last_command = command

        if self.pacing == LOCKSTEP:
            # The command went out first: the release is what starts the world, so the period that follows is
            # driven by the action this step is answering for and not by the one before it.
            self.client.release_pacing()

        wall_delta, fresh = self._wait_control_period()
        world = self.world()
        self._wall_seconds += wall_delta
        self._last_wall_delta = wall_delta
        self._advance_episode_clock(world)
        self._steps += 1

        task = self.task(world)
        observation = self.observation(world, task)
        reward = self.reward(self._task, task, command, world)
        terminated = self.terminated(task)
        truncated = False if terminated else self.truncated(task)
        if terminated or truncated:
            self._end_episode()
        information = self.info(world, task)
        information["fresh"] = fresh
        self._task = task
        return observation, reward, terminated, truncated, information

    def render(self) -> None:
        """Draw the session in a pygame window, on ``render_mode="human"``.

        The window is the one :mod:`robotsnap.viewer` draws, and it follows the
        primary robot. Nothing is drawn for any other render mode, and the window
        closes with Esc, ``q`` or its close button.
        """
        if self.render_mode != "human":
            return None
        if self._viewer is None:
            import pygame

            from robotsnap.viewer import Viewer

            pygame.init()
            self._pygame = pygame
            self._viewer = Viewer(pygame, self.client)
        if not self._viewer.handle_events():
            return None
        self._viewer.update()
        return None

    def close(self) -> None:
        """Stop the bridge this environment started, and close its window.

        A client handed to the constructor is left running: it belongs to the
        caller. A lockstep session the caller stopped mid-episode gets its clock
        back; a session an episode ended on stays held, so the world does not
        restart around a mission that is over. See :meth:`_end_episode`.
        """
        if self._pygame is not None:
            try:
                self._pygame.display.quit()
                self._pygame.quit()
            finally:
                self._pygame = None
                self._viewer = None
        if self.pacing == LOCKSTEP and not self._world_held:
            # Leaving the session stopped would freeze the Unity scene for whoever comes next - and a frozen
            # world cannot even rebuild its scenario, so the next run would time out. The environment took the
            # clock away mid-episode; it gives it back. An episode that ended already holds the world for its
            # own sake, and starting it again here would be exactly what that hold exists to prevent.
            try:
                self.client.set_pacing(FREE)
            except Exception:
                pass
        if self._owns_client:
            self.client.stop()

    # -- reading the session ------------------------------------------------

    def world(self) -> World:
        """Read the session once, as a :class:`~robotsnap.envs.world.World`.

        The read is also where the environment learns the session's time scale,
        which :meth:`step` uses to pace the next control period; caching it here
        means a step never needs a second snapshot just for the scale.
        """
        world = read_world(self.client, robot=self.robot)
        self._scale_from_world = world.time_scale
        return world

    @property
    def odom_topic(self) -> str:
        """The ``/odom`` name of the robot this environment drives."""
        return topics.ODOM if self.robot is None else topics.robot_topic(self.robot, topics.ODOM)
