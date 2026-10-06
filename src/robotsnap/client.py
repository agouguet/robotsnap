"""High-level client for a running RobotSNAP Unity bridge.

Unity speaks the ROS-TCP-Connector wire protocol as a TCP *client*, and the
Python :class:`~robotsnap.bridge.server.RobotSNAPBridge` is the server. This
module adds the script-facing façade on top of that bridge: typed commands on
``/simulation/control`` and JSON reads from ``/simulation/state``,
``/simulation/agents`` and ``/simulation/control_result``.

Nothing here needs ROS, ``rclpy`` or a running ROS2 graph. The transport is a
single TCP socket and every payload in either direction is a
``std_msgs/String`` whose body is a JSON object.

Typical use::

    from robotsnap.client import RobotSNAPClient

    with RobotSNAPClient(port=10000) as client:
        client.wait_until_ready()
        client.play()
        state = client.snapshot()
        client.set_human_velocities([(3, 1.0, 0.0)])
        client.send_cmd_vel(0.4, 0.0)

The topic names and their message types come from :mod:`robotsnap.topics`, so a
client and the bridge always agree on them.

The session snapshot carries a roster of robots. An unprefixed stream name
(``/cmd_vel``, ``/simulation/agents``) reaches the primary robot only, so a
robot addressed by id is reached through its own ``/robot_<id>/<topic>`` name::

    state = client.snapshot()
    client.robot("robot_2")
    client.agents(robot="robot_2")
    client.send_cmd_vel(0.4, 0.0, robot="robot_2")
    client.set_robot_goal(4.0, 0.0, robot="robot_2")

Every command method is meant to be called in a tight loop: the answer is
waited for with a bounded poll on the bridge's decoders (a counter and a short
sleep), never with a busy spin, and a missing answer returns ``None`` with
:attr:`RobotSNAPClient.last_error` set instead of raising.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from robotsnap import scenario as _scenario
from robotsnap import topics
from robotsnap.analysis import metrics as _metrics
from robotsnap.bridge import codec
from robotsnap.bridge.server import RobotSNAPBridge

__all__ = [
    "RobotSNAPClient",
    "open_client",
    "TRANSPORTS",
    "CONTROL_TOPIC",
    "CONTROL_RESULT_TOPIC",
    "STATE_TOPIC",
    "AGENTS_TOPIC",
    "METRICS_TOPIC",
    "COMMAND_POLL_INTERVAL",
    "DEFAULT_COMMAND_TIMEOUT",
    "SCENARIO_COMMAND_TIMEOUT",
    "SCENARIO_SETTLE_TIMEOUT",
]

#: The transports a session can be reached over. ``tcp`` is the bridge's own socket, ``ros2`` is a
#: ROS2 graph the endpoint is already serving; the rest of the package is written against the same
#: surface either way.
TRANSPORTS = ("tcp", "ros2")

#: Unity subscribes to this topic for every command, crowd commands included.
CONTROL_TOPIC = topics.SIMULATION_CONTROL
#: Unity publishes one acknowledgement per command here.
CONTROL_RESULT_TOPIC = topics.SIMULATION_CONTROL_RESULT
#: Unity publishes the whole simulation state here as JSON.
STATE_TOPIC = topics.SIMULATION_STATE
#: Unity publishes every agent, in the robot frame, here as JSON.
AGENTS_TOPIC = topics.SIMULATION_AGENTS
#: Unity publishes one finished episode here as JSON; see :mod:`robotsnap.analysis.metrics`.
METRICS_TOPIC = _metrics.METRICS_TOPIC

_STRING = topics.STRING_TYPE

#: Sleep between two polls of the bridge decoders while waiting for a result.
COMMAND_POLL_INTERVAL = 0.005
#: Default time budget for one command acknowledgement, in seconds.
DEFAULT_COMMAND_TIMEOUT = 5.0
#: Time budget for the two commands that rebuild the scene, in seconds. Loading a
#: scenario destroys the environment, spawns the crowd and rebuilds the navigation
#: grid on Unity's main thread, and the acknowledgement is only published once all
#: of that is done, so the light-command budget above is far too short for it.
SCENARIO_COMMAND_TIMEOUT = 180.0
#: Time budget for a scenario to be *built*, in seconds: the load is queued behind one already running, or the
#: map, the NavMesh, the human pool and the spawn each take their turn before the world reports the scenario.
SCENARIO_SETTLE_TIMEOUT = 180.0


class RobotSNAPClient:
    """Façade over a :class:`~robotsnap.bridge.server.RobotSNAPBridge`.

    The client owns the command surface described in
    ``docs/robotsnap-unity-contract.md``: it serialises one JSON body per command
    on ``/simulation/control`` and returns the parsed
    ``/simulation/control_result`` acknowledgement.

    When ``bridge`` is ``None`` a bridge is created on ``host``/``port`` and
    started if ``autostart`` is true. A caller-supplied bridge that is already
    running is used as-is and is never stopped by :meth:`stop`; only a bridge
    this client started is stopped.

    A session can run several robots. :meth:`robots` reads the whole roster and
    :meth:`robot` one entry of it; the robot commands and :meth:`send_cmd_vel`
    take the id of the robot they address, so one robot is driven without
    touching the others::

        client.send_cmd_vel(0.4, 0.0, robot="robot_2")
        client.set_robot_goal(4.0, 0.0, robot="robot_2")
        client.set_control_mode("ROS", robot="robot_2")
    """

    def __init__(
        self,
        bridge: RobotSNAPBridge | None = None,
        port: int = 10000,
        host: str = "0.0.0.0",
        autostart: bool = True,
    ):
        """Create the façade, starting an owned bridge when ``autostart`` is true.

        ``bridge`` lets a session share one server between the client and the
        rest of the process; ``port`` and ``host`` are only used when it is
        ``None``. Starting a bridge binds the port, so a second client on the
        same port must be given an existing bridge instead.
        """
        self._bridge = (
            RobotSNAPBridge(host=host, port=port)
            if bridge is None
            else bridge
        )
        self._started = False
        self._last_error: str | None = None
        self._scenario_start: float | None = None
        #: Per-command time budget in seconds; raise it for slow scenario loads.
        self.command_timeout = DEFAULT_COMMAND_TIMEOUT
        if autostart:
            self.start()

    # -- lifecycle ---------------------------------------------------------

    @property
    def bridge(self) -> RobotSNAPBridge:
        """The underlying server, for raw access to topics and raw payloads."""
        return self._bridge

    @property
    def last_error(self) -> str | None:
        """Most recent client-side error, else the bridge's own ``last_error``.

        The client records a message when a command cannot be sent or when no
        acknowledgement arrives inside the timeout. It is not cleared
        automatically, so compare it before and after a call when a stale note
        would be misleading.
        """
        if self._last_error is not None:
            return self._last_error
        return self._bridge.last_error

    @property
    def is_connected(self) -> bool:
        """True while a Unity peer holds the socket."""
        return self._bridge.is_connected

    def start(self) -> None:
        """Start the bridge if it is not running. Does nothing when it is."""
        if not self._bridge.is_running:
            self._bridge.start()
            self._started = True

    def stop(self) -> None:
        """Stop the bridge if this client started it. Idempotent."""
        if self._started:
            self._bridge.stop()
            self._started = False

    def __enter__(self) -> "RobotSNAPClient":
        """Start the bridge and return the client, for use as a context manager."""
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        """Stop the bridge this client started."""
        self.stop()
        return False

    def wait_until_ready(self, timeout: float = 30.0) -> bool:
        """Block until Unity has connected and listens for commands.

        Starts the bridge if needed. Returns ``True`` when the peer is ready,
        ``False`` on timeout, in which case :attr:`last_error` explains it.

        A connected socket is not enough. The components that own each stream
        register it from their ``Start``, which runs while the scene is still
        being built, and the connector discards a message whose topic holds no
        subscriber yet: a command written in that window is answered by nobody
        and looks like a hang. The ``__publish`` and ``__subscribe`` frames
        Unity sends are the first observable proof that registration is done,
        so this waits for the ``__subscribe`` of :data:`CONTROL_TOPIC` - the one
        topic every method of this class writes to - rather than for the
        transport alone. Waiting for *any* registration is not enough: the
        publishers of the state and the map announce themselves before the
        listener of the commands does.

        The wait is on the peer's own announcements, never on the bridge's seed
        table, which is populated before anything connects and would make every
        freshly opened socket look ready.

        A session that only reads - no remote control - never subscribes, so
        :class:`~robotsnap.bridge.server.RobotSNAPBridge` and its
        ``wait_for_connection`` are the entry point for that case.

        Whether Unity already publishes ``/simulation/state`` is a separate
        question, answered by :meth:`snapshot`.
        """
        self.start()
        if not self._bridge.wait_for_connection(timeout=timeout):
            self._record_error(f"no Unity peer connected within {timeout:g}s")
            return False

        if self._bridge.wait_for_subscription(CONTROL_TOPIC, timeout=timeout):
            return True

        self._record_error(
            f"Unity connected but never subscribed to {CONTROL_TOPIC} within {timeout:g}s"
        )
        return False

    # -- state -------------------------------------------------------------

    def snapshot(self) -> dict[str, Any] | None:
        """Parsed body of the last ``/simulation/state`` message, or ``None``.

        ``None`` means nothing has arrived on that topic yet, that the payload
        is not valid JSON, or that it is not a JSON object. In the last two
        cases :attr:`last_error` carries the reason. The returned dict is the
        simulator's own view, keys and values included, in the ROS frame (x
        forward, y left, z up, yaw in radians).
        """
        state = self._read_json(STATE_TOPIC)
        if state is None:
            return None
        if not isinstance(state, dict):
            self._record_error(f"{STATE_TOPIC} payload is not a JSON object")
            return None
        return state

    def state(self, topic: str):
        """Last decoded message on ``topic``, or ``None``.

        Alias of :meth:`last_message`, kept for symmetry with
        :meth:`snapshot`.
        """
        return self._bridge.latest(topic)

    def last_message(self, topic: str):
        """Last decoded message on ``topic``, or ``None`` before the first one.

        ``topic`` accepts the base name, with or without a leading slash or an
        environment prefix (``odom``, ``/odom``, ``/robot0/odom``). The message
        is the rosbags instance, so its fields are typed.
        """
        return self._bridge.latest(topic)

    @property
    def held(self) -> bool | None:
        """The snapshot's ``held`` flag: whether the simulation is stopped right now.

        ``True`` while the world is effectively stopped - a lockstep gate
        between two releases, a frozen session - and ``False`` while it runs, as
        the session's own snapshot reports it (the engine's ``Time.timeScale``
        at or below zero, not the configured scale, which stays above zero while
        a lockstep client holds the world). A build that predates the key, or a
        session that has published nothing yet, answers ``None``, so a caller
        can tell "the world is moving" from "this session cannot say".
        """
        state = self.snapshot()
        if not isinstance(state, dict):
            return None
        value = state.get("held")
        return value if isinstance(value, bool) else None

    def humans(self) -> list[dict[str, Any]] | None:
        """The ``humans`` list of the last ``/simulation/state`` snapshot.

        Each entry is ``{"id": int, "x": float, "y": float, "z": float, "vx":
        float, "vy": float, "vz": float, "speed": float, "goal": {...}|null,
        "group": ...|null, "controller": "sfm"|"external", "end_behavior":
        str}``, in the world frame. ``None`` when no snapshot has arrived or the
        snapshot carries no ``humans`` key.
        """
        state = self.snapshot()
        if state is None:
            return None
        humans = state.get("humans")
        return humans if isinstance(humans, list) else None

    def agents(self, robot: str | None = None) -> dict[str, Any] | None:
        """Parsed body of the last agent message of ``robot``, or the primary one.

        The body is ``{"agents": [{"id": ..., "x": ..., "y": ..., "z": ...,
        "vx": ..., "vy": ..., "vz": ..., "visible": ...}], "frame": "robot"}``:
        every agent of the session, in the robot frame. ``None`` means nothing
        has arrived on that topic yet, that the payload is not valid JSON, or
        that it does not match that shape; :attr:`last_error` carries the reason
        in the last two cases.

        The agent snapshot is taken in the frame of the robot that carries the
        detector, so ``robot="robot_2"`` reads ``/robot_2/simulation/agents``:
        what robot 2 can see. ``robot`` left as ``None`` reads the unprefixed
        ``/simulation/agents``, the primary robot's stream, exactly as before.
        """
        topic = (
            AGENTS_TOPIC
            if robot is None
            else topics.robot_topic(robot, topics.SIMULATION_AGENTS)
        )
        body = self._read_json(topic)
        if body is None:
            return None
        try:
            return codec.decode_agents_json(body)
        except codec.CodecError as exc:
            self._record_error(f"{topic} payload is invalid: {exc}")
            return None

    def robots(self) -> list[dict[str, Any]] | None:
        """The ``robots`` roster of the last ``/simulation/state`` snapshot.

        ``None`` means no snapshot has arrived; an empty list is a valid answer,
        and means the scene holds no robot. Each entry is ``{"id": str, "type":
        str, "is_primary": bool, "x": float, "y": float, "z": float, "yaw":
        float, "has_goal": bool, "goal": {...}|null, "start_pose": {...}|null,
        "target_pose": {...}|null}``, the contract's roster view.
        """
        state = self.snapshot()
        if state is None:
            return None
        robots = state.get("robots")
        return robots if isinstance(robots, list) else None

    def robot(self, robot_id: str | None = None) -> dict[str, Any] | None:
        """One entry of the ``robots`` roster of the last snapshot, or ``None``.

        With ``robot_id`` given, the entry whose ``id`` matches it; ``None`` when
        no snapshot has arrived or the roster does not carry that id. With
        ``robot_id`` left as ``None``, the primary robot: the entry flagged
        ``is_primary``, falling back to the first entry.

        Older sessions carried a single top-level ``robot`` key and no roster;
        when the snapshot has no ``robots`` array (or an empty one) and the
        caller asked for the primary, that legacy key is returned instead, so
        the pre-roster behaviour is preserved.
        """
        state = self.snapshot()
        if state is None:
            return None
        robot_id = _robot_id(robot_id)
        if robot_id is not None:
            for entry in state.get("robots") or ():
                if isinstance(entry, dict) and _robot_id(entry.get("id")) == robot_id:
                    return entry
            return None

        robots = state.get("robots")
        if isinstance(robots, list):
            entries = [entry for entry in robots if isinstance(entry, dict)]
            primary = next(
                (entry for entry in entries if entry.get("is_primary")), None
            )
            if primary is None and entries:
                primary = entries[0]
            if primary is not None:
                return primary
        robot = state.get("robot")
        return robot if isinstance(robot, dict) else None

    # -- scene and robot commands -----------------------------------------

    def play(self) -> dict[str, Any] | None:
        """Start or resume the simulation (``{"command": "play"}``)."""
        return self._send_command({"command": "play"})

    def pause(self) -> dict[str, Any] | None:
        """Pause the simulation (``{"command": "pause"}``)."""
        return self._send_command({"command": "pause"})

    def toggle_pause(self) -> dict[str, Any] | None:
        """Flip the paused flag (``{"command": "toggle_pause"}``)."""
        return self._send_command({"command": "toggle_pause"})

    def reset(
        self, scenario: str | None = None, seed: int | None = None
    ) -> dict[str, Any] | None:
        """Reset the episode (``{"command": "reset"}``).

        ``scenario`` and ``seed`` are only added to the body when they are not
        ``None``, so Unity keeps its own defaults otherwise.

        The acknowledgement comes back as soon as the reset is queued, before
        the world is rebuilt, so it does not move :attr:`scenario_time_seconds`:
        call :meth:`mark_scenario_start` once the new world is there, or use
        :meth:`launch_scenario`, which waits and marks.
        """
        body: dict[str, Any] = {"command": "reset"}
        if scenario is not None:
            body["scenario"] = str(scenario)
        if seed is not None:
            body["seed"] = int(seed)
        return self._send_command(body, timeout=SCENARIO_COMMAND_TIMEOUT)

    def load_scenario(
        self,
        name: str,
        start: bool = True,
        apply: bool = True,
        seed: int | None = None,
    ) -> dict[str, Any] | None:
        """Load ``name`` and optionally start it.

        Sends ``{"command": "load_scenario", "scenario": name, "start": start,
        "apply": apply}``, plus ``seed`` when it is not ``None``.

        The answer describes what the simulator accepted, not what is built: when
        another scenario is already loading, the message says the request was
        queued behind it. Follow with :meth:`wait_for_scenario` when the next
        statement depends on the new world being there.
        """
        body: dict[str, Any] = {
            "command": "load_scenario",
            "scenario": str(name),
            "start": bool(start),
            "apply": bool(apply),
        }
        if seed is not None:
            body["seed"] = int(seed)
        return self._send_command(body, timeout=SCENARIO_COMMAND_TIMEOUT)

    def wait_for_scenario(
        self,
        name: str,
        timeout: float = SCENARIO_SETTLE_TIMEOUT,
        applied: bool = True,
    ) -> bool:
        """Block until the session reports ``name`` as its scenario.

        ``load_scenario`` answers as soon as the request is accepted, which is
        not the same as done. A scenario being built cannot be interrupted - the
        map, the NavMesh and the crowd are half made - so a request that arrives
        during one is queued by the simulator and started when that one ends; the
        answer then says "queued". Building the world itself takes seconds
        afterwards.

        This polls ``/simulation/state`` until its ``scenario_id`` is the one
        asked for, and until ``scenario_applied`` is true when ``applied`` is
        true, which is the moment the agents exist. Returns ``True`` on success,
        ``False`` on timeout with :attr:`last_error` set. It sends nothing, so it
        is safe to call while the simulator is busy.
        """
        wanted = str(name).strip().lower()
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            state = self.snapshot()
            if isinstance(state, dict):
                current = state.get("scenario_id")
                loaded = isinstance(current, str) and current.strip().lower() == wanted
                if loaded and (not applied or bool(state.get("scenario_applied"))):
                    self.mark_scenario_start()
                    return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._record_error(
                    f"scenario {name!r} was not applied within {timeout:g}s"
                )
                return False
            time.sleep(min(COMMAND_POLL_INTERVAL, remaining))

    # -- writing, launching and stopping a scenario -------------------------

    def create_scenario(
        self,
        name: str,
        *,
        document: Mapping[str, Any] | None = None,
        directory: str | Path | None = None,
        overwrite: bool = True,
        launch: bool = False,
        seed: int | None = None,
        timeout: float = SCENARIO_SETTLE_TIMEOUT,
        **fields: Any,
    ) -> Path | None:
        """Write a scenario file the simulator can load, optionally loading it.

        ``name`` is the id the session knows the scenario by, and the file is
        ``<name>.yaml`` in the scenarios directory of the Unity project - found
        like :func:`robotsnap.scenario.scenarios_dir` does, unless ``directory``
        names another one.

        The document is written from ``document`` when one is given, and built
        by :func:`robotsnap.scenario.build` from ``fields`` otherwise, so the
        usual call reads::

            client.create_scenario(
                "python_demo",
                map_name="basic/crowd",
                robots=[scenario.robot("robot_1", "jackal", (2.0, -6.0, 90), (20.0, -6.0))],
                launch=True,
            )

        With ``launch`` true the new scenario is loaded and waited for, which is
        :meth:`launch_scenario` with the same ``seed`` and ``timeout``. Returns
        the path that was written, or ``None`` with :attr:`last_error` set when
        the document is refused or the launch fails.

        Unity caches a scenario it has already loaded, so writing a new document
        under a name the session has used does not change what it loads; use a
        new name, or overwrite before the session knows the old one.
        """
        try:
            if document is None:
                document = _scenario.build(name, **fields)
            path = _scenario.write(
                document, name=name, directory=directory, overwrite=overwrite
            )
        except (OSError, TypeError, ValueError, KeyError) as exc:
            self._record_error(f"cannot write scenario {name!r}: {exc}")
            return None

        scenario_id = path.stem
        if launch and not self.launch_scenario(scenario_id, seed=seed, timeout=timeout):
            return None
        return path

    def launch_scenario(
        self,
        name: str,
        *,
        start: bool = True,
        seed: int | None = None,
        timeout: float = SCENARIO_SETTLE_TIMEOUT,
    ) -> bool:
        """Load ``name`` and wait until the world is built around it.

        The pair a script actually needs: :meth:`load_scenario` only has the
        request acknowledged, and building the map, the crowd and the navigation
        grid happens after that. Returns ``True`` once the session reports the
        scenario as applied, ``False`` on a refusal or a timeout, with
        :attr:`last_error` explaining which. The episode clock of
        :attr:`scenario_time_seconds` starts at that moment.

        ``name`` is the file name of the scenario without its extension, the id
        ``/simulation/state`` reports as ``scenario_id``.
        """
        result = self.load_scenario(name, start=start, seed=seed)
        if not self._command_ok(result):
            if result is None:
                self._record_error(f"load_scenario {name!r} got no answer")
            return False
        return self.wait_for_scenario(name, timeout=timeout)

    def stop_scenario(
        self, park_robots: bool = True, stop_crowd: bool = True
    ) -> bool:
        """Freeze the running scenario, without unloading it.

        Pauses the session, and by default holds every robot of the roster still
        and stops the crowd, so nothing keeps moving under a paused flag. The
        world, its map and its agents stay in the scene: this is what "stop"
        means here, and :meth:`launch_scenario` brings another one in, while
        :meth:`load_scenario` with ``start=False`` leaves one loaded and paused.

        This is a freeze, not the application's red stop button: every agent
        stays in the scene. :meth:`stop_simulation` is that button - it pulls
        the agents out and leaves the application in standby.

        Returns ``True`` when every step was acknowledged, ``False`` with
        :attr:`last_error` set otherwise. The scenario clock of
        :attr:`scenario_time_seconds` keeps its anchor: start it again with
        :meth:`launch_scenario` or :meth:`mark_scenario_start`.
        """
        ok = self._command_ok(self.pause())

        if park_robots:
            roster = self.robots()
            ids = [
                entry.get("id")
                for entry in roster or ()
                if isinstance(entry, dict) and entry.get("id")
            ]
            for robot_id in ids or (None,):
                ok = self._command_ok(self.stop_robot(robot_id)) and ok

        if stop_crowd and self.humans():
            ok = self._command_ok(self.stop_humans()) and ok

        return ok

    def stop_simulation(self) -> bool:
        """Stop the simulation outright, as the red stop button does
        (``{"command": "stop_simulation"}``).

        The command behind the application's red stop button: the episode clock
        is paused, every agent is taken out of the scene and the state goes back
        to ``Ready``, while the map and the loaded scenario stay in place. The
        application ends up in standby, ready to be played again.

        This is not :meth:`stop_scenario`, which only freezes the session where
        it stands - it pauses the clock, parks every robot of the roster and
        stops the crowd, but keeps every agent in the scene. Use
        :meth:`stop_simulation` to leave the application in standby,
        :meth:`stop_scenario` to keep the world in place.

        Returns ``True`` when the command was acknowledged, ``False`` with
        :attr:`last_error` set otherwise.
        """
        return self._command_ok(self._send_command({"command": "stop_simulation"}))

    # -- episode clock ------------------------------------------------------

    @property
    def scenario_time_seconds(self) -> float | None:
        """Simulated seconds since the current scenario was launched, or ``None``.

        The ``sim_time_seconds`` of a snapshot is the simulator's own clock: it
        belongs to the Unity session, keeps running while a scenario is swapped,
        and a scenario load does not put it back to zero. A script that measures
        an episode therefore measures it from a mark of its own, and this is
        that mark subtracted from the clock - the value
        :meth:`launch_scenario` and :meth:`wait_for_scenario` take once the
        world is applied, or the one :meth:`mark_scenario_start` takes on
        demand.

        ``None`` before any mark, and when the last snapshot carries no clock.
        It is the clock the simulator scales, not wall-clock time: with
        ``set_time_scale(2)`` one second here is two of the wall.
        """
        if self._scenario_start is None:
            return None
        state = self.snapshot()
        current = state.get("sim_time_seconds") if isinstance(state, dict) else None
        if not isinstance(current, (int, float)):
            return None
        return float(current) - self._scenario_start

    def mark_scenario_start(self) -> float | None:
        """Anchor :attr:`scenario_time_seconds` on the current simulated clock.

        Called by :meth:`launch_scenario` and :meth:`wait_for_scenario` once the
        scenario is applied; call it yourself after a :meth:`reset`, whose
        acknowledgement arrives before the world is rebuilt. Returns the value
        kept, or ``None`` when no snapshot carries a clock yet.
        """
        state = self.snapshot()
        current = state.get("sim_time_seconds") if isinstance(state, dict) else None
        self._scenario_start = (
            float(current) if isinstance(current, (int, float)) else None
        )
        return self._scenario_start

    def set_time_scale(
        self,
        scale: float,
        fixed_timestep: float | None = None,
        maximum_delta_time: float | None = None,
    ) -> dict[str, Any] | None:
        """Set how fast the session runs (``{"command": "set_time_scale"}``).

        ``scale`` is Unity's ``Time.timeScale``. ``fixed_timestep`` is the physics step in seconds and
        ``maximum_delta_time`` the most simulation one drawn frame may catch up on; left out, the simulator
        keeps the step it has and lets its catch-up ceiling follow the scale.

        The two matter as much as the scale itself once a session runs fast. The rate a session actually
        reaches is the physics step times the steps per second the machine can pay for, so the same machine
        delivers twice the simulation at a step of 0.04 as at 0.02; and the ceiling is what a frame that
        draws slowly is allowed to owe, so a ceiling below the ask answers only part of it.
        """
        body: dict[str, Any] = {"command": "set_time_scale", "time_scale": float(scale)}
        if fixed_timestep is not None:
            body["fixed_timestep"] = float(fixed_timestep)
        if maximum_delta_time is not None:
            body["maximum_delta_time"] = float(maximum_delta_time)
        return self._send_command(body)

    def set_random_seed(self, seed: int) -> dict[str, Any] | None:
        """Seed the simulator's random generator
        (``{"command": "set_random_seed"}``)."""
        return self._send_command({"command": "set_random_seed", "seed": int(seed)})

    def set_pacing(
        self,
        mode: str,
        step_seconds: float | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any] | None:
        """Choose how the session is paced (``{"command": "set_pacing"}``).

        ``mode`` is ``"free"`` - the world runs at its time scale while the client thinks - or
        ``"lockstep"``, where it stops and only :meth:`release_pacing` moves it. In lockstep a step covers
        exactly ``step_seconds`` of simulation, so a policy that takes longer to answer costs wall time and
        not fidelity, and an episode cannot run past the simulated time it was given.

        ``timeout`` bounds the wait for the acknowledgement; left out, the client's own
        ``command_timeout`` applies. A caller that sends the command as a courtesy - the hold at the end of
        an episode, say - passes a short one, so a session that has gone away costs a bounded wait rather
        than the whole timeout.
        """
        body: dict[str, Any] = {"command": "set_pacing", "mode": str(mode)}
        if step_seconds is not None:
            body["step_seconds"] = float(step_seconds)
        return self._send_command(body, timeout=timeout)

    def release_pacing(self) -> bool:
        """Let a lockstep session spend one period, and do not wait for the answer.

        Called once per control step, after the command the period is answering for. The acknowledgement is
        sent but not waited for: it would cost a round trip per step for something the caller learns anyway,
        which is that the world has reached the clock it asked for.
        """
        return self._bridge.publish(
            CONTROL_TOPIC, _STRING, _string_message({"command": "release_pacing"})
        )

    def set_robot_goal(
        self,
        x: float,
        z: float,
        y: float | None = None,
        robot: str | None = None,
    ) -> dict[str, Any] | None:
        """Set the robot goal (``{"command": "set_robot_goal"}``).

        ``x`` and ``z`` are the simulator's own world axes, the ones the
        scenario file authors: ``x`` is Unity ``x`` and ``z`` is Unity ``z``.
        ``y``, sent only when it is not ``None``, is the height above the
        ground. This is not the frame :meth:`robot` and :meth:`snapshot`
        report in: those carry the ROS frame, where the ground axes are ``x``
        and ``y`` and ``z`` is up. Use :meth:`set_robot_goal_ros` to send a
        goal a snapshot just gave back. ``robot`` names the robot the goal is
        for and adds the contract's ``"robot"`` key when it is set; left as
        ``None`` the body carries no ``robot`` key and the simulator's primary
        robot applies.

        ``y`` stays the third positional argument, so a call written before the
        robot key existed - ``set_robot_goal(x, z, y)`` - keeps its meaning; pass
        the robot by keyword, ``set_robot_goal(x, z, robot="robot_2")``.

        The acknowledgement echoes the id the simulator acted on under
        ``"robot"``, and is returned as it arrives.
        """
        body: dict[str, Any] = {
            "command": "set_robot_goal",
            "x": float(x),
            "z": float(z),
        }
        if y is not None:
            body["y"] = float(y)
        _add_robot(body, robot)
        return self._send_command(body)

    def set_robot_goal_ros(
        self,
        x: float,
        y: float,
        z: float | None = None,
        robot: str | None = None,
    ) -> dict[str, Any] | None:
        """Set the robot goal, given in the ROS frame (``x``, ``y`` metres).

        ``/simulation/state`` reports every pose in the ROS frame - the robot
        itself, its ``goal``, its ``start_pose`` and its ``target_pose`` - so a
        goal read from one of them can be sent back through this method without
        being reexpressed: the ROS ``y`` axis is the negated simulator ``x``
        axis and the ROS ``x`` axis is the simulator ``z`` axis.

        ``z`` is the height above the ground and is only sent when it is not
        ``None``, so a client that only reads the two ground axes can leave it
        out. ``robot`` behaves as it does in :meth:`set_robot_goal`.
        """
        return self.set_robot_goal(-float(y), float(x), y=z, robot=robot)

    def clear_robot_goal(self, robot: str | None = None) -> dict[str, Any] | None:
        """Drop the current robot goal (``{"command": "clear_robot_goal"}``).

        ``robot`` names the robot and adds the contract's ``"robot"`` key when it
        is set; left as ``None`` the simulator's primary robot applies.
        """
        body: dict[str, Any] = {"command": "clear_robot_goal"}
        _add_robot(body, robot)
        return self._send_command(body)

    def stop_robot(self, robot: str | None = None) -> dict[str, Any] | None:
        """Stop the robot where it stands (``{"command": "stop_robot"}``).

        ``robot`` names the robot and adds the contract's ``"robot"`` key when it
        is set; left as ``None`` the simulator's primary robot applies.
        """
        body: dict[str, Any] = {"command": "stop_robot"}
        _add_robot(body, robot)
        return self._send_command(body)

    def set_control_mode(
        self, mode: str, robot: str | None = None
    ) -> dict[str, Any] | None:
        """Choose how the robot is driven (``{"command": "set_control_mode"}``).

        The Unity ``RobotInputController`` expects ``ROS``, ``Hybrid`` or
        ``Keyboard``; only ``ROS`` avoids the keyboard fallback that fights an
        intermittent Python client.

        ``robot`` names the robot that switches mode and adds the contract's
        ``"robot"`` key when it is set; left as ``None`` the simulator's primary
        robot applies.
        """
        body: dict[str, Any] = {
            "command": "set_control_mode",
            "mode": str(mode),
        }
        _add_robot(body, robot)
        return self._send_command(body)

    def set_agent_controller(self, mode: str) -> dict[str, Any] | None:
        """Switch the humans' controller (``{"command": "set_agent_controller"}``).

        ``mode`` is ``"sfm"`` or ``"external"``, matching the ``controller``
        field of :meth:`humans`.
        """
        return self._send_command(
            {"command": "set_agent_controller", "mode": str(mode)}
        )

    # -- human commands ----------------------------------------------------

    def set_human_velocities(self, commands: Sequence) -> dict[str, Any] | None:
        """Send one batch of human velocity commands.

        ``commands`` is a sequence whose entries are either ``(id, vx, vz)``
        tuples or dicts. A dict is passed through as
        ``{"id": int, "vx": float, "vz": float}``, and ``{"id": int, "stop":
        true}`` stops that human instead of driving it. Anything else in the
        entry is forwarded untouched.

        The body is ``{"command": "humans", "commands": [...]}`` on
        ``/simulation/control``, the same topic as every other command. The
        answer carries the usual keys plus ``"unknown_ids"``, the ids the scene
        does not hold, and is returned as parsed; ``None`` means nothing came
        back before the timeout.

        Velocities are metres per second along the simulator's world x and z,
        not the ROS frame.
        """
        entries = [_human_entry(entry) for entry in commands]
        return self._send_command({"command": "humans", "commands": entries})

    def stop_human(self, id: int) -> dict[str, Any] | None:
        """Stop one human
        (``{"command": "humans", "commands": [{"id": id, "stop": true}]}``)."""
        return self.set_human_velocities([{"id": int(id), "stop": True}])

    def stop_humans(self) -> dict[str, Any] | None:
        """Stop every human listed in the last snapshot.

        The command carries an id per entry, so this enumerates the ids from the
        last ``/simulation/state`` snapshot and sends one ``{"id": id, "stop":
        true}`` entry per human, in a single ``humans`` command. When no
        snapshot or no human id is known, nothing is sent and the method returns
        ``None`` with :attr:`last_error` set.
        """
        humans = self.humans() or []
        entries = [
            {"id": int(human["id"]), "stop": True}
            for human in humans
            if isinstance(human, Mapping) and human.get("id") is not None
        ]
        if not entries:
            self._record_error(
                f"stop_humans: no human id in the last {STATE_TOPIC} snapshot"
            )
            return None
        return self.set_human_velocities(entries)

    def send_cmd_vel(
        self, linear_x: float, angular_z: float, robot: str | None = None
    ) -> bool:
        """Send a ``geometry_msgs/Twist`` through the bridge.

        Returns ``False`` when nothing is connected or the payload cannot be
        encoded; see :attr:`last_error`. Unity flips the angular sign and clamps
        both values, so a command is not necessarily the motion that happens.

        ``robot`` names the robot that is driven and publishes on its
        ``/robot_<id>/cmd_vel``; left as ``None`` the Twist goes to ``/cmd_vel``,
        which reaches the primary robot. Both forms go through the bridge's
        public ``publish(topic, msg_type, msg)``.
        """
        topic = (
            topics.CMD_VEL
            if robot is None
            else topics.robot_topic(robot, topics.CMD_VEL)
        )
        return self._bridge.publish(
            topic, topics.TWIST_TYPE, _twist(linear_x, angular_z)
        )

    # -- metrics -----------------------------------------------------------

    def episodes(self) -> list[dict[str, Any]] | None:
        """Every episode the running session has finished, oldest first.

        The episodes come from Unity's own store over the control topic, so they are the same ones the
        in-editor dashboard lists - the two views cannot disagree about a session. Each entry is the episode
        document described by :mod:`robotsnap.analysis.metrics`: the outcome, the timings, the social numbers and the
        trajectories of every robot the scenario fielded and of every human it saw. The scalar numbers belong
        to the one robot the document names as ``robot``; the other robots are tracks beside it.

        An empty list means the session has not finished an episode yet, which is a valid answer and not an
        error. ``None`` means no answer arrived at all, or that the answer was not the document this method
        expects; :attr:`last_error` carries the reason in the second case.
        """
        payload = self._metrics_payload("metrics_episodes")
        if payload is None:
            return None

        raw = payload.get("episodes")
        if not isinstance(raw, list):
            return []
        return [dict(episode) for episode in raw if isinstance(episode, Mapping)]

    def episode(self, identifier: str) -> dict[str, Any] | None:
        """One episode of the running session by its identifier, or ``None``.

        ``None`` covers both "the session does not hold that id" - a cleared session, a typo - and "no answer
        arrived"; a caller that needs to tell them apart reads :attr:`last_error`, which is set only in the
        second case. An episode that exists is the same document :meth:`episodes` returns for it.
        """
        payload = self._metrics_payload("metrics_episode", {"id": str(identifier)})
        if payload is None:
            return None

        found = payload.get("episode")
        return dict(found) if isinstance(found, Mapping) else None

    def clear_episodes(self) -> int | None:
        """Empty the session's episode list and start a new session.

        Returns how many episodes were dropped, so a caller can report what it did rather than what it asked
        for. The export files already on disk are untouched - they are the record of a session that ran, and
        this clears the live list. ``None`` when no answer arrived; see :attr:`last_error`.
        """
        payload = self._metrics_payload("metrics_clear")
        if payload is None:
            return None

        cleared = payload.get("cleared")
        return int(cleared) if isinstance(cleared, (int, float)) else None

    def metrics_session(self) -> dict[str, Any] | None:
        """Identity of the session holding the episodes: ``{"session": ..., "started_at": ..., "count": n}``.

        Read from the same answer :meth:`episodes` uses, so it costs one command and describes exactly the
        list that came back with it. ``None`` when no answer arrived.
        """
        payload = self._metrics_payload("metrics_episodes")
        if payload is None:
            return None
        return {
            "session": payload.get("session"),
            "started_at": payload.get("started_at"),
            "count": payload.get("count"),
        }

    def latest_episode(self) -> dict[str, Any] | None:
        """The last episode Unity *published*, read from ``/simulation/metrics``.

        Unlike :meth:`episodes`, which asks for the session's whole list, this reads the stream: it is what a
        client watching a run reacts to as each episode ends, and it answers ``None`` until the first episode
        of the session has finished.
        """
        episode = self._read_json(METRICS_TOPIC)
        if episode is None:
            return None
        if not isinstance(episode, dict):
            self._record_error(f"{METRICS_TOPIC} payload is not a JSON object")
            return None
        return episode

    def metrics_summary(self) -> dict[str, Any] | None:
        """The session's episodes, aggregated: success rate, failure rates, per-metric mean/min/max.

        The aggregation is :func:`robotsnap.analysis.metrics.summarize`, so a Python caller and a reader of the
        exported files compute the same numbers from the same episodes. ``None`` when no answer arrived.
        """
        episodes = self.episodes()
        if episodes is None:
            return None
        return _metrics.summarize(episodes)

    def exported_episodes(self, directory: str | Path | None = None) -> list[dict[str, Any]]:
        """Every episode exported to disk under ``directory``, for a session this client did not watch.

        ``directory`` defaults to ``$ROBOTSNAP_METRICS_DIR`` when that is set, which is how a caller points
        the reader at the ``StreamingAssets/metrics`` folder of the build it trained against. A directory
        that is not there is an empty list, so a benchmark script can call this before the first session.
        """
        root = directory if directory is not None else os.environ.get("ROBOTSNAP_METRICS_DIR")
        if root is None:
            self._record_error(
                "no export directory: pass one, or set ROBOTSNAP_METRICS_DIR to the "
                "StreamingAssets/metrics folder of the build"
            )
            return []
        return _metrics.load_export(root)

    def exported_summary(self, directory: str | Path | None = None) -> dict[str, Any]:
        """Aggregate the exported episodes of ``directory``; see :meth:`exported_episodes`."""
        return _metrics.summarize(self.exported_episodes(directory=directory))

    def _metrics_payload(
        self,
        command: str,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Run one metrics command and return the document Unity answered with.

        Every metrics question is answered the same way - an ``ok`` acknowledgement carrying a ``payload``
        document - so the three public methods above share this one trip. A command the simulator refuses, or
        an answer without a payload, records why and returns ``None`` instead of an empty result a caller
        would read as "no episodes".
        """
        body: dict[str, Any] = {"command": command}
        if extra is not None:
            body.update(extra)

        result = self._send_command(body)
        if not self._command_ok(result):
            return None

        payload = result.get("payload")
        if not isinstance(payload, Mapping):
            self._record_error(f"{command} answered without a payload document")
            return None
        return dict(payload)

    # -- plumbing ----------------------------------------------------------

    def _send_command(
        self,
        body: dict[str, Any],
        timeout: float | None = None,
    ) -> dict[str, Any] | None:
        """Send one JSON command and wait for its acknowledgement."""
        budget = self.command_timeout if timeout is None else timeout
        before = self._bridge.topic_counts().get(topics.base(CONTROL_RESULT_TOPIC), 0)
        if not self._bridge.publish(CONTROL_TOPIC, _STRING, _string_message(body)):
            self._record_error(f"cannot send command on {CONTROL_TOPIC}: no Unity peer")
            return None

        arrived, result = self._wait_for_result(before, budget)
        if not arrived:
            self._record_error(
                f"no {CONTROL_RESULT_TOPIC} within {budget:g}s "
                f"for command {body.get('command')!r}"
            )
            return None
        if result is None:
            self._record_error(f"{CONTROL_RESULT_TOPIC} payload is not valid JSON")
            return None
        return result

    def _command_ok(self, result: dict[str, Any] | None) -> bool:
        """Whether an acknowledgement says the simulator did what was asked.

        ``None`` means no answer arrived, and :meth:`_send_command` has already
        recorded why. A body without an ``ok`` key is a plain acknowledgement, so
        only an explicit ``false`` is a refusal; its ``message`` is kept, since
        that is where the simulator names the id or the scenario it did not find.
        """
        if result is None:
            return False
        if result.get("ok") is False:
            self._record_error(str(result.get("message") or "the simulator refused the command"))
            return False
        return True

    def _wait_for_result(
        self, before: int, timeout: float
    ) -> tuple[bool, dict[str, Any] | None]:
        """Poll the bridge for a newer ``/simulation/control_result`` message.

        Returns ``(arrived, parsed)``. The loop sleeps
        :data:`COMMAND_POLL_INTERVAL` between two reads of the bridge's decoded
        messages, so it costs far less than the round trip and never touches the
        socket itself.
        """
        deadline = time.monotonic() + max(0.0, timeout)
        key = topics.base(CONTROL_RESULT_TOPIC)
        while True:
            if self._bridge.topic_counts().get(key, 0) > before:
                return True, self._read_json(CONTROL_RESULT_TOPIC)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False, None
            time.sleep(min(COMMAND_POLL_INTERVAL, remaining))

    def _read_json(self, topic: str) -> Any | None:
        """Parse the body of the last message on a JSON ``std_msgs/String`` topic.

        The message already decoded by the bridge is preferred; when the bridge
        does not know the topic type, the raw payload is decoded here with the
        ``std_msgs/String`` codec so a late ``__publish`` registration is not
        needed to read a value that arrived.
        """
        message = self._bridge.latest(topic)
        if message is None:
            raw = self._bridge.latest_raw(topic)
            if raw is None:
                return None
            try:
                message = codec.decode(_STRING, raw)
            except codec.CodecError as exc:
                self._record_error(f"{topic} payload cannot be decoded: {exc}")
                return None

        text = message.data if hasattr(message, "data") else message
        if isinstance(text, (dict, list)):
            return text
        try:
            return json.loads(text)
        except (TypeError, ValueError) as exc:
            self._record_error(f"{topic} payload is not valid JSON: {exc}")
            return None

    def _record_error(self, message: str) -> None:
        """Keep the most recent client-side error for :attr:`last_error`."""
        self._last_error = str(message)


def open_client(
    *,
    transport: str = "tcp",
    host: str = "0.0.0.0",
    port: int = 10000,
    autostart: bool = True,
    node_name: str = "robotsnap_rl",
) -> RobotSNAPClient:
    """A client on the transport the caller asked for, so a run says *what* not *how*.

    ``tcp`` is the bridge's own socket and is what every run used before this existed: the client
    binds the port and Unity dials it. ``ros2`` joins a graph someone else is already serving - the
    ROS2 endpoint serves Unity - and the client is the same façade over a
    :class:`~robotsnap_ros2.Ros2Client`, which answers the bridge's whole surface. Nothing above the
    client knows which one it holds; the environment, the policies and the metrics are written once.

    Raises :class:`ValueError` on an unknown transport rather than quietly falling back, because a
    run that silently used the socket would look exactly like a working one.
    """
    if transport == "tcp":
        return RobotSNAPClient(host=host, port=port, autostart=autostart)
    if transport != "ros2":
        raise ValueError(f"unknown transport {transport!r}: use one of {', '.join(TRANSPORTS)}")

    # Loaded here, through the entry point, so a machine without ROS2 can import this module; the
    # failure a caller meets when it asks for the graph is Ros2Unavailable, with the line that fixes
    # it.
    from robotsnap.bridge.transport import load_ros2

    Ros2Client = load_ros2().Ros2Client
    session = Ros2Client(node_name=node_name)
    return RobotSNAPClient(bridge=session.bridge, autostart=autostart)


def _string_message(body: Mapping[str, Any]):
    """Wrap a JSON body in a ``std_msgs/msg/String`` typestore instance."""
    return codec.TYPESTORE.types[_STRING](data=json.dumps(body))


def _robot_id(value: Any) -> str | None:
    """Normalise a robot id to its ``robot_<id>`` form, or ``None`` when blank.

    The contract names a robot ``robot_2``; a bare ``2`` or a slashed
    ``/robot_2/`` is accepted and normalised to that same id, so the id, the
    command body and the ``/robot_<id>/...`` topic name agree.
    """
    if value is None:
        return None
    name = str(value).strip().strip("/")
    if not name:
        return None
    if not name.startswith("robot_"):
        name = "robot_" + name
    return name


def _add_robot(body: dict[str, Any], robot: Any) -> None:
    """Add the contract's optional ``"robot"`` key when a robot is named."""
    name = _robot_id(robot)
    if name is not None:
        body["robot"] = name


def _twist(linear_x: float, angular_z: float):
    """Build the ``geometry_msgs/msg/Twist`` instance of a velocity command."""
    vector = codec.TYPESTORE.types["geometry_msgs/msg/Vector3"]
    return codec.TYPESTORE.types[topics.TWIST_TYPE](
        linear=vector(x=float(linear_x), y=0.0, z=0.0),
        angular=vector(x=0.0, y=0.0, z=float(angular_z)),
    )


def _human_entry(entry) -> dict[str, Any]:
    """Normalise one ``set_human_velocities`` entry into its wire form."""
    if isinstance(entry, Mapping):
        normalised: dict[str, Any] = dict(entry)
        if "id" in normalised and normalised["id"] is not None:
            normalised["id"] = int(normalised["id"])
        for key in ("vx", "vz", "x", "z"):
            if key in normalised and normalised[key] is not None:
                normalised[key] = float(normalised[key])
        if "stop" in normalised:
            normalised["stop"] = bool(normalised["stop"])
        return normalised

    if isinstance(entry, Sequence) and not isinstance(entry, (str, bytes)):
        values = list(entry)
        if len(values) != 3:
            raise ValueError(
                "a human velocity entry must be (id, vx, vz), got "
                f"{len(values)} value(s)"
            )
        return {"id": int(values[0]), "vx": float(values[1]), "vz": float(values[2])}

    raise TypeError(
        f"unsupported human command entry {entry!r}: expected a dict or an "
        "(id, vx, vz) tuple"
    )
