"""Writing a scenario from Python, launching it and driving it by hand."""

from __future__ import annotations

import sys
import time
from typing import Any

from robotsnap import scenario

from robotsnap.runs.scenario_document import scenario_document
from robotsnap.runs.session import (
    _announce_session,
    _close_viewer,
    _open_viewer,
    _pump_viewer,
    _scenario_directory,
)


def run_scenario(
    *,
    scenario_id: str = "python_demo",
    client: Any = None,
    port: int = 10000,
    host: str = "0.0.0.0",
    unity_project: str | None = None,
    timeout: float = 120.0,
    seconds: float = 6.0,
    speed: float = 0.4,
    no_drive: bool = False,
    render: bool = False,
    keep: bool = False,
) -> int:
    """Write a scenario, launch it, drive its robot and freeze it again.

    The reverse of the usual direction: a script writes a scenario file into the
    project, asks the session to build it, moves the robot inside it, and stops
    it. The bridge has to be listening before the scene dials it, so start this
    first and press Play in Unity after.

    What it prints on purpose is ``sim_time_seconds`` next to ``scenario_time``:
    the first is the simulator's own clock and counts from the moment the
    session started, the second is measured from the launch, which is the number
    an episode is made of.
    """
    from robotsnap.client import RobotSNAPClient

    directory = _scenario_directory(unity_project)
    if directory is None:
        return 2

    path = scenario.write(
        scenario_document(scenario_id), name=scenario_id, directory=directory
    )
    print(f"scenario written : {path}")

    if client is None:
        client = RobotSNAPClient(host=host, port=port)
    # The window is opened once Unity is there, and it is the same one the environment draws: this
    # command drives the client itself, so nothing else would redraw the world while it does.
    viewer = None
    try:
        _announce_session(client, host=host, port=port)
        if not client.wait_until_ready(timeout=timeout):
            print(f"Unity never connected: {client.last_error}", file=sys.stderr)
            return 1
        print("Unity connected   : scene registered its command listener")
        if render:
            viewer = _open_viewer(client)

        if not client.launch_scenario(scenario_id, timeout=timeout):
            print(f"scenario refused  : {client.last_error}", file=sys.stderr)
            return 1

        state = client.snapshot() or {}
        robots = client.robots() or []
        humans = client.humans() or []
        print(
            f"scenario applied  : {state.get('scenario_id')} ({state.get('scenario_name')})"
        )
        print(
            f"roster            : {', '.join(str(entry.get('id')) for entry in robots) or 'none'}"
        )
        print(f"crowd             : {len(humans)} agent(s)")
        print(f"sim_time_seconds  : {state.get('sim_time_seconds')} (the Unity clock)")
        print(f"scenario_time     : {client.scenario_time_seconds} (since the launch)")

        client.set_control_mode("ros")
        if not no_drive:
            print(f"driving           : {speed} m/s for {seconds:g}s")
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline and _pump_viewer(viewer):
                client.send_cmd_vel(speed, 0.0)
                time.sleep(0.05)
            client.send_cmd_vel(0.0, 0.0)
        elif render:
            _pump_viewer(viewer)

        state = client.snapshot() or {}
        robot = client.robot() or {}
        print(
            "robot pose        : "
            f"x={robot.get('x'):.2f} y={robot.get('y'):.2f} yaw={robot.get('yaw'):.2f} (ROS frame)"
        )
        print(f"sim_time_seconds  : {state.get('sim_time_seconds')}")
        print(f"scenario_time     : {client.scenario_time_seconds}")

        print("stopping          : pause, park every robot, stop the crowd")
        if not client.stop_scenario():
            print(f"stop failed       : {client.last_error}", file=sys.stderr)
            return 1
        print("scenario stopped  : the world stays in the scene, frozen")
        return 0
    finally:
        _close_viewer(viewer)
        client.stop()
        if not keep:
            removed = scenario.delete(scenario_id, directory=directory)
            print(f"scenario removed  : {removed}")
        else:
            print(f"scenario kept     : {path}")
