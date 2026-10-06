"""Tests for the ``robotsnap.client`` façade against the fake-Unity peer."""

import json
import socket
import threading
import time

import pytest

from robotsnap import topics
from robotsnap.bridge import codec
from robotsnap.client import (
    AGENTS_TOPIC,
    CONTROL_RESULT_TOPIC,
    CONTROL_TOPIC,
    STATE_TOPIC,
    RobotSNAPClient,
)

_WAIT_TIMEOUT = 2.0
_POLL_INTERVAL = 0.01

_SAMPLE_STATE = {
    "simulation_state": "playing",
    "playing": True,
    "paused": False,
    "stopped": False,
    "scenario_applied": True,
    "sim_time_seconds": 12.5,
    "time_scale": 1.0,
    "scenario_id": "corridor_1",
    "scenario_name": "corridor",
    "environment": "indoor",
    "map_name": "corridor_map",
    "map_width": 20,
    "map_height": 15,
    "map_resolution": 0.05,
    "map_origin_x": -1.0,
    "map_origin_y": 2.5,
    "human_count": 2,
    "humans": [
        {
            "id": 3,
            "x": 1.0,
            "y": 0.0,
            "z": 2.0,
            "vx": 0.1,
            "vy": 0.0,
            "vz": 0.0,
            "speed": 0.1,
            "goal": {"x": 5.0, "z": 2.0},
            "group": "G1",
            "controller": "sfm",
            "end_behavior": "stop",
        },
        {
            "id": 4,
            "x": -1.5,
            "y": 0.0,
            "z": 0.5,
            "vx": 0.0,
            "vy": 0.0,
            "vz": -0.2,
            "speed": 0.2,
            "goal": None,
            "group": None,
            "controller": "external",
            "end_behavior": "resume",
        },
    ],
    "robot": {"x": 0.5, "y": 0.0, "z": -0.25, "yaw": 0.1},
    "robot_has_goal": True,
    "robot_goal": {"x": 9.0, "y": 0.0, "z": 1.0},
    "camera_focus": "robot",
    "camera_focus_is_robot": True,
    "camera_focus_is_human": False,
    "camera_focus_agent_id": None,
    "camera_tool": "orbit",
    "camera_mode": "follow",
    "camera_pose": {"x": -4.0, "y": 1.5, "z": 0.0, "yaw": 0.0},
    "refreshed_at": 41.25,
    "robots": [
        {
            "id": "robot_1",
            "type": "jackal",
            "is_primary": True,
            "x": 0.5,
            "y": 0.0,
            "z": -0.25,
            "yaw": 0.1,
            "has_goal": True,
            "goal": {"x": 9.0, "y": 0.0, "z": 1.0},
            "start_pose": None,
            "target_pose": None,
        }
    ],
    "robot_count": 1,
}

#: A two-robot roster whose primary is the *second* entry, so a test tells the
#: ``is_primary`` resolution apart from "the first one".
_ROSTER_STATE = {
    **_SAMPLE_STATE,
    "robot": {"x": 2.0, "y": 0.0, "z": 1.0, "yaw": 0.0},
    "robot_has_goal": False,
    "robot_goal": None,
    "robots": [
        {
            "id": "robot_2",
            "type": "jackal",
            "is_primary": True,
            "x": 2.0,
            "y": 0.0,
            "z": 1.0,
            "yaw": 0.0,
            "has_goal": False,
            "goal": None,
            "start_pose": None,
            "target_pose": None,
        },
        {
            "id": "robot_1",
            "type": "husky",
            "is_primary": False,
            "x": -1.0,
            "y": 0.0,
            "z": 0.0,
            "yaw": 0.5,
            "has_goal": True,
            "goal": {"x": 1.0, "y": 0.0, "z": 2.0},
            "start_pose": None,
            "target_pose": None,
        },
    ],
    "robot_count": 2,
}

_SAMPLE_AGENTS = {
    "agents": [
        {
            "id": "human_3",
            "x": 1.0,
            "y": 0.0,
            "z": 2.0,
            "vx": 0.1,
            "vy": 0.0,
            "vz": 0.0,
            "visible": True,
        },
        {
            "id": "human_4",
            "x": -1.5,
            "y": 0.0,
            "z": 0.5,
            "vx": 0.0,
            "vy": 0.0,
            "vz": -0.2,
            "visible": False,
        },
    ],
    "frame": "robot",
}


def _wait_for(predicate, timeout=_WAIT_TIMEOUT):
    """Poll ``predicate`` with short sleeps; returns its last value."""
    deadline = time.monotonic() + timeout
    while True:
        value = predicate()
        if value:
            return value
        if time.monotonic() >= deadline:
            return value
        time.sleep(_POLL_INTERVAL)


def _wait_for_topic(bridge, topic, msg_type):
    """Wait until the bridge has learned a topic's type from the peer."""
    assert _wait_for(lambda: bridge.topic_types().get(topic) == msg_type)


# --- command round trip ----------------------------------------------------


def test_command_round_trip_returns_the_parsed_result(client, bridge, unity):
    """A command reaches Unity and its acknowledgement is parsed and returned."""
    _wait_for_topic(bridge, "simulation/control_result", "std_msgs/msg/String")
    unity.start_responder(sim_time_seconds=12.5)

    assert client.wait_until_ready(timeout=_WAIT_TIMEOUT) is True
    result = client.play()

    assert result == {
        "command": "play",
        "ok": True,
        "message": "",
        "sim_time_seconds": 12.5,
    }
    assert unity.commands == [(CONTROL_TOPIC, {"command": "play"})]
    assert client.is_connected is True
    assert client.last_error is None


def test_a_result_published_before_the_request_is_not_the_answer(client, bridge, unity):
    """Only a message newer than the request counts as its acknowledgement."""
    _wait_for_topic(bridge, "simulation/control_result", "std_msgs/msg/String")
    unity.publish_string(
        CONTROL_RESULT_TOPIC,
        json.dumps(
            {
                "command": "pause",
                "ok": False,
                "message": "stale",
                "sim_time_seconds": 1.0,
            }
        ),
    )
    assert _wait_for(lambda: bridge.latest(CONTROL_RESULT_TOPIC) is not None)

    client.command_timeout = 0.3
    assert client.pause() is None
    assert client.last_error is not None
    assert "no /simulation/control_result" in client.last_error


def test_every_typed_command_sends_its_documented_body(client, bridge, unity):
    """Each typed command maps to the documented JSON body and topic."""
    _wait_for_topic(bridge, "simulation/control_result", "std_msgs/msg/String")
    unity.start_responder()

    calls = [
        (lambda: client.play(), {"command": "play"}),
        (lambda: client.pause(), {"command": "pause"}),
        (lambda: client.toggle_pause(), {"command": "toggle_pause"}),
        (lambda: client.reset(), {"command": "reset"}),
        (
            lambda: client.reset(scenario="corridor", seed=7),
            {"command": "reset", "scenario": "corridor", "seed": 7},
        ),
        (
            lambda: client.load_scenario("corridor"),
            {
                "command": "load_scenario",
                "scenario": "corridor",
                "start": True,
                "apply": True,
            },
        ),
        (
            lambda: client.load_scenario("corridor", start=False, apply=False, seed=3),
            {
                "command": "load_scenario",
                "scenario": "corridor",
                "start": False,
                "apply": False,
                "seed": 3,
            },
        ),
        (
            lambda: client.set_time_scale(0.5),
            {"command": "set_time_scale", "time_scale": 0.5},
        ),
        (lambda: client.set_random_seed(11), {"command": "set_random_seed", "seed": 11}),
        (
            lambda: client.set_robot_goal(1.0, 2.0),
            {"command": "set_robot_goal", "x": 1.0, "z": 2.0},
        ),
        (
            lambda: client.set_robot_goal(1.0, 2.0, y=0.5),
            {"command": "set_robot_goal", "x": 1.0, "z": 2.0, "y": 0.5},
        ),
        (
            # ``y`` stays the third positional argument, the pre-roster form.
            lambda: client.set_robot_goal(1.0, 2.0, 0.5),
            {"command": "set_robot_goal", "x": 1.0, "z": 2.0, "y": 0.5},
        ),
        (
            # A snapshot reports the ROS frame: the goal it calls (-7, -3) is
            # (3, -7) on the simulator's own axes.
            lambda: client.set_robot_goal_ros(-7.0, -3.0, robot="robot_1"),
            {"command": "set_robot_goal", "x": 3.0, "z": -7.0, "robot": "robot_1"},
        ),
        (
            # The ROS height travels as the simulator's height, untouched.
            lambda: client.set_robot_goal_ros(5.0, -5.0, z=0.25),
            {"command": "set_robot_goal", "x": 5.0, "z": 5.0, "y": 0.25},
        ),
        (lambda: client.clear_robot_goal(), {"command": "clear_robot_goal"}),
        (lambda: client.stop_robot(), {"command": "stop_robot"}),
        (
            lambda: client.set_control_mode("ROS"),
            {"command": "set_control_mode", "mode": "ROS"},
        ),
        (
            lambda: client.set_agent_controller("external"),
            {"command": "set_agent_controller", "mode": "external"},
        ),
    ]

    for call, expected in calls:
        result = call()
        assert result is not None, f"{expected['command']} got no acknowledgement"
        assert result["ok"] is True
        assert unity.commands[-1] == (CONTROL_TOPIC, expected)


# --- state -----------------------------------------------------------------


def test_snapshot_parses_the_state_json(client, bridge, unity):
    """``/simulation/state`` bodies are parsed, not stringified."""
    _wait_for_topic(bridge, "simulation/state", "std_msgs/msg/String")
    unity.publish_state(_SAMPLE_STATE)

    assert _wait_for(lambda: client.snapshot() is not None)
    state = client.snapshot()
    assert state["simulation_state"] == "playing"
    assert state["playing"] is True
    assert state["sim_time_seconds"] == pytest.approx(12.5)
    assert state["map_resolution"] == pytest.approx(0.05)
    assert state["camera_mode"] == "follow"

    assert client.humans() == _SAMPLE_STATE["humans"]
    assert client.robot() == _SAMPLE_STATE["robots"][0]
    assert client.robots() == _SAMPLE_STATE["robots"]

    # The general accessors hand back the decoded rosbags message.
    message = client.last_message(STATE_TOPIC)
    assert message.data == json.dumps(_SAMPLE_STATE)
    assert client.state(STATE_TOPIC) is message
    assert client.last_message("/robot0/simulation/state") is message


def test_snapshot_reaches_the_typed_state_model(client, bridge, unity):
    """The bridge turns the same body into the typed world-frame state."""
    _wait_for_topic(bridge, "simulation/state", "std_msgs/msg/String")
    unity.publish_state(_SAMPLE_STATE)

    assert _wait_for(lambda: len(bridge.snapshot().humans) == 2)
    snapshot = bridge.snapshot()
    assert [human.human_id for human in snapshot.humans] == [3, 4]
    assert snapshot.humans[0].controller == "sfm"
    assert snapshot.humans[1].end_behavior == "resume"
    assert snapshot.humans[1].goal is None
    assert snapshot.simulation is not None
    assert snapshot.simulation.scenario_id == "corridor_1"
    assert snapshot.simulation.robot_goal.x == pytest.approx(9.0)
    assert snapshot.simulation.camera_pose.y == pytest.approx(1.5)


def test_snapshot_is_none_before_anything_arrives(client, bridge, unity):
    """No state yet is ``None``, not an exception and not an empty dict."""
    assert client.snapshot() is None
    assert client.humans() is None
    assert client.robots() is None
    assert client.robot() is None
    assert client.agents() is None


def test_snapshot_survives_a_malformed_body(client, bridge, unity):
    """A non-JSON body is reported through ``last_error``, never raised."""
    _wait_for_topic(bridge, "simulation/state", "std_msgs/msg/String")
    unity.publish_string(STATE_TOPIC, "not json")

    assert _wait_for(lambda: bridge.latest(STATE_TOPIC) is not None)
    assert client.snapshot() is None
    assert "not valid JSON" in client.last_error


# --- robots ----------------------------------------------------------------


def test_robots_reads_the_roster_and_resolves_the_primary(client, bridge, unity):
    """``robots()`` is the whole roster; ``robot()`` picks by id or by ``is_primary``."""
    _wait_for_topic(bridge, "simulation/state", "std_msgs/msg/String")
    unity.publish_state(_ROSTER_STATE)

    assert _wait_for(lambda: client.robots() is not None)
    assert [entry["id"] for entry in client.robots()] == ["robot_2", "robot_1"]

    # No id asks for the primary, which is the second entry here.
    assert client.robot()["id"] == "robot_2"
    assert client.robot("robot_1")["id"] == "robot_1"
    assert client.robot("robot_2")["is_primary"] is True
    assert client.robot("robot_7") is None


def test_robots_is_an_empty_list_when_the_scene_has_none(client, bridge, unity):
    """An empty roster is a valid answer, not ``None``."""
    _wait_for_topic(bridge, "simulation/state", "std_msgs/msg/String")
    unity.publish_state({**_SAMPLE_STATE, "robots": [], "robot_count": 0})

    assert _wait_for(lambda: (client.snapshot() or {}).get("robot_count") == 0)
    assert client.robots() == []


def test_robot_falls_back_to_the_legacy_key_without_a_roster(client, bridge, unity):
    """An older snapshot has no ``robots`` array, and still answers for the primary."""
    legacy = {
        key: value
        for key, value in _SAMPLE_STATE.items()
        if key not in ("robots", "robot_count")
    }
    unity.publish_string(STATE_TOPIC, json.dumps(legacy))

    assert _wait_for(lambda: client.robot() is not None)
    assert client.robot() == legacy["robot"]
    assert client.robots() is None


def test_every_robot_command_names_its_robot_and_echoes_it(client, bridge, unity):
    """All four robot commands take the ``robot`` key, and the answer echoes it."""
    _wait_for_topic(bridge, "simulation/control_result", "std_msgs/msg/String")
    unity.publish_state(_ROSTER_STATE)
    unity.start_responder()

    named = [
        (
            lambda: client.set_robot_goal(1.0, 2.0, robot="robot_1"),
            {"command": "set_robot_goal", "x": 1.0, "z": 2.0, "robot": "robot_1"},
        ),
        (
            lambda: client.clear_robot_goal(robot="robot_1"),
            {"command": "clear_robot_goal", "robot": "robot_1"},
        ),
        (
            lambda: client.stop_robot(robot="robot_1"),
            {"command": "stop_robot", "robot": "robot_1"},
        ),
        (
            lambda: client.set_control_mode("ROS", robot="robot_1"),
            {"command": "set_control_mode", "mode": "ROS", "robot": "robot_1"},
        ),
    ]
    for call, expected in named:
        result = call()
        assert result["ok"] is True
        assert result["robot"] == "robot_1"
        assert unity.commands[-1] == (CONTROL_TOPIC, expected)

    # No robot named: the simulator's primary applies, and the answer says which.
    unnamed = [
        (
            lambda: client.set_robot_goal(1.0, 2.0),
            {"command": "set_robot_goal", "x": 1.0, "z": 2.0},
        ),
        (lambda: client.clear_robot_goal(), {"command": "clear_robot_goal"}),
        (lambda: client.stop_robot(), {"command": "stop_robot"}),
        (
            lambda: client.set_control_mode("ROS"),
            {"command": "set_control_mode", "mode": "ROS"},
        ),
    ]
    for call, expected in unnamed:
        result = call()
        assert result["robot"] == "robot_2"
        assert unity.commands[-1] == (CONTROL_TOPIC, expected)


def test_an_unknown_robot_is_refused_and_names_the_roster(client, bridge, unity):
    """The contract refuses an id the roster does not carry, and lists the ones it has."""
    _wait_for_topic(bridge, "simulation/control_result", "std_msgs/msg/String")
    unity.publish_state(_ROSTER_STATE)
    unity.start_responder()

    result = client.set_control_mode("ROS", robot="robot_9")

    assert result is not None
    assert result["ok"] is False
    assert "robot_9" in result["message"]
    assert "robot_1" in result["message"]
    assert "robot_2" in result["message"]
    assert unity.commands[-1] == (
        CONTROL_TOPIC,
        {"command": "set_control_mode", "mode": "ROS", "robot": "robot_9"},
    )


# --- agents ----------------------------------------------------------------


def test_agents_parses_the_robot_frame_body(client, bridge, unity):
    """``/simulation/agents`` is parsed into its flat agent dicts."""
    _wait_for_topic(bridge, "simulation/agents", "std_msgs/msg/String")
    unity.publish_agents(_SAMPLE_AGENTS)

    assert _wait_for(lambda: client.agents() is not None)
    body = client.agents()
    assert body["frame"] == "robot"
    assert [agent["id"] for agent in body["agents"]] == ["human_3", "human_4"]
    assert body["agents"][0]["x"] == pytest.approx(1.0)
    assert body["agents"][1]["visible"] is False

    # The typed snapshot carries the same agents.
    assert _wait_for(lambda: len(bridge.snapshot().agents) == 2)
    assert bridge.snapshot().agents[0].agent_id == "human_3"


def test_agents_rejects_a_body_that_is_not_the_contract(client, bridge, unity):
    """A wrong shape is reported through ``last_error``, never raised."""
    _wait_for_topic(bridge, "simulation/agents", "std_msgs/msg/String")
    unity.publish_string(AGENTS_TOPIC, json.dumps([{"id": "human_3"}]))

    assert _wait_for(lambda: bridge.latest(AGENTS_TOPIC) is not None)
    assert client.agents() is None
    assert "is invalid" in client.last_error


def test_agents_of_one_robot_reads_its_namespaced_topic(client, bridge, unity):
    """Naming a robot reads what *that* robot's detector sees."""
    _wait_for_topic(bridge, "simulation/agents", "std_msgs/msg/String")
    unity.register_publisher("/robot_2/simulation/agents", "std_msgs/String")
    unity.publish_agents(_SAMPLE_AGENTS)
    unity.publish_string(
        "/robot_2/simulation/agents",
        json.dumps(
            {
                "agents": [
                    {
                        "id": "human_9",
                        "x": 0.4,
                        "y": 0.0,
                        "z": 1.0,
                        "vx": 0.0,
                        "vy": 0.0,
                        "vz": 0.0,
                        "visible": True,
                    }
                ],
                "frame": "robot",
            }
        ),
    )

    # The bridge resolves a prefix tolerantly, so wait for the exact stream
    # rather than for ``agents(robot=...)`` to stop falling back to the legacy one.
    assert _wait_for(
        lambda: bridge.topic_counts().get("robot_2/simulation/agents", 0) >= 1
    )
    assert [agent["id"] for agent in client.agents(robot="robot_2")["agents"]] == [
        "human_9"
    ]
    # The unprefixed read is untouched: it is the primary robot's stream.
    assert [agent["id"] for agent in client.agents()["agents"]] == [
        "human_3",
        "human_4",
    ]


# --- humans ----------------------------------------------------------------


def test_set_human_velocities_encodes_the_documented_body(client, bridge, unity):
    """Tuples and dicts both become a ``humans`` command on the control topic."""
    _wait_for_topic(bridge, "simulation/control_result", "std_msgs/msg/String")
    _wait_for_topic(bridge, "simulation/state", "std_msgs/msg/String")
    unity.publish_state(_SAMPLE_STATE)
    assert _wait_for(lambda: client.humans() is not None)
    unity.start_responder()

    result = client.set_human_velocities(
        [(3, 1.0, 0.0), {"id": 4, "vx": 0.0, "vz": -0.5}]
    )

    assert result is not None
    assert result["ok"] is True
    assert result["command"] == "humans"
    assert result["unknown_ids"] == []
    assert unity.commands[-1] == (
        CONTROL_TOPIC,
        {
            "command": "humans",
            "commands": [
                {"id": 3, "vx": 1.0, "vz": 0.0},
                {"id": 4, "vx": 0.0, "vz": -0.5},
            ]
        },
    )


def test_set_human_velocities_reports_unknown_ids(client, bridge, unity):
    """The answer names the ids the scene does not hold, and says ``ok: false``."""
    _wait_for_topic(bridge, "simulation/control_result", "std_msgs/msg/String")
    _wait_for_topic(bridge, "simulation/state", "std_msgs/msg/String")
    unity.publish_state(_SAMPLE_STATE)
    assert _wait_for(lambda: client.humans() is not None)
    unity.start_responder()

    result = client.set_human_velocities([(3, 1.0, 0.0), (99, 0.0, 0.0)])

    assert result is not None
    assert result["command"] == "humans"
    assert result["ok"] is False
    assert result["unknown_ids"] == [99]
    assert unity.commands[-1] == (
        CONTROL_TOPIC,
        {
            "command": "humans",
            "commands": [
                {"id": 3, "vx": 1.0, "vz": 0.0},
                {"id": 99, "vx": 0.0, "vz": 0.0},
            ],
        },
    )


def test_stop_human_and_stop_humans_use_stop_entries(client, bridge, unity):
    """Stopping uses the ``stop: true`` entry the contract documents."""
    _wait_for_topic(bridge, "simulation/control_result", "std_msgs/msg/String")
    _wait_for_topic(bridge, "simulation/state", "std_msgs/msg/String")
    unity.start_responder()

    client.stop_human(7)
    assert unity.commands[-1] == (
        CONTROL_TOPIC,
        {"command": "humans", "commands": [{"id": 7, "stop": True}]},
    )

    unity.publish_state(_SAMPLE_STATE)
    assert _wait_for(lambda: client.humans() is not None)

    client.stop_humans()
    assert unity.commands[-1] == (
        CONTROL_TOPIC,
        {
            "command": "humans",
            "commands": [{"id": 3, "stop": True}, {"id": 4, "stop": True}],
        },
    )


def test_stop_humans_without_a_snapshot_reports_why(client, bridge, unity):
    """``stop_humans`` needs a snapshot to learn the ids, and says so."""
    assert client.humans() is None
    assert client.stop_humans() is None
    assert "no human id" in client.last_error


def test_set_human_velocities_rejects_unknown_entry_shapes(client, bridge, unity):
    """A programming error in the entry shape raises instead of sending junk."""
    with pytest.raises(TypeError):
        client.set_human_velocities(["human_3"])
    with pytest.raises(ValueError):
        client.set_human_velocities([(3, 1.0)])


# --- velocity commands -----------------------------------------------------


def test_send_cmd_vel_routes_a_named_robot_to_its_topic(client, bridge, unity):
    """A robot named by id is driven on ``/robot_<id>/cmd_vel``, Twist included."""
    assert client.send_cmd_vel(0.4, -0.2, robot="robot_2") is True

    destination, payload = unity.recv_frame()
    assert destination == "/robot_2/cmd_vel"
    twist = codec.decode(topics.TWIST_TYPE, payload)
    assert twist.linear.x == pytest.approx(0.4)
    assert twist.linear.y == pytest.approx(0.0)
    assert twist.angular.z == pytest.approx(-0.2)


def test_send_cmd_vel_defaults_to_the_legacy_topic(client, bridge, unity):
    """No robot named keeps the old ``/cmd_vel``, which reaches the primary."""
    assert client.send_cmd_vel(0.5, 0.1) is True

    destination, payload = unity.recv_frame()
    assert destination == "/cmd_vel"
    twist = codec.decode(topics.TWIST_TYPE, payload)
    assert twist.linear.x == pytest.approx(0.5)
    assert twist.angular.z == pytest.approx(0.1)


# --- lifecycle without Unity ----------------------------------------------


def test_wait_for_scenario_returns_once_the_state_reports_it(client, unity):
    """The scenario a client is waiting for is the one /simulation/state names."""
    unity.publish_state({**_SAMPLE_STATE, "scenario_id": "crowd", "scenario_applied": True})

    assert client.wait_for_scenario("crowd", timeout=_WAIT_TIMEOUT) is True
    assert client.last_error is None


def test_wait_for_scenario_waits_for_the_agents_to_be_spawned(client, unity):
    """A scenario that is loaded but not applied is not the world a caller asked for."""
    unity.publish_state({**_SAMPLE_STATE, "scenario_id": "crowd", "scenario_applied": False})

    assert client.wait_for_scenario("crowd", applied=False, timeout=_WAIT_TIMEOUT) is True
    assert client.wait_for_scenario("crowd", timeout=0.2) is False
    assert "not applied within" in client.last_error


def test_wait_for_scenario_times_out_on_another_scenario(client, unity):
    """The load that was queued behind another one is only done once it is the current one."""
    unity.publish_state({**_SAMPLE_STATE, "scenario_id": "default", "scenario_applied": True})

    assert client.wait_for_scenario("crowd", timeout=0.2) is False
    assert "not applied within" in client.last_error

    unity.publish_state({**_SAMPLE_STATE, "scenario_id": "crowd", "scenario_applied": True})

    assert client.wait_for_scenario("crowd", timeout=_WAIT_TIMEOUT) is True


# --- scenarios: create, launch, stop --------------------------------------=


def _answer_next(unity, result, timeout=_WAIT_TIMEOUT):
    """Answer the next control command with ``result``, from a thread."""

    def run():
        deadline = time.monotonic() + timeout
        unity.sock.settimeout(0.2)
        while time.monotonic() < deadline:
            try:
                destination, _ = unity.recv_frame()
            except (socket.timeout, OSError):
                continue
            if destination.startswith("__"):
                continue
            unity.publish_string(CONTROL_RESULT_TOPIC, json.dumps(result))
            return

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def test_launch_scenario_loads_it_and_waits_for_the_world(client, bridge, unity):
    """The pair a script needs: the command, then the scenario the state reports."""
    _wait_for_topic(bridge, "simulation/control_result", "std_msgs/msg/String")
    unity.start_responder()
    unity.publish_state({**_SAMPLE_STATE, "scenario_id": "demo", "scenario_applied": True})

    assert client.launch_scenario("demo", timeout=_WAIT_TIMEOUT) is True
    assert unity.commands[-1] == (
        CONTROL_TOPIC,
        {
            "command": "load_scenario",
            "scenario": "demo",
            "start": True,
            "apply": True,
        },
    )
    assert client.last_error is None


def test_launch_scenario_passes_the_seed(client, bridge, unity):
    """A seed is only on the wire when the caller has one."""
    _wait_for_topic(bridge, "simulation/control_result", "std_msgs/msg/String")
    unity.start_responder()
    unity.publish_state({**_SAMPLE_STATE, "scenario_id": "demo", "scenario_applied": True})

    assert client.launch_scenario("demo", seed=7, timeout=_WAIT_TIMEOUT) is True
    assert unity.commands[-1][1]["seed"] == 7


def test_launch_scenario_reports_a_refusal(client, bridge, unity):
    """A refusal is an answer, not a timeout: its message is what is kept."""
    _wait_for_topic(bridge, "simulation/control_result", "std_msgs/msg/String")
    _answer_next(
        unity,
        {
            "command": "load_scenario",
            "ok": False,
            "message": "unknown scenario 'demo'; available scenarios: default, crowd",
            "sim_time_seconds": 1.0,
        },
    )

    assert client.launch_scenario("demo", timeout=0.3) is False
    assert "available scenarios: default, crowd" in client.last_error


def test_launch_scenario_reports_a_world_that_never_arrives(client, bridge, unity):
    """An accepted load that never becomes the current scenario is still a failure."""
    _wait_for_topic(bridge, "simulation/control_result", "std_msgs/msg/String")
    unity.start_responder()
    unity.publish_state({**_SAMPLE_STATE, "scenario_id": "default", "scenario_applied": True})

    assert client.launch_scenario("demo", timeout=0.3) is False
    assert "not applied within" in client.last_error


def test_stop_scenario_pauses_parks_and_stops_the_crowd(client, bridge, unity):
    """Stopping freezes everything that could keep moving under a paused flag."""
    _wait_for_topic(bridge, "simulation/control_result", "std_msgs/msg/String")
    unity.publish_state(_ROSTER_STATE)
    unity.start_responder()

    assert client.stop_scenario() is True
    assert [body for _, body in unity.commands] == [
        {"command": "pause"},
        # The roster's own order, which is the scenario's.
        {"command": "stop_robot", "robot": "robot_2"},
        {"command": "stop_robot", "robot": "robot_1"},
        {
            "command": "humans",
            "commands": [{"id": 3, "stop": True}, {"id": 4, "stop": True}],
        },
    ]


def test_stop_scenario_without_a_roster_parks_the_primary(client, bridge, unity):
    """The legacy single-robot session is stopped without naming a robot."""
    _wait_for_topic(bridge, "simulation/control_result", "std_msgs/msg/String")
    unity.publish_state({**_SAMPLE_STATE, "robots": [], "robot_count": 0, "humans": []})
    unity.start_responder()

    assert client.stop_scenario() is True
    assert [body for _, body in unity.commands] == [
        {"command": "pause"},
        {"command": "stop_robot"},
    ]


def test_create_scenario_writes_the_file_and_launches_it(client, bridge, unity, tmp_path):
    """One call writes the YAML where Unity reads it, then loads it."""
    from robotsnap import scenario

    _wait_for_topic(bridge, "simulation/control_result", "std_msgs/msg/String")
    unity.start_responder()
    unity.publish_state({**_SAMPLE_STATE, "scenario_id": "demo", "scenario_applied": True})

    path = client.create_scenario(
        "demo",
        directory=tmp_path,
        launch=True,
        timeout=_WAIT_TIMEOUT,
        map_name="basic/crowd",
        robots=[scenario.robot("robot_1", "jackal", (1.0, 1.0), (5.0, 1.0))],
    )

    assert path == tmp_path / "demo.yaml"
    assert "scenario_info:" in path.read_text(encoding="utf-8")
    assert unity.commands[-1][1]["command"] == "load_scenario"


def test_create_scenario_reports_a_document_it_cannot_build(client, tmp_path):
    """A scenario with no robot is refused here, where the caller can see it."""
    path = client.create_scenario(
        "empty", directory=tmp_path, map_name="basic", robots=[]
    )

    assert path is None
    assert "at least one robot" in client.last_error
    assert list(tmp_path.iterdir()) == []


# --- the episode clock -----------------------------------------------------=


def test_scenario_time_is_measured_from_the_launch(client, bridge, unity):
    """The simulator clock keeps running; the episode clock starts at the launch."""
    _wait_for_topic(bridge, "simulation/control_result", "std_msgs/msg/String")
    unity.start_responder()
    unity.publish_state({**_SAMPLE_STATE, "sim_time_seconds": 100.0, "scenario_id": "demo"})
    assert _wait_for(lambda: (client.snapshot() or {}).get("sim_time_seconds") == 100.0)

    assert client.scenario_time_seconds is None  # nothing launched yet
    assert client.launch_scenario("demo", timeout=_WAIT_TIMEOUT) is True
    assert client.scenario_time_seconds == pytest.approx(0.0)

    unity.publish_state({**_SAMPLE_STATE, "sim_time_seconds": 112.5, "scenario_id": "demo"})
    assert _wait_for(lambda: (client.snapshot() or {}).get("sim_time_seconds") == 112.5)

    assert client.snapshot()["sim_time_seconds"] == 112.5  # the age of the session
    assert client.scenario_time_seconds == pytest.approx(12.5)  # the age of the episode


def test_scenario_time_can_be_marked_by_hand(client, unity):
    """A reset moves the anchor by hand, since its answer precedes the new world."""
    unity.publish_state({**_SAMPLE_STATE, "sim_time_seconds": 8.0})
    assert _wait_for(lambda: (client.snapshot() or {}).get("sim_time_seconds") == 8.0)

    assert client.mark_scenario_start() == 8.0
    assert client.scenario_time_seconds == pytest.approx(0.0)

    unity.publish_state({**_SAMPLE_STATE, "sim_time_seconds": 3.0})
    assert _wait_for(lambda: (client.snapshot() or {}).get("sim_time_seconds") == 3.0)

    assert client.scenario_time_seconds == pytest.approx(-5.0)


def test_wait_until_ready_needs_the_peer_to_register(client, bridge, bare_peer):
    """A connected socket is not readiness: the peer must have subscribed.

    A command written before Unity's listener exists goes to a topic nobody
    reads and is silently dropped, which a caller sees as a hang. The publisher
    registrations are not the signal either: the state and the map announce
    themselves before the command listener does, which is how a client used to
    be let through early.
    """
    assert bridge.is_connected is True
    assert bridge.topic_types()  # seeded, and therefore not evidence of anything
    assert bridge.announced_topics() == {}

    bare_peer.register_publisher("/simulation/state", "std_msgs/String")
    assert client.wait_until_ready(timeout=0.3) is False
    assert "never subscribed to" in client.last_error

    bare_peer.register_subscriber("/simulation/control", "std_msgs/String")
    assert _wait_for(lambda: bridge.subscriptions())
    assert bridge.subscriptions() == {"simulation/control": "/simulation/control"}
    assert client.wait_until_ready(timeout=_WAIT_TIMEOUT) is True


def test_announced_topics_is_cleared_when_the_session_ends(client, bridge, bare_peer):
    """A new peer re-registers, so the previous evidence must not survive it."""
    bare_peer.register_publisher("/simulation/state", "std_msgs/String")
    assert _wait_for(lambda: "simulation/state" in bridge.announced_topics())

    bare_peer.close()

    assert _wait_for(lambda: bridge.announced_topics() == {})


def test_facade_without_a_connection_is_quiet():
    """With nothing connected each call returns a value, and ``last_error`` says why."""
    with RobotSNAPClient(port=0, host="127.0.0.1") as client:
        assert client.is_connected is False
        assert client.wait_until_ready(timeout=0.05) is False
        assert client.snapshot() is None
        assert client.humans() is None
        assert client.robots() is None
        assert client.robot() is None
        assert client.agents() is None
        assert client.last_message("odom") is None

        assert client.play() is None
        assert client.last_error is not None
        assert "no Unity peer" in client.last_error

        assert client.set_human_velocities([(3, 1.0, 0.0)]) is None
        assert client.stop_human(3) is None
        assert client.send_cmd_vel(0.5, 0.0) is False
        assert client.send_cmd_vel(0.5, 0.0, robot="robot_2") is False
