"""Tests for the topic table, and for it being the only place topics are named."""

import ast
from pathlib import Path

from robotsnap import topics

#: The package under test, so a stray topic string is found wherever it lives.
_PACKAGE = Path(__file__).resolve().parents[2] / "src" / "robotsnap"
_TOPICS_MODULE = _PACKAGE / "topics.py"


def test_the_surface_is_exactly_the_contract():
    """The ten topics the simulator exchanges, with their types."""
    assert topics.TOPIC_TYPES == {
        "/clock": "rosgraph_msgs/msg/Clock",
        "/odom": "nav_msgs/msg/Odometry",
        "/scan": "sensor_msgs/msg/LaserScan",
        "/map": "nav_msgs/msg/OccupancyGrid",
        "/cmd_vel": "geometry_msgs/msg/Twist",
        "/simulation/state": "std_msgs/msg/String",
        "/simulation/agents": "std_msgs/msg/String",
        "/simulation/control": "std_msgs/msg/String",
        "/simulation/control_result": "std_msgs/msg/String",
        "/reset_done": "std_msgs/msg/Bool",
    }
    assert topics.base_types()["simulation/agents"] == topics.STRING_TYPE
    for topic in topics.JSON_TOPICS:
        assert topics.TOPIC_TYPES[topic] == topics.STRING_TYPE


def test_the_dead_topics_are_gone():
    """Nothing of the pre-unification surface survives in the table."""
    dead = [
        "/agents",
        "/agents/global",
        "/agents/pose",
        "/human_pose",
        "/robot_pose",
        "/robot_odom",
        "/global_path",
        "/local_goal_from_robot",
        "/local_goal_from_map",
        "/simulation/humans/control",
        "/simulation/people",
        "/simulation/map",
        "/simulation/scene_info",
    ]
    for topic in dead:
        assert topic not in topics.TOPIC_TYPES


def test_no_other_module_names_a_topic():
    """A topic name written anywhere but topics.py would drift on a rename."""
    names = set(topics.TOPIC_TYPES) | {
        topics.base(topic) for topic in topics.TOPIC_TYPES
    }
    # The words a scenario file is written with are a schema of their own, and
    # one of them, "map", is also the leaf of a topic. It is a document key
    # there, not a stream, so that one word is allowed in that one module.
    schema_words = {"scenario.py": {"map"}}
    offenders = [
        f"{path.relative_to(_PACKAGE)}: {literal!r}"
        for path in sorted(_PACKAGE.rglob("*.py"))
        if path != _TOPICS_MODULE
        for literal in _string_literals(path)
        if literal in names
        and literal not in schema_words.get(path.name, set())
    ]
    assert offenders == []


def test_robot_topic_builds_the_per_robot_name():
    """One rule for every per-robot stream: ``/robot_<id>/<topic>``."""
    assert topics.robot_topic("robot_2", topics.SCAN) == "/robot_2/scan"
    assert topics.robot_topic("robot_2", topics.ODOM) == "/robot_2/odom"
    assert (
        topics.robot_topic("robot_2", topics.SIMULATION_AGENTS)
        == "/robot_2/simulation/agents"
    )
    assert topics.robot_topic("robot_2", topics.CMD_VEL) == "/robot_2/cmd_vel"


def test_robot_topic_tolerates_slashes_and_a_namespaced_id():
    """Slashes on either argument, and an id already carrying its namespace."""
    assert topics.robot_topic("/robot_2/", "/scan") == "/robot_2/scan"
    assert topics.robot_topic("robot_2", "scan") == "/robot_2/scan"
    assert topics.robot_topic("/robot_2", "scan/") == "/robot_2/scan"
    assert topics.robot_topic("2", topics.SCAN) == "/robot_2/scan"
    assert topics.robot_topic("robot_2", "/scan") == topics.robot_topic(
        "/robot_2/", "scan"
    )


def test_robot_topic_is_exported():
    """The helper is part of the module's public surface."""
    assert "robot_topic" in topics.__all__


def test_no_other_module_needs_the_simulation_msgs_package():
    """No custom ``.msg`` is needed any more, so nothing may reference one."""
    offenders = [
        f"{path.relative_to(_PACKAGE)}: {literal!r}"
        for path in sorted(_PACKAGE.rglob("*.py"))
        for literal in _string_literals(path)
        if "simulation_msgs" in literal or literal == "sim_msgs"
    ]
    assert offenders == []


def _string_literals(path: Path):
    """Every string constant of a module, docstrings excluded."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            first = node.body[0] if node.body else None
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                docstrings.add(id(first.value))

    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    ]
