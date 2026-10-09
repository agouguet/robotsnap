"""Tests for ``robotsnap.cli`` and the runs behind the one entry point.

The parsers and the observation handling are checked without a session; the two
short runs go against the fake Unity peer of ``tests/unity_peer.py``, through
the ``client`` and ``unity`` fixtures. Nothing here needs Unity, and nothing
here needs PyTorch - the training import is deferred to its own run.
"""

import sys
import time
import types

import numpy as np
import pytest

from robotsnap import cli, runs

_WAIT_TIMEOUT = 2.0
_CONTROL_PERIOD = 0.05
_EPISODE_SECONDS = 0.2

#: The commands by their own name, in the order ``--help`` lists them.
COMMANDS = ("watch", "episode", "goal", "train", "scenario", "bench", "benchmark")
#: Spellings kept accepted for scripts written before a command was renamed.
COMMAND_ALIASES = ("viewer", "infer")


# -- the fake session -------------------------------------------------------


def _publish_world(
    unity, x=0.0, y=0.0, yaw=0.0, goal=(5.0, 0.0), sim_time=10.0, robots=None
):
    """Publish the whole world one environment read needs, in one go.

    ``robots`` replaces the single-robot roster, for a test that drives a fleet
    and needs the peer to know the id its commands name.
    """
    entry = {
        "id": "robot_1",
        "type": "jackal",
        "is_primary": True,
        "x": x,
        "y": y,
        "z": 0.0,
        "yaw": yaw,
        "has_goal": goal is not None,
        "goal": None if goal is None else {"x": goal[0], "y": goal[1], "z": 0.0},
        "start_pose": None,
        "target_pose": None,
    }
    unity.publish_state(
        {
            "simulation_state": "running",
            "playing": True,
            "paused": False,
            "stopped": False,
            "scenario_applied": True,
            "scenario_id": "demo",
            "scenario_name": "demo",
            "sim_time_seconds": sim_time,
            "time_scale": 1.0,
            "map_width": 12,
            "map_height": 12,
            "map_resolution": 1.0,
            "map_origin_x": 0.0,
            "map_origin_y": 0.0,
            "human_count": 0,
            "humans": [],
            "robots": [entry] if robots is None else robots,
        }
    )
    unity.publish_odom(x=x, y=y, yaw=yaw, linear_x=0.0, angular_z=0.0, stamp=sim_time)
    unity.publish_scan(
        [3.0] * 8,
        angle_min=-np.pi / 4,
        angle_increment=np.pi / 16,
        range_max=5.0,
        stamp=sim_time,
    )
    unity.publish_map(
        [[0] * 12 for _ in range(12)],
        resolution=1.0,
        origin_x=0.0,
        origin_y=0.0,
        stamp=sim_time,
    )
    unity.publish_agents({"agents": [], "frame": "robot"})


def _start(unity, bridge, **world):
    """Bring the fake session up: reader thread, closed loop, one world."""
    deadline = time.monotonic() + _WAIT_TIMEOUT
    while bridge.topic_types().get("simulation/control_result") != "std_msgs/msg/String":
        assert time.monotonic() < deadline, "the peer never registered its streams"
        time.sleep(0.01)
    unity.start_responder()
    _publish_world(unity, **world)


@pytest.fixture
def project(tmp_path):
    """A throwaway Unity project, so a scenario file has somewhere to go."""
    root = tmp_path / "Project"
    (root / "Assets" / "StreamingAssets" / "Scenarios").mkdir(parents=True)
    return root


# -- the command line -------------------------------------------------------


def test_top_help_lists_every_subcommand(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--help"])
    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    for command in COMMANDS:
        assert command in out
    for alias in COMMAND_ALIASES:
        assert alias in out


@pytest.mark.parametrize("command", COMMANDS)
def test_every_subcommand_has_its_own_help(command, capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main([command, "--help"])
    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    assert "usage:" in out
    assert f"python -m robotsnap {command}" in out


def test_commands_that_do_not_need_unity_print_their_help(capsys):
    for command in ("bench", "scenario"):
        with pytest.raises(SystemExit) as exit_info:
            cli.main([command, "--help"])
        assert exit_info.value.code == 0
        assert "usage:" in capsys.readouterr().out


def test_the_commands_that_take_hyperparameters_accept_a_config_file(capsys):
    """``--config`` is where a run's hyperparameters come from; the flag exists.

    The file itself is applied in a second pass of the parser - see
    ``robotsnap.cli.main`` - so reading the flag back is what a command module
    has to get right, and ``tests/robotsnap/config`` covers what it does.
    """
    for command in ("train", "play", "bench", "benchmark"):
        extra = ["--load", "policy.pt"] if command == "play" else []
        args = cli.build_parser().parse_args([command, "--config", "ppo", *extra])
        assert args.config == "ppo"


def test_benchmark_and_bench_are_two_different_questions(capsys):
    """``bench`` times the session; ``benchmark`` scores a suite of scenarios."""
    for spelling in ("benchmark", "campaign"):
        with pytest.raises(SystemExit) as exit_info:
            cli.main([spelling, "--help"])
        assert exit_info.value.code == 0
        assert "python -m robotsnap benchmark" in capsys.readouterr().out


def test_observation_options_default_to_the_historical_ones():
    args = cli.build_parser().parse_args(["bench"])
    assert args.observations is None
    assert args.observation_params is None
    assert args.observation_structure == "flat"
    assert args.max_agents == 8
    assert args.lidar_bins == 48


def test_observation_params_parses_a_json_object():
    args = cli.build_parser().parse_args(
        [
            "bench",
            "--observation-params",
            '{"lidar": {"bins": 24}, "agents": {"max": 4}}',
        ]
    )
    assert args.observation_params == {"lidar": {"bins": 24}, "agents": {"max": 4}}


def test_observation_params_reports_bad_json(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["bench", "--observation-params", "{oops"])
    assert exit_info.value.code == 2
    assert "not valid JSON" in capsys.readouterr().err


def test_observation_params_rejects_a_non_object(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["bench", "--observation-params", "[1, 2]"])
    assert exit_info.value.code == 2
    assert "JSON object" in capsys.readouterr().err


def test_observations_takes_names_as_they_are_written():
    args = cli.build_parser().parse_args(["episode", "--observations", "pose, goal ,lidar"])
    assert args.observations == "pose, goal ,lidar"


def test_observations_takes_a_json_object_too():
    """A part and the parameters that size it, in one argument."""
    args = cli.build_parser().parse_args(
        ["episode", "--observations", '{"lidar": {"bins": 24}, "pose": {}}']
    )
    assert args.observations == {"lidar": {"bins": 24}, "pose": {}}


def test_observations_reports_bad_json_rather_than_a_bad_part_name(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["episode", "--observations", "{oops"])
    assert exit_info.value.code == 2
    assert "not valid JSON" in capsys.readouterr().err


# -- the wiring of the sub-commands -----------------------------------------


def test_episode_forwards_the_observation_and_environment_options(monkeypatch):
    captured = {}

    def fake(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(runs, "run_random_episode", fake)
    code = cli.main(
        [
            "episode",
            "--observations", "pose,goal",
            "--observation-params", '{"lidar": {"bins": 24}}',
            "--observation-structure", "dict",
            "--max-agents", "4",
            "--lidar-bins", "24",
            "--goal-radius", "0.8",
        ]
    )
    assert code == 0
    assert captured["observations"] == "pose,goal"
    assert captured["observation_params"] == {"lidar": {"bins": 24}}
    assert captured["observation_structure"] == "dict"
    assert captured["max_agents"] == 4
    assert captured["lidar_bins"] == 24
    assert captured["goal_radius"] == 0.8


def test_goal_keeps_its_own_scenario_and_episode_default(monkeypatch):
    captured = {}

    def fake(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(runs, "run_goal_episode", fake)
    assert cli.main(["goal"]) == 0
    assert captured["scenario_id"] == "python_social_demo"
    assert captured["episodes"] == 1
    assert captured["control_period"] == 0.2
    assert captured["observations"] is None


def test_goal_forwards_an_observation_that_was_asked_for(monkeypatch):
    """The social task has a default observation, not a fixed one."""
    captured = {}

    def fake(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(runs, "run_goal_episode", fake)
    assert cli.main(["goal", "--observations", "pose,goal"]) == 0
    assert captured["observations"] == "pose,goal"


def test_bench_forwards_the_step_count_and_the_scale(monkeypatch):
    captured = {}

    def fake(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(runs, "run_benchmark", fake)
    assert cli.main(["bench", "--steps", "250", "--time-scale", "5"]) == 0
    assert captured["steps"] == 250
    assert captured["time_scale"] == 5.0
    assert captured["scenario_id"] == "python_bench_demo"


def test_training_takes_lockstep_and_inference_runs_free(monkeypatch):
    """Training asks for lockstep; playing a policy back keeps the world running.

    A training run is the one whose control period has to mean something - a
    policy that takes its time must cost wall time rather than change the task -
    so it takes one control period at a time whatever the policy costs. That is
    also what stops the world while the policy answers, which is what a watcher
    sees as lag at a single times speed: playing a checkpoint back is how a run
    is watched, so it keeps the historical free pacing and the smooth picture,
    and ``--pacing lockstep`` is there for a caller who wants the exact step.
    """
    captured = {}

    def fake(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(runs, "run_training", fake)
    monkeypatch.setattr(runs, "run_policy_episodes", fake)
    monkeypatch.setattr(runs, "run_random_episode", fake)
    # Which reader a checkpoint needs is read from the file itself, and this
    # test is about the pacing flags: give it an answer instead of a file that
    # has to exist somewhere in the working directory.
    monkeypatch.setattr(runs, "_checkpoint_kind", lambda path: "torch")

    assert cli.main(["train", "--episodes", "1"]) == 0
    assert captured["pacing"] == "lockstep"
    assert cli.main(["play", "--load", "policy.pt"]) == 0
    assert captured["pacing"] == "free"
    assert cli.main(["episode"]) == 0
    assert captured["pacing"] == "free"
    assert cli.main(["train", "--episodes", "1", "--pacing", "free"]) == 0
    assert captured["pacing"] == "free"


def test_a_method_and_a_learning_rule_are_two_axes(monkeypatch):
    """``--method`` names a method, ``--algo`` names a learning rule, and each run says which.

    CADRL and SARL are not the same kind of thing as PPO: they bring their own
    value network, their own observation, their own discrete command set and
    their own reward, which is why each one also brings the environment that
    defines them. ``--method`` is where that lives now, and spelling one in
    ``--algo`` still means the method so a command line written before the flag
    existed keeps working.
    """
    called = {}

    def record(name):
        def fake(**kwargs):
            called.clear()
            called.update(kwargs, _run=name)
            return 0
        return fake

    monkeypatch.setattr(runs, "run_training", record("reinforce"))
    monkeypatch.setattr(runs, "run_sb3_training", record("sb3"))
    monkeypatch.setattr(runs, "run_social_training", record("social"))

    assert cli.main(["train", "--episodes", "1"]) == 0
    assert called["_run"] == "reinforce", "no flag keeps the historical default"

    assert cli.main(["train", "--episodes", "1", "--algo", "ppo"]) == 0
    assert called["_run"] == "sb3" and called["algo"] == "ppo"

    assert cli.main(["train", "--episodes", "1", "--method", "cadrl"]) == 0
    assert called["_run"] == "social" and called["algo"] == "cadrl"

    assert cli.main(["train", "--episodes", "1", "--algo", "sarl"]) == 0
    assert called["_run"] == "social", "--algo still spells the method it always did"
    assert called["algo"] == "sarl"


def test_a_method_refuses_to_be_paired_with_a_learning_rule(monkeypatch):
    """Asking for two different things is refused by name, not resolved silently."""
    monkeypatch.setattr(runs, "run_social_training", lambda **kwargs: 0)
    monkeypatch.setattr(runs, "run_sb3_training", lambda **kwargs: 0)

    with pytest.raises(SystemExit) as refused:
        cli.main(["train", "--episodes", "1", "--method", "cadrl", "--algo", "ppo"])
    message = str(refused.value)
    assert "--method cadrl" in message and "--algo ppo" in message

    with pytest.raises(SystemExit) as two_methods:
        cli.main(["train", "--episodes", "1", "--method", "cadrl", "--algo", "sarl"])
    assert "two different methods" in str(two_methods.value)


def test_play_takes_a_method(monkeypatch):
    """Inference names its method the same way training does."""
    captured = {}

    def fake(**kwargs):
        captured.clear()
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(runs, "run_social_policy", fake)

    assert cli.main(["play", "--load", "policy.pt", "--method", "sarl"]) == 0
    assert captured["algo"] == "sarl"
    assert captured["load"] == "policy.pt"


def test_the_environment_options_default_to_the_constructor_defaults():
    """The options added last must not move a value the constructor already set."""
    args = cli.build_parser().parse_args(["episode"])
    assert args.robot is None
    assert args.max_humans == 8
    assert args.wait_timeout == 30.0
    assert args.goal_reward == 10.0
    assert args.progress_reward == 1.0
    assert args.collision_penalty == 10.0
    assert args.out_of_bounds_penalty == 5.0
    assert args.time_penalty == 0.1


def test_episode_forwards_the_environment_constructor_options(monkeypatch):
    captured = {}

    def fake(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(runs, "run_random_episode", fake)
    assert (
        cli.main(
            [
                "episode",
                "--robot",
                "robot_2",
                "--max-humans",
                "3",
                "--wait-timeout",
                "2.5",
                "--goal-reward",
                "7.0",
                "--progress-reward",
                "2.0",
                "--collision-penalty",
                "4.0",
                "--out-of-bounds-penalty",
                "6.0",
                "--time-penalty",
                "0.25",
            ]
        )
        == 0
    )
    assert captured["robot"] == "robot_2"
    assert captured["max_humans"] == 3
    assert captured["wait_timeout"] == 2.5
    assert captured["goal_reward"] == 7.0
    assert captured["progress_reward"] == 2.0
    assert captured["collision_penalty"] == 4.0
    assert captured["out_of_bounds_penalty"] == 6.0
    assert captured["time_penalty"] == 0.25


def test_goal_forwards_the_robot_it_was_given(monkeypatch):
    """The social task hands the fleet id to the constructor it inherits."""
    captured = {}

    def fake(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(runs, "run_goal_episode", fake)
    assert cli.main(["goal", "--robot", "robot_2"]) == 0
    assert captured["robot"] == "robot_2"


def test_a_subcommand_returns_the_exit_code_of_its_run(monkeypatch):
    monkeypatch.setattr(runs, "run_scenario", lambda **kwargs: 7)
    assert cli.main(["scenario"]) == 7


def test_viewer_delegates_to_the_viewer_entry_point(monkeypatch):
    import robotsnap.viewer as viewer

    seen = {}

    def fake_main(argv):
        seen["argv"] = argv
        return 3

    monkeypatch.setattr(viewer, "main", fake_main)
    assert cli.main(["viewer", "--port", "0", "--rate", "25"]) == 3
    assert seen["argv"] == ["--host", "0.0.0.0", "--port", "0", "--rate", "25.0"]


def test_watch_is_the_command_and_viewer_its_alias(capsys):
    """The command was renamed; both spellings show the same canonical help."""
    for spelling in ("watch", "viewer"):
        with pytest.raises(SystemExit) as exit_info:
            cli.main([spelling, "--help"])
        assert exit_info.value.code == 0
        assert "python -m robotsnap watch" in capsys.readouterr().out


@pytest.mark.parametrize(
    "command", ("episode", "goal", "train", "play", "bench", "scenario")
)
def test_viewer_is_the_render_option(command):
    """``--viewer`` and ``--render`` are the same flag on the commands that draw."""
    extra = ["--load", "policy.pt"] if command == "play" else []
    parser = cli.build_parser()
    assert parser.parse_args([command, "--viewer", *extra]).render is True
    assert parser.parse_args([command, "--render", *extra]).render is True
    assert parser.parse_args([command, *extra]).render is False


@pytest.mark.parametrize("command", ("bench", "scenario"))
def test_the_window_is_asked_for_all_the_way_down(command, monkeypatch):
    """The flag is not enough: the run behind it has to be told to draw."""
    seen = {}

    def fake(**kwargs):
        seen.update(kwargs)
        return 0

    monkeypatch.setattr(runs, f"run_{'benchmark' if command == 'bench' else command}", fake)

    assert cli.main([command, "--viewer"]) == 0
    assert seen["render"] is True


def test_bridge_forwards_its_options_including_the_viewer(monkeypatch):
    """``--viewer`` reaches the bridge runner, beside the transport options."""
    from robotsnap.bridge import __main__ as bridge_runner

    seen = {}

    def fake_main(argv):
        seen["argv"] = argv
        return 5

    monkeypatch.setattr(bridge_runner, "main", fake_main)
    assert cli.main(["bridge", "--ros2", "--viewer", "--rate", "7"]) == 5
    assert seen["argv"] == [
        "--host",
        "0.0.0.0",
        "--port",
        "10000",
        "--rate",
        "7.0",
        "--ros2",
        "--ros2-node-name",
        "robotsnap_gateway",
        "--viewer",
    ]


# -- the two short runs, against the fake peer ------------------------------


def test_run_random_episode_against_the_fake_peer(client, unity, bridge, project, capsys):
    _start(unity, bridge)
    code = runs.run_random_episode(
        scenario_id="demo",
        scenario_fields={},
        client=client,
        port=bridge.port,
        host="127.0.0.1",
        control_period=_CONTROL_PERIOD,
        seconds=_EPISODE_SECONDS,
        seed=0,
        unity_project=str(project),
    )
    out = capsys.readouterr().out
    assert code == 0
    assert f"bridge on 127.0.0.1:{bridge.port}" in out
    assert "episode started  : scenario demo" in out
    assert "episode ended" in out


def test_run_goal_episode_against_the_fake_peer(client, unity, bridge, project, capsys):
    _start(unity, bridge)
    code = runs.run_goal_episode(
        episodes=1,
        scenario_id="demo",
        scenario_fields={},
        client=client,
        port=bridge.port,
        host="127.0.0.1",
        control_period=_CONTROL_PERIOD,
        seconds=_EPISODE_SECONDS,
        unity_project=str(project),
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "episode 1:" in out
    assert "/1 episodes reached the goal" in out


def test_a_run_writes_its_scenario_where_it_was_pointed_at(
    client, unity, bridge, project, capsys
):
    """The project of ``--unity-project`` is where the file goes, not the default one.

    The environment used to resolve the scenarios directory by itself, so a run
    pointed at another project wrote into the checkout beside the home: the
    tests of this file were creating files in the real Unity project of the
    machine they ran on.
    """
    _start(unity, bridge)

    code = runs.run_random_episode(
        scenario_id="demo",
        scenario_fields=runs.random_episode_fields(),
        client=client,
        port=bridge.port,
        host="127.0.0.1",
        control_period=_CONTROL_PERIOD,
        seconds=_EPISODE_SECONDS,
        seed=0,
        unity_project=str(project),
        keep=True,
    )

    assert code == 0
    written = project / "Assets" / "StreamingAssets" / "Scenarios" / "demo.yaml"
    assert written.is_file()


def test_run_random_episode_with_named_observation_parts(
    client, unity, bridge, project, capsys
):
    """``observations="pose,goal"`` reaches the constructor and builds a vector."""
    _start(unity, bridge)
    code = runs.run_random_episode(
        scenario_id="demo",
        scenario_fields={},
        client=client,
        port=bridge.port,
        host="127.0.0.1",
        control_period=_CONTROL_PERIOD,
        seconds=_EPISODE_SECONDS,
        seed=0,
        unity_project=str(project),
        observations="pose,goal",
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "observation      : (" in out
    assert "float32" in out


def test_run_random_episode_with_a_dict_observation(
    client, unity, bridge, project, capsys
):
    """A structured observation prints its parts instead of a flat shape."""
    _start(unity, bridge)
    code = runs.run_random_episode(
        scenario_id="demo",
        scenario_fields={},
        client=client,
        port=bridge.port,
        host="127.0.0.1",
        control_period=_CONTROL_PERIOD,
        seconds=_EPISODE_SECONDS,
        seed=0,
        unity_project=str(project),
        observations="pose,goal",
        observation_structure="dict",
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "observation      : dict(" in out


def test_run_benchmark_against_the_fake_peer(client, unity, bridge, project, capsys):
    """The timed run takes its steps and prints the simulated-to-wall ratio."""
    _start(unity, bridge)
    code = runs.run_benchmark(
        steps=3,
        scenario_id="demo",
        scenario_fields={},
        client=client,
        port=bridge.port,
        host="127.0.0.1",
        control_period=_CONTROL_PERIOD,
        unity_project=str(project),
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "steps            : 3" in out
    assert "sim/wall         :" in out


def test_the_environment_options_of_the_command_line_reach_the_constructor(
    client, unity, bridge, project, monkeypatch
):
    """``cli.main`` hands ``--robot``/``--goal-reward``/``--max-humans`` to the environment.

    The command normally owns its bridge; here the environment's client is
    stubbed to the fixture session so the run can be observed, the same session
    the direct runs of this file are given.
    """
    from robotsnap import envs

    built = []

    class Recording(envs.RobotSNAPEnv):
        def __init__(self, **kwargs):
            built.append(self)
            super().__init__(**kwargs)

    monkeypatch.setattr(envs, "RobotSNAPEnv", Recording)
    monkeypatch.setattr(envs.base, "RobotSNAPClient", lambda **kwargs: client)
    fleet = [
        {
            "id": "robot_1",
            "type": "jackal",
            "is_primary": True,
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "yaw": 0.0,
            "has_goal": False,
            "goal": None,
        },
        {
            "id": "robot_2",
            "type": "kuri",
            "is_primary": False,
            "x": 3.0,
            "y": 4.0,
            "z": 0.0,
            "yaw": 0.0,
            "has_goal": True,
            "goal": {"x": 3.0, "y": 9.0, "z": 0.0},
        },
    ]
    _start(unity, bridge, robots=fleet)
    code = cli.main(
        [
            "episode",
            "--host",
            "127.0.0.1",
            "--port",
            str(bridge.port),
            "--scenario",
            "demo",
            "--unity-project",
            str(project),
            "--control-period",
            str(_CONTROL_PERIOD),
            "--seconds",
            str(_EPISODE_SECONDS),
            "--robot",
            "robot_2",
            "--goal-reward",
            "3.5",
            "--max-humans",
            "2",
        ]
    )
    assert code == 0
    assert len(built) == 1
    environment = built[0]
    assert environment.robot == "robot_2"
    assert environment.goal_reward == 3.5
    assert environment.max_humans == 2


def test_the_goal_run_hands_the_robot_to_the_social_environment(
    client, unity, bridge, project, monkeypatch
):
    """``SocialNavEnv`` inherits the constructor, so ``--robot`` reaches it too."""
    from robotsnap import envs

    built = []

    class Recording(envs.RobotSNAPEnv):
        def __init__(self, **kwargs):
            built.append(self)
            super().__init__(**kwargs)

    monkeypatch.setattr(envs, "RobotSNAPEnv", Recording)
    monkeypatch.setattr(envs.base, "RobotSNAPClient", lambda **kwargs: client)
    fleet = [
        {
            "id": "robot_1",
            "type": "jackal",
            "is_primary": True,
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "yaw": 0.0,
            "has_goal": False,
            "goal": None,
        },
        {
            "id": "robot_2",
            "type": "kuri",
            "is_primary": False,
            "x": 3.0,
            "y": 4.0,
            "z": 0.0,
            "yaw": 0.0,
            "has_goal": True,
            "goal": {"x": 3.0, "y": 9.0, "z": 0.0},
        },
    ]
    _start(unity, bridge, robots=fleet)
    code = cli.main(
        [
            "goal",
            "--host",
            "127.0.0.1",
            "--port",
            str(bridge.port),
            "--scenario",
            "demo",
            "--unity-project",
            str(project),
            "--control-period",
            str(_CONTROL_PERIOD),
            "--seconds",
            str(_EPISODE_SECONDS),
            "--robot",
            "robot_2",
        ]
    )
    assert code == 0
    assert len(built) == 1
    assert built[0].robot == "robot_2"


def test_a_missing_unity_project_is_reported(monkeypatch, capsys):
    def missing(*args, **kwargs):
        raise FileNotFoundError("no Unity project found")

    monkeypatch.setattr(runs.scenario, "scenarios_dir", missing)
    code = runs.run_random_episode(scenario_id="demo", scenario_fields={})
    assert code == 2
    assert "cannot find the Unity project" in capsys.readouterr().err


# -- playing a saved policy --------------------------------------------------


def test_play_help_lists_its_options(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["play", "--help"])
    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    assert "usage:" in out
    assert "python -m robotsnap play" in out
    for option in ("--load", "--episodes", "--sample", "--stop", "--no-stop"):
        assert option in out


def test_play_without_a_checkpoint_is_an_error(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["play"])
    assert exit_info.value.code == 2
    assert "--load" in capsys.readouterr().err


def test_play_forwards_the_checkpoint_the_episodes_and_the_flags(monkeypatch):
    captured = {}

    def fake(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(runs, "run_policy_episodes", fake)
    monkeypatch.setattr(runs, "_checkpoint_kind", lambda path: "torch")
    code = cli.main(
        ["play", "--load", "policy.pt", "--episodes", "3", "--sample", "--no-stop"]
    )
    assert code == 0
    assert captured["load"] == "policy.pt"
    assert captured["episodes"] == 3
    assert captured["sample"] is True
    assert captured["stop"] is False
    assert captured["scenario_id"] == "python_policy_demo"


def test_play_aliases_to_infer(monkeypatch):
    captured = {}

    def fake(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(runs, "run_policy_episodes", fake)
    monkeypatch.setattr(runs, "_checkpoint_kind", lambda path: "torch")
    assert cli.main(["infer", "--load", "policy.pt"]) == 0
    assert captured["load"] == "policy.pt"


def test_train_forwards_the_checkpoint_to_resume_from(monkeypatch):
    captured = {}

    def fake(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(runs, "run_training", fake)
    assert cli.main(["train", "--load", "policy.pt"]) == 0
    assert captured["load"] == "policy.pt"


def test_stop_stays_on_for_a_run_that_leaves_the_scene_at_rest():
    parser = cli.build_parser()
    assert parser.parse_args(["train"]).stop is True
    assert parser.parse_args(["play", "--load", "p.pt"]).stop is True


def test_stop_stays_off_for_the_short_probes():
    parser = cli.build_parser()
    for argv in (["episode"], ["bench"], ["goal"]):
        assert parser.parse_args(argv).stop is False


def test_no_stop_and_stop_turn_the_default_of_a_command_around():
    parser = cli.build_parser()
    assert parser.parse_args(["train", "--no-stop"]).stop is False
    assert parser.parse_args(["episode", "--stop"]).stop is True
    assert parser.parse_args(["play", "--load", "p.pt", "--no-stop"]).stop is False


def test_the_stop_flag_stops_the_simulation_on_the_peer(
    client, unity, bridge, project, capsys
):
    """A run asked to stop sends ``stop_simulation`` before it closes."""
    _start(unity, bridge)
    code = runs.run_random_episode(
        scenario_id="demo",
        scenario_fields={},
        client=client,
        port=bridge.port,
        host="127.0.0.1",
        control_period=_CONTROL_PERIOD,
        seconds=_EPISODE_SECONDS,
        seed=0,
        unity_project=str(project),
        stop=True,
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "simulation stopped : True" in out
    stop_bodies = [
        body for destination, body in unity.commands
        if destination.strip("/") == "simulation/control"
        and body.get("command") == "stop_simulation"
    ]
    assert stop_bodies == [{"command": "stop_simulation"}]
    control = [
        (destination, body.get("command"))
        for destination, body in unity.commands
        if destination.strip("/") == "simulation/control"
    ]
    assert ("/simulation/control", "stop_simulation") in control


def test_the_default_probe_does_not_pause_the_session(
    client, unity, bridge, project, capsys
):
    _start(unity, bridge)
    code = runs.run_goal_episode(
        episodes=1,
        scenario_id="demo",
        scenario_fields={},
        client=client,
        port=bridge.port,
        host="127.0.0.1",
        control_period=_CONTROL_PERIOD,
        seconds=_EPISODE_SECONDS,
        unity_project=str(project),
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "simulation stopped" not in out
    commands = [body.get("command") for _, body in unity.commands]
    assert "stop_simulation" not in commands
    assert "pause" not in commands


def test_a_checkpoint_that_cannot_be_loaded_leaves_the_session_alone(
    client, unity, bridge, project, monkeypatch, capsys
):
    """A run that never started has no business freezing the scene."""
    _start(unity, bridge)
    # ``run_policy_episodes`` imports torch before it reads the checkpoint; the
    # load fails first, so an empty module is enough to reach the branch.
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))

    def refuse(*args, **kwargs):
        raise ValueError("no such checkpoint")

    monkeypatch.setattr(runs.policy, "load_policy", refuse)
    code = runs.run_policy_episodes(
        "policy.pt",
        scenario_id="demo",
        scenario_fields={},
        client=client,
        port=bridge.port,
        host="127.0.0.1",
        control_period=_CONTROL_PERIOD,
        seconds=_EPISODE_SECONDS,
        unity_project=str(project),
        stop=True,
    )
    out = capsys.readouterr()

    assert code == 2
    assert "cannot play policy.pt" in out.err
    assert "simulation stopped" not in out.out
    assert "pause" not in [body.get("command") for _, body in unity.commands]
    assert "stop_simulation" not in [
        body.get("command") for _, body in unity.commands
    ]
