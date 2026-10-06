"""The worked example: it builds, it steps, and it learns.

The template is what a reader copies, so what is worth pinning is that the copy
would work: the environment builds without a session, one step of the task -
observation, action, command, reward - runs on plain dataclasses with no socket
at all, and the agent the catalogue hands out for it really takes a gradient
step and really replays itself from a checkpoint.
"""

import numpy as np
import pytest

from robotsnap.envs.base import TaskState
from robotsnap.envs.world import Pose2D, World
from robotsnap.models import ALGORITHMS
from robotsnap.models.template import (
    TEMPLATE_COMMANDS,
    TEMPLATE_OBSERVATION_SIZE,
    TemplateAgent,
    TemplateEnv,
)

_WAIT_TIMEOUT = 2.0
_CONTROL_PERIOD = 0.05


def _wait_for(predicate, timeout=_WAIT_TIMEOUT):
    import time

    deadline = time.monotonic() + timeout
    while True:
        value = predicate()
        if value:
            return value
        if time.monotonic() >= deadline:
            return value
        time.sleep(0.01)


def _publish_world(unity, *, x=0.0, sim_time=10.0, scenario_id="demo"):
    """Publish the whole world one environment read needs, in one go."""
    entry = {
        "id": "robot_1",
        "type": "jackal",
        "is_primary": True,
        "x": x,
        "y": 0.0,
        "z": 0.0,
        "yaw": 0.0,
        "has_goal": True,
        "goal": {"x": 5.0, "y": 0.0, "z": 0.0},
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
            "scenario_id": scenario_id,
            "scenario_name": scenario_id,
            "sim_time_seconds": sim_time,
            "time_scale": 1.0,
            "map_width": 12,
            "map_height": 12,
            "map_resolution": 1.0,
            "map_origin_x": 0.0,
            "map_origin_y": 0.0,
            "human_count": 0,
            "humans": [],
            "robots": [entry],
        }
    )
    unity.publish_odom(x=x, y=0.0, yaw=0.0, linear_x=0.0, angular_z=0.0, stamp=sim_time)
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


def _start(unity, bridge):
    """Bring the fake session up: reader thread, closed loop, one world."""
    _wait_for(
        lambda: bridge.topic_types().get("simulation/control_result")
        == "std_msgs/msg/String"
    )
    unity.start_responder()
    _publish_world(unity)
    _wait_for(lambda: bridge.topic_counts().get("simulation/state", 0) > 0)


def _env(**kwargs) -> TemplateEnv:
    """A template environment with a stand-in client, so no port is bound."""
    return TemplateEnv(client=object(), **kwargs)


def _task(
    distance=5.0,
    bearing=0.0,
    *,
    goal_reached=False,
    elapsed=0.0,
) -> TaskState:
    return TaskState(
        distance_to_goal=distance,
        bearing_to_goal=bearing,
        goal_reached=goal_reached,
        collision=False,
        out_of_bounds=False,
        min_lidar=None,
        nearest_agent_distance=None,
        elapsed_seconds=elapsed,
        wall_seconds=elapsed,
        steps=0,
    )


def _world(linear=0.0) -> World:
    return World(pose=Pose2D(0.0, 0.0, 0.0), linear_velocity=linear)


def test_the_registry_hands_out_the_template_like_any_other_method():
    assert ALGORITHMS["template"] == (TemplateEnv, TemplateAgent)
    assert TemplateAgent.name == "template"


def test_the_template_builds_a_reduced_observation_and_a_tiny_action_set():
    env = _env()

    assert env.observation_space.shape == (TEMPLATE_OBSERVATION_SIZE,)
    assert env.action_space.n == len(TEMPLATE_COMMANDS) == 3
    assert env.observation_space.contains(env.observation(_world(linear=0.4), _task()))


def test_the_reduced_observation_carries_the_goal_error_and_the_speed():
    env = _env()
    observation = env.observation(_world(linear=0.4), _task(distance=4.0, bearing=0.0))

    assert observation == pytest.approx([4.0, 0.0, 4.0, 0.0, 0.4])
    # A session that has published no goal yet is a zero vector, not an error.
    assert env.observation(_world(), _task(distance=None, bearing=None)) == pytest.approx(
        np.zeros(TEMPLATE_OBSERVATION_SIZE)
    )


def test_the_action_index_becomes_a_clipped_command():
    env = _env()

    assert env.action_to_command(0) == (0.0, 0.0)
    assert env.action_to_command(1) == (0.5, 0.0)
    assert env.action_to_command(2) == (0.0, 0.5)
    with pytest.raises(ValueError, match="outside the"):
        env.action_to_command(len(TEMPLATE_COMMANDS))


def test_the_reward_is_the_two_terms_the_template_names():
    env = _env()
    previous, current = _task(distance=5.0), _task(distance=4.5, elapsed=0.2)

    # Half a metre of progress, and nothing else: the template prices neither
    # the time a step took nor the crash, so what is left is the progress term.
    assert env.reward(previous, current, (0.5, 0.0), _world()) == pytest.approx(0.5)
    arrived = _task(distance=4.5, goal_reached=True)
    assert env.reward(previous, arrived, (0.5, 0.0), _world()) == pytest.approx(
        0.5 + env.goal_reward
    )


def test_the_template_takes_what_the_shared_method_runner_hands_it():
    """``run_social_training`` builds every method from one set of options.

    A method that does not take one of them has to accept it anyway, and the
    template pays two terms whatever the runner passes - the line a run prints
    is what would otherwise name a reward this task does not have.
    """
    from robotsnap.runs.spec import resolve_run_spec

    env = _env(
        max_neighbours=8,
        social_parameters=None,
        control_period=0.2,
        collision_penalty=10.0,
        out_of_bounds_penalty=5.0,
        time_penalty=0.1,
    )

    assert env.collision_penalty == 0.0
    assert env.out_of_bounds_penalty == 0.0
    assert env.time_penalty == 0.0
    # One metre of progress over a step that took a whole control period.
    assert env.reward(
        _task(distance=4.0), _task(distance=3.0, elapsed=0.2), (0.5, 0.0), _world()
    ) == pytest.approx(1.0)

    named = resolve_run_spec(env, algorithm="dqn", method="template")
    assert named.reward == "goal 10 progress 1 collision 0 oob 0 time 0"


def test_one_step_of_the_template_task_runs_without_a_socket():
    """The four parts a step touches, in order, on plain dataclasses."""
    env = _env()
    world = _world(linear=0.2)
    previous, current = _task(distance=5.0), _task(distance=4.6, bearing=0.05, elapsed=0.2)

    observation = env.observation(world, previous)
    action = 1
    command = env.action_to_command(action)
    reward = env.reward(previous, current, command, world)
    following = env.observation(world, current)

    assert env.observation_space.contains(observation)
    assert env.observation_space.contains(following)
    assert env.action_space.contains(action)
    assert command == (0.5, 0.0)
    assert reward == pytest.approx(0.4)


def test_the_template_agent_learns_from_the_steps_of_its_own_task():
    pytest.importorskip("torch")
    from robotsnap.models import make_agent

    env = _env()
    agent = make_agent("template", env, batch_size=2, hidden=8, seed=0)

    assert agent.ego_features == TEMPLATE_OBSERVATION_SIZE
    assert agent.actions == len(TEMPLATE_COMMANDS)

    world = _world(linear=0.2)
    observation = env.observation(world, _task(distance=5.0))
    losses = []
    for index in range(6):
        action = agent.act(observation)
        current = _task(distance=5.0 - 0.1 * (index + 1), elapsed=0.2 * (index + 1))
        reward = env.reward(_task(distance=5.0), current, env.action_to_command(action), world)
        following = env.observation(world, current)
        agent.observe(observation, action, reward, following, False)
        losses.append(agent.learn())
        observation = following

    assert any(loss is not None for loss in losses)
    assert all(np.isfinite(loss) for loss in losses if loss is not None)
    assert agent.epsilon < agent.epsilon_start


def test_a_saved_template_agent_replays_the_same_greedy_action(tmp_path):
    pytest.importorskip("torch")
    from robotsnap.models import load_agent, make_agent

    env = _env()
    agent = make_agent("template", env, batch_size=2, hidden=8, seed=3)
    world = _world(linear=0.2)
    observation = env.observation(world, _task(distance=3.0, bearing=0.2))
    for _ in range(4):
        agent.observe(observation, 1, 0.1, env.observation(world, _task(distance=2.5)), False)

    path = tmp_path / "template.pt"
    agent.save(path)
    reloaded = load_agent(path)

    assert isinstance(reloaded, TemplateAgent)
    assert reloaded.act(observation, greedy=True) == agent.act(observation, greedy=True)


def test_the_template_really_steps_a_session(client, unity, bridge):
    """The whole transport path, against the fake peer rather than a dataclass."""
    from robotsnap import topics
    from robotsnap.bridge import codec

    environment = TemplateEnv(
        client=client,
        scenario="demo",
        control_period=_CONTROL_PERIOD,
        wait_timeout=_WAIT_TIMEOUT,
    )
    try:
        _start(unity, bridge)
        observation, _ = environment.reset(seed=0)
        assert observation == pytest.approx([5.0, 0.0, 5.0, 0.0, 0.0])

        # The robot advanced one metre toward the goal while the step ran.
        _publish_world(unity, x=1.0, sim_time=10.5)
        observation, reward, terminated, truncated, info = environment.step(1)

        assert observation == pytest.approx([4.0, 0.0, 4.0, 0.0, 0.0])
        # One metre of progress and nothing else: the template prices neither
        # the time the step took nor a crash.
        assert reward == pytest.approx(1.0)
        assert info["action"] == pytest.approx((0.5, 0.0))
        assert terminated is False and truncated is False

        frames = _wait_for(
            lambda: [frame for frame in unity.inbound if frame[0].strip("/") == "cmd_vel"]
        )
        assert frames, "the template never wrote a /cmd_vel"
        twist = codec.decode(topics.TWIST_TYPE, frames[-1][1])
        assert (twist.linear.x, twist.angular.z) == pytest.approx((0.5, 0.0))
    finally:
        environment.close()
