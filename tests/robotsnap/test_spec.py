"""What a run says it is.

The command line has one ``--algo`` knob whose values do not all name the same
kind of thing, so the line a run prints is what keeps the difference visible:
one field per axis, read from the environment that was built rather than from a
table beside it. These tests pin the reading and the wording, on plain objects,
because the interesting part is the introspection and not a live session.
"""

from types import SimpleNamespace

import gymnasium

from robotsnap.runs import spec
from robotsnap.runs.spec import describe_run, resolve_run_spec


def _base(**overrides):
    """An environment shaped like the plain one: named parts and a twist box."""
    fields = dict(
        observation_names=("pose", "goal", "agents", "lidar"),
        observation_structure="flat",
        action_space=gymnasium.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype="float32"
        ),
        goal_reward=10.0,
        progress_reward=1.0,
        collision_penalty=10.0,
        out_of_bounds_penalty=5.0,
        time_penalty=0.1,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _social(**overrides):
    """An environment shaped like the social one: its own dict and its own table."""
    fields = dict(
        observation_names=(),
        observation_structure="flat",
        observation_space=gymnasium.spaces.Dict(
            {
                "ego": gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype="float32"),
                "goal": gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype="float32"),
            }
        ),
        # The real size of the CADRL set: five speeds crossed with sixteen
        # heading offsets. Keeping the true number here is what lets the
        # expectation below agree with the live line instead of with a
        # placeholder nobody would notice had drifted.
        action_space=gymnasium.spaces.Discrete(5 * 16),
        social_actions=object(),
        max_neighbours=8,
        social_reward=object(),
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_a_learning_rule_leaves_the_task_alone():
    line = describe_run(_base(), algorithm="ppo")

    assert line.startswith("run: algorithm=ppo")
    assert "observations=pose,goal,agents,lidar" in line
    assert "actions=Box(2)" in line
    assert "method=" not in line, "a rule is not a method and says so by saying nothing"


def test_a_method_is_named_beside_its_learner():
    line = describe_run(_social(), algorithm="dqn", method="cadrl")

    assert line.startswith("run: method=cadrl | algorithm=dqn")
    assert "observations=social(dict, 8 neighbours)" in line
    assert "actions=Discrete(80)" in line
    assert "reward=social" in line


def test_the_action_count_is_read_from_the_space_and_not_assumed():
    """A size the space was not built with cannot be printed by accident."""
    line = describe_run(_social(action_space=gymnasium.spaces.Discrete(37)), algorithm="dqn", method="cadrl")

    assert "actions=Discrete(37)" in line


def test_the_fixture_is_shaped_like_the_real_social_environment():
    """The plain object above is only worth something if it matches the real one.

    The stand-in client means no port is ever bound; the spaces are built in
    the constructor, so this reads the real action set, the real neighbour
    room and the real reward label without a session.
    """
    from robotsnap.models.social import SocialNavEnv

    real = resolve_run_spec(
        SocialNavEnv(client=object()), algorithm="dqn", method="cadrl"
    )
    shaped = resolve_run_spec(_social(), algorithm="dqn", method="cadrl")

    assert shaped.actions == real.actions
    assert shaped.observations == real.observations
    assert shaped.reward == real.reward


def test_the_reward_weights_are_read_off_the_environment():
    line = describe_run(_base(goal_reward=2.5, time_penalty=0.25), algorithm="reinforce")

    assert "reward=goal 2.5 progress 1 collision 10 oob 5 time 0.25" in line


def test_a_structured_observation_says_so():
    line = describe_run(_base(observation_structure="dict"), algorithm="ppo")

    assert "observations=pose,goal,agents,lidar (dict)" in line


def test_an_environment_with_no_parts_says_none():
    line = describe_run(_base(observation_names=()), algorithm="ppo")

    assert "observations=none" in line


def test_the_environment_is_named_by_its_class():
    spec_line = resolve_run_spec(_base(), algorithm="ppo")

    assert spec_line.environment == "SimpleNamespace"
    assert spec_line.method == ""


def test_the_method_names_are_the_registrys_names():
    """What a run narrates is read off the environment, not off a table here."""
    from robotsnap.models import ALGORITHMS, METHODS

    assert set(METHODS) == set(ALGORITHMS)


def test_a_task_of_the_catalogue_is_named_honestly():
    """The template is neither the plain environment nor the social one.

    It publishes a flat observation of its own with no named parts, three
    commands, and two rewards of its own - and the line a run prints has to say
    that rather than borrow the wording of the task it does not share.
    """
    from robotsnap.models.template import TemplateEnv

    named = resolve_run_spec(TemplateEnv(client=object()), algorithm="dqn", method="template")

    assert named.environment == "TemplateEnv"
    assert named.method == "template"
    assert named.observations == "flat(5)"
    assert named.actions == "Discrete(3)"
    assert named.reward == "goal 10 progress 1 collision 0 oob 0 time 0"
