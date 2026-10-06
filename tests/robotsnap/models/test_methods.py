"""The named method environments: distinct, both social, unchanged by default.

CADRL and SARL are children of the shared social task. A refactor of that split
can quietly break three things, so each has a test here: the registry hands out
the children and nobody else, each child inherits the task while naming its own
reward constants, and the defaults a run of either method starts from are the
ones the previous single-class arrangement used.

Every environment is built with a stand-in client - ``CadrlEnv(client=object())``
- so nothing here binds a port, starts a session or needs Unity.
"""

import pytest

from robotsnap.models import ALGORITHMS
from robotsnap.models.cadrl import CADRL_REWARD_PARAMETERS, CadrlAgent, CadrlEnv
from robotsnap.models.ga3c_cadrl import Ga3cAgent, Ga3cCadrlEnv
from robotsnap.models.rgl import RglAgent, RglEnv
from robotsnap.models.sarl import SARL_REWARD_PARAMETERS, SarlAgent, SarlEnv
from robotsnap.models.social import SocialNavEnv, SocialReward

#: The reward constants the social task paid before the two children existed.
#: Written out rather than read from ``SocialReward`` so a change to the base
#: defaults cannot silently move what a method's run starts from.
HISTORICAL_REWARD = {
    "progress_weight": 1.0,
    "collision_penalty": 10.0,
    "goal_bonus": 10.0,
    "time_penalty": 0.1,
    "comfort_distance": 1.0,
    "personal_space_weight": 1.0,
    "too_close_distance": 0.5,
    "too_close_penalty": 5.0,
}


def _env(environment_class, **kwargs):
    """An environment with a stand-in client, so no port is ever bound."""
    return environment_class(client=object(), **kwargs)


def test_the_registry_hands_out_the_named_methods():
    from robotsnap.models.template import TemplateAgent, TemplateEnv

    assert set(ALGORITHMS) == {"cadrl", "sarl", "ga3c_cadrl", "rgl", "template"}
    assert ALGORITHMS["cadrl"] == (CadrlEnv, CadrlAgent)
    assert ALGORITHMS["sarl"] == (SarlEnv, SarlAgent)
    assert ALGORITHMS["ga3c_cadrl"] == (Ga3cCadrlEnv, Ga3cAgent)
    assert ALGORITHMS["rgl"] == (RglEnv, RglAgent)
    assert ALGORITHMS["template"] == (TemplateEnv, TemplateAgent)
    # The template is a model too, but a task of its own: it sits on the base
    # environment rather than on the shared social one.
    assert not issubclass(TemplateEnv, SocialNavEnv)


def test_the_two_children_are_distinct_classes_of_the_shared_task():
    assert CadrlEnv is not SarlEnv
    assert issubclass(CadrlEnv, SocialNavEnv)
    assert issubclass(SarlEnv, SocialNavEnv)
    assert not issubclass(SocialNavEnv, CadrlEnv)
    assert not issubclass(SocialNavEnv, SarlEnv)
    # Neither child rewrites the task it inherits: both use the base's setup.
    assert CadrlEnv.__init__ is SocialNavEnv.__init__
    assert SarlEnv.__init__ is SocialNavEnv.__init__


def test_each_child_names_its_own_reward_constants():
    """A method's shaping is its own attribute, not a shared module global."""
    assert CadrlEnv.reward_parameters is CADRL_REWARD_PARAMETERS
    assert SarlEnv.reward_parameters is SARL_REWARD_PARAMETERS
    assert CADRL_REWARD_PARAMETERS is not SARL_REWARD_PARAMETERS
    assert dict(CADRL_REWARD_PARAMETERS) == HISTORICAL_REWARD
    assert dict(SARL_REWARD_PARAMETERS) == HISTORICAL_REWARD


def test_a_child_can_shape_differently_without_touching_the_base():
    """The point of the hook: override on one child, base and sibling stand."""

    class Harsher(CadrlEnv):
        reward_parameters = dict(CADRL_REWARD_PARAMETERS, collision_penalty=99.0)

    assert _env(Harsher).social_reward.collision_penalty == 99.0
    assert _env(CadrlEnv).social_reward.collision_penalty == 10.0
    assert _env(SarlEnv).social_reward.collision_penalty == 10.0
    assert _env(SocialNavEnv).social_reward.collision_penalty == 10.0


def test_a_ready_made_reward_is_taken_as_it_is():
    reward = SocialReward(goal_bonus=42.0)

    assert _env(SarlEnv, social_parameters=reward).social_reward is reward


def test_a_callers_constants_layer_over_the_childs_own():
    env = _env(SarlEnv, social_parameters={"goal_bonus": 25.0})

    assert env.social_reward.goal_bonus == 25.0
    assert env.social_reward.collision_penalty == HISTORICAL_REWARD["collision_penalty"]


def test_an_unknown_constant_is_still_refused():
    with pytest.raises(ValueError, match="unknown social parameter"):
        _env(CadrlEnv, social_parameters={"not_a_weight": 1.0})


def test_the_defaults_are_the_ones_the_base_used_to_hand_out():
    """Same eighty actions, same neighbour room, same reward as before."""
    base = _env(SocialNavEnv)

    for environment_class in (CadrlEnv, SarlEnv):
        env = _env(environment_class)
        assert env.action_space.n == base.action_space.n == 80
        assert env.max_neighbours == base.max_neighbours == 8
        assert env.social_reward == base.social_reward
        assert env.social_reward.parameters() == HISTORICAL_REWARD
        assert env.observation_space == base.observation_space
        assert len(env.social_actions) == len(base.social_actions)
