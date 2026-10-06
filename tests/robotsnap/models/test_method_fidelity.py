"""The learner the two methods really are, not the one they used to be.

A source-level check lives here beside the behavioural ones: CADRL's Algorithm 1
is "Deep V-learning" - a state value regressed on ``r + gamma * V(s')`` with no
maximum over the actions - and the module that describes CADRL must not claim
otherwise. The tests that follow pin the public constants and names the rest of
the catalogue reads, and the shapes that make the target formula structural
rather than a promise.
"""

import inspect
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from robotsnap.models import cadrl as cadrl_module  # noqa: E402
from robotsnap.models import lookahead as lookahead_module  # noqa: E402
from robotsnap.models import rgl as rgl_module  # noqa: E402
from robotsnap.models import sarl as sarl_module  # noqa: E402
from robotsnap.models.cadrl import (  # noqa: E402
    CADRL_REWARD_PARAMETERS,
    CadrlAgent,
    CadrlEnv,
    cadrl_class,
)
from robotsnap.models.lookahead import (  # noqa: E402
    LookaheadAgent,
    state_value_base,
)
from robotsnap.models.sarl import (  # noqa: E402
    SARL_REWARD_PARAMETERS,
    SarlAgent,
    SarlEnv,
    sarl_class,
)
from robotsnap.models.social import SocialNavEnv, _DQNAgent  # noqa: E402
from robotsnap.models.template import TemplateAgent  # noqa: E402

#: The reward constants the social task paid before the two children existed.
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

_MODULES = {
    "cadrl": cadrl_module,
    "sarl": sarl_module,
    "rgl": rgl_module,
    "lookahead": lookahead_module,
}


@pytest.mark.parametrize("name", sorted(_MODULES))
def test_the_method_sources_no_longer_claim_a_dqn(name):
    source = inspect.getsource(_MODULES[name])
    assert "DQN" not in source
    assert "V(s)" in source


def test_the_docstrings_name_the_paper_the_learner_comes_from():
    cadrl_doc = inspect.getdoc(cadrl_module)
    sarl_doc = inspect.getdoc(sarl_module)
    assert "Chen" in cadrl_doc and "2017" in cadrl_doc
    assert "lookahead" in cadrl_doc.lower()
    assert "Deep V-learning" in cadrl_doc or "V-learning" in cadrl_doc
    assert "Chen" in sarl_doc and "2019" in sarl_doc
    assert "lookahead" in sarl_doc.lower()
    assert "no maximum over the actions" in sarl_doc or "no max over the actions" in sarl_doc


def test_the_method_agents_are_lookahead_agents_not_state_action_learners():
    assert issubclass(CadrlAgent, LookaheadAgent)
    assert issubclass(SarlAgent, LookaheadAgent)
    # The shared action-value learner stays the template's, untouched.
    assert issubclass(TemplateAgent, _DQNAgent)
    assert not issubclass(CadrlAgent, _DQNAgent)
    assert not issubclass(SarlAgent, _DQNAgent)


def test_the_two_learners_the_catalogue_ships_are_named_honestly():
    """A run prints the rule a method really uses, because ``--help`` reads it."""
    from robotsnap.models import METHOD_LEARNER

    assert METHOD_LEARNER["cadrl"] == "v-learning"
    assert METHOD_LEARNER["sarl"] == "v-learning"
    assert METHOD_LEARNER["rgl"] == "v-learning"
    assert METHOD_LEARNER["ga3c_cadrl"] == "a3c"
    assert METHOD_LEARNER["template"] == "dqn"


def test_the_value_classes_are_state_value_networks():
    assert issubclass(cadrl_class(), state_value_base())
    assert issubclass(sarl_class(), state_value_base())
    assert CadrlAgent(actions=3, hidden=8).value_class is cadrl_class()
    assert SarlAgent(actions=3, hidden=8).value_class is sarl_class()


def test_the_update_takes_no_maximum_over_the_actions():
    source = inspect.getsource(LookaheadAgent.update)
    assert ".max(" not in source
    assert "argmax" not in source
    assert "V_target" in source or "future" in source
    # The greedy rule, on the other hand, *is* an argmax over the lookahead.
    assert "argmax" in inspect.getsource(LookaheadAgent._lookahead_action)


def test_the_two_children_keep_their_own_reward_constants():
    assert CadrlEnv.reward_parameters is CADRL_REWARD_PARAMETERS
    assert SarlEnv.reward_parameters is SARL_REWARD_PARAMETERS
    assert CADRL_REWARD_PARAMETERS is not SARL_REWARD_PARAMETERS
    assert dict(CADRL_REWARD_PARAMETERS) == HISTORICAL_REWARD
    assert dict(SARL_REWARD_PARAMETERS) == HISTORICAL_REWARD


def test_the_lookahead_reward_reads_the_environments_own_thresholds():
    env = SocialNavEnv(client=object(), max_neighbours=2)
    env.collision_distance = 0.4
    env.goal_radius = 0.3

    def propagated(goal_distance, neighbour_distance):
        neighbours = np.zeros((1, 4), dtype=np.float64)
        mask = np.zeros(1, dtype=bool)
        if neighbour_distance is not None:
            neighbours[0] = (neighbour_distance, 0.0, 0.0, 0.0)
            mask[0] = True
        return {
            "ego": np.zeros(5, dtype=np.float64),
            "goal": np.array([goal_distance, 0.0, goal_distance, 1.0, 0.0]),
            "neighbours": neighbours,
            "mask": mask,
            "previous_distance_to_goal": goal_distance,
        }

    env.social_reward = type(env.social_reward)(
        progress_weight=0.0,
        time_penalty=0.0,
        personal_space_weight=0.0,
        too_close_distance=0.0,
        too_close_penalty=0.0,
    )
    # Inside the collision threshold the flat price is paid; just outside it is
    # not, which is exactly `collision_distance` doing the deciding.
    assert env.lookahead_reward(propagated(5.0, 0.39), 0) < 0.0
    assert env.lookahead_reward(propagated(5.0, 0.41), 0) == pytest.approx(0.0)
    # A propagated goal inside `goal_radius` takes the arrival bonus.
    assert env.lookahead_reward(propagated(0.29, None), 0) > 0.0
    assert env.lookahead_reward(propagated(0.31, None), 0) == pytest.approx(0.0)


def test_the_agent_names_and_action_space_are_unchanged():
    assert CadrlAgent.name == "cadrl"
    assert SarlAgent.name == "sarl"
    for environment_class in (CadrlEnv, SarlEnv):
        env = environment_class(client=object())
        assert env.action_space.n == 80
        assert env.max_neighbours == 8
        assert env.social_reward.parameters() == HISTORICAL_REWARD
        assert math.isclose(env.control_period, 0.1)
