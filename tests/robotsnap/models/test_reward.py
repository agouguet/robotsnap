"""Tests for ``robotsnap.models.social.SocialReward``."""

import math

import pytest

from robotsnap.models.social import SocialReward, as_social_reward


def test_a_collision_costs_more_than_a_clear_step():
    reward = SocialReward()
    clear = reward(previous_distance_to_goal=5.0, distance_to_goal=5.0)
    crash = reward(previous_distance_to_goal=5.0, distance_to_goal=5.0, collided=True)
    assert crash < clear


def test_a_goal_pays_more_than_progress_alone():
    reward = SocialReward()
    moving = reward(previous_distance_to_goal=5.0, distance_to_goal=4.0)
    arriving = reward(previous_distance_to_goal=5.0, distance_to_goal=4.0, arrived=True)
    assert arriving > moving
    assert arriving > reward.goal_bonus


def test_a_human_inside_the_personal_space_costs_more_than_one_far_away():
    reward = SocialReward()
    close = reward(previous_distance_to_goal=5.0, distance_to_goal=4.0, min_human_distance=0.2)
    far = reward(previous_distance_to_goal=5.0, distance_to_goal=4.0, min_human_distance=6.0)
    assert close < far


def test_the_progress_term_is_zero_when_the_robot_does_not_move():
    reward = SocialReward()
    assert reward(previous_distance_to_goal=3.0, distance_to_goal=3.0) == pytest.approx(0.0)


def test_backing_away_is_charged():
    reward = SocialReward()
    assert reward(previous_distance_to_goal=3.0, distance_to_goal=3.5) < 0.0


def test_a_step_costs_the_seconds_it_took():
    reward = SocialReward(time_penalty=2.0)
    assert reward(
        previous_distance_to_goal=3.0, distance_to_goal=3.0, dt=0.5
    ) == pytest.approx(-1.0)


def test_the_reward_needs_no_goal_at_all():
    reward = SocialReward()
    assert reward(min_human_distance=3.0) == pytest.approx(-reward.social_penalty(3.0))


def test_no_human_means_no_social_penalty():
    reward = SocialReward()
    assert reward.social_penalty(None) == 0.0
    assert reward.social_penalty(float("inf")) == 0.0


def test_the_too_close_band_is_priced_harder_than_the_well_alone():
    reward = SocialReward()
    assert reward.social_penalty(0.4) > reward.social_penalty(0.6)
    assert reward.social_penalty(0.4) == pytest.approx(
        reward.personal_space_weight * math.exp(-((0.4 / reward.comfort_distance) ** 2))
        + reward.too_close_penalty
    )


def test_the_well_fades_with_distance():
    reward = SocialReward(too_close_distance=0.0, too_close_penalty=0.0)
    assert reward.social_penalty(0.0) == pytest.approx(reward.personal_space_weight)
    assert reward.social_penalty(10.0) < 1e-3


def test_the_constants_are_parameters():
    strict = SocialReward(personal_space_weight=100.0)
    assert strict.social_penalty(2.0) > SocialReward().social_penalty(2.0)
    wide = SocialReward(comfort_distance=0.25)
    assert wide.social_penalty(1.0) < SocialReward().social_penalty(1.0)


def test_a_mapping_builds_a_reward_and_an_unknown_key_is_refused():
    reward = as_social_reward({"goal_bonus": 3.0})
    assert reward.goal_bonus == 3.0
    assert reward.progress_weight == SocialReward().progress_weight
    assert as_social_reward(None) == SocialReward()
    assert as_social_reward(reward) is reward
    with pytest.raises(ValueError, match="unknown social parameter"):
        as_social_reward({"goal_bonuss": 1.0})
