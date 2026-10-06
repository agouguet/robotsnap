"""Tests for ``robotsnap.models.social.SocialActionSpace``."""

import math
import random

import numpy as np
import pytest

from robotsnap.models.social import SocialActionSpace, heading_offset_to_angular


def test_the_default_grid_is_the_cadrl_set():
    space = SocialActionSpace()
    assert len(space) == 5 * 16


def test_the_zero_action_comes_first():
    space = SocialActionSpace()
    assert space[0] == (0.0, 0.0)
    assert space.index_of(0.0, 0.0) == 0


def test_every_command_round_trips_through_index_of():
    space = SocialActionSpace()
    for index in range(len(space)):
        command = space[index]
        assert space.index_of(*command) == index


def test_the_commands_are_all_distinct():
    space = SocialActionSpace()
    assert len(set(space)) == len(space)


def test_an_unknown_command_is_refused():
    space = SocialActionSpace()
    with pytest.raises(ValueError, match="not one of the"):
        space.index_of(0.123, 4.0)


def test_a_heading_offset_becomes_an_angular_velocity_over_the_period():
    assert heading_offset_to_angular(math.pi / 2, 0.5) == pytest.approx(math.pi)
    period = 0.25
    space = SocialActionSpace(control_period=period)
    half_turn = space.heading_offsets[-1]
    speed, angular = space[space.index_of(0.25, heading_offset_to_angular(half_turn, period))]
    assert speed == pytest.approx(0.25)
    assert angular == pytest.approx(half_turn / period)


def test_a_longer_control_period_asks_for_a_slower_turn():
    quick = SocialActionSpace(control_period=0.1)
    slow = SocialActionSpace(control_period=1.0)
    offset = quick.heading_offsets[-1]
    assert quick[quick.index_of(0.5, offset / 0.1)][1] > slow[slow.index_of(0.5, offset / 1.0)][1]


def test_the_heading_offsets_cover_a_full_turn():
    offsets = SocialActionSpace(headings=8).heading_offsets
    assert offsets[0] == pytest.approx(0.0)
    assert offsets[1] == pytest.approx(math.pi / 4)
    assert offsets[7] == pytest.approx(7.0 * math.pi / 4)
    assert all(0.0 <= offset < 2.0 * math.pi for offset in offsets)


def test_a_non_positive_period_is_refused():
    with pytest.raises(ValueError):
        heading_offset_to_angular(0.1, 0.0)
    with pytest.raises(ValueError):
        SocialActionSpace(control_period=-1.0)


def test_explicit_speeds_are_used_as_given():
    space = SocialActionSpace(speeds=(0.0, 0.3), headings=4)
    assert sorted({command[0] for command in space}) == [0.0, 0.3]
    assert len(space) == 2 * 4


def test_the_commands_array_lines_up_with_indexing():
    space = SocialActionSpace(speeds=3, headings=4)
    commands = space.commands
    assert commands.shape == (len(space), 2)
    for index, command in enumerate(commands):
        assert tuple(command) == pytest.approx(space[index])


def test_random_index_prefers_the_zero_action():
    space = SocialActionSpace()
    rng = np.random.default_rng(0)
    assert space.random_index(rng, near_stop_probability=1.0) == 0
    draws = {space.random_index(rng) for _ in range(200)}
    assert draws <= set(range(len(space)))
    assert len(draws) > 1


def test_a_plain_random_module_works_as_a_source():
    space = SocialActionSpace()
    assert space.random_index(random.Random(1)) in range(len(space))


def test_describe_states_the_rule_and_the_size():
    text = SocialActionSpace().describe()
    assert "80 social actions" in text
    assert "angular_z" in text
    assert "0, 0" in text
