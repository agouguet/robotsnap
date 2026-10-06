"""Tests for putting a configuration's values on a real command parser."""

import pytest

from robotsnap import config
from robotsnap.cli.main import build_parser


def _train_parser():
    return build_parser()._robotsnap_commands["train"]


def test_the_file_supplies_the_defaults():
    parser = _train_parser()
    applied = config.apply_to(parser, "train", {"episodes": 7, "pacing": "lockstep"})
    assert applied == ["episodes", "pacing"]

    namespace = parser.parse_args([])
    assert namespace.episodes == 7
    assert namespace.pacing == "lockstep"


def test_an_option_typed_on_the_line_wins():
    parser = _train_parser()
    config.apply_to(parser, "train", {"gamma": 0.99, "lr": 1.0e-5})

    from_file = parser.parse_args([])
    assert from_file.gamma == pytest.approx(0.99)
    assert from_file.lr == pytest.approx(1.0e-5)

    from_line = parser.parse_args(["--gamma", "0.5"])
    assert from_line.gamma == pytest.approx(0.5)


def test_the_top_level_parser_finds_the_command():
    parser = build_parser()
    config.apply_to(parser, "train", {"episodes": 3})
    assert parser.parse_args(["train"]).episodes == 3


def test_an_unknown_key_names_the_key_and_the_command():
    with pytest.raises(config.ConfigError) as refusal:
        config.apply_to(_train_parser(), "train", {"not_an_option": 1})
    message = str(refusal.value)
    assert "not_an_option" in message
    assert "'train'" in message


def test_a_key_of_another_command_is_refused():
    parser = build_parser()._robotsnap_commands["bench"]
    with pytest.raises(config.ConfigError) as refusal:
        config.apply_to(parser, "bench", {"timesteps": 10})
    assert "timesteps" in str(refusal.value)


def test_a_document_that_is_not_a_mapping_is_refused():
    with pytest.raises(config.ConfigError):
        config.apply_to(_train_parser(), "train", ["algo", "dqn"])
