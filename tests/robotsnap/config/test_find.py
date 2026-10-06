"""Tests for resolving a configuration name or path to the file it means."""

from pathlib import Path

import pytest

from robotsnap import config


def test_a_bare_name_resolves_in_each_tree():
    assert config.find("ppo") == config.CONFIG_DIRECTORY / "algorithms" / "ppo.yaml"
    assert config.find("cadrl") == config.CONFIG_DIRECTORY / "methods" / "cadrl.yaml"


def test_an_explicit_path_is_used_as_given(monkeypatch):
    monkeypatch.chdir(config.CONFIG_DIRECTORY.parent)
    relative = Path("configs/algorithms/ppo.yaml")
    assert config.find(relative) == (config.CONFIG_DIRECTORY / "algorithms" / "ppo.yaml")

    absolute = config.CONFIG_DIRECTORY / "methods" / "sarl.yaml"
    assert config.find(absolute) == absolute.resolve()


def test_a_bare_file_name_is_a_name_when_the_file_is_not_in_the_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert config.find("ppo.yaml") == config.CONFIG_DIRECTORY / "algorithms" / "ppo.yaml"


def test_an_unknown_name_lists_what_exists():
    with pytest.raises(config.ConfigError) as refusal:
        config.find("does-not-exist")
    message = str(refusal.value)
    assert "does-not-exist" in message
    for present in ("ppo", "dqn", "cadrl", "sarl"):
        assert present in message


def test_known_is_read_from_the_filesystem():
    names = config.known()
    assert set(names) == {"algorithms", "methods", "curriculum", "benchmarks"}
    assert "ppo" in names["algorithms"]
    assert "dqn" in names["algorithms"]
    assert "cadrl" in names["methods"]
    assert all(isinstance(value, tuple) for value in names.values())
