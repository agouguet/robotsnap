"""Tests for reading a configuration file into a mapping."""

import json

import pytest

from robotsnap import config


def test_a_yaml_file_reads_back_as_a_mapping():
    document = config.load("ppo")
    assert document["algo"] == "ppo"
    assert document["gamma"] == pytest.approx(0.99)
    assert isinstance(document["algo_kwargs"], dict)


def test_a_json_file_reads_back_as_a_mapping(tmp_path):
    path = tmp_path / "mine.json"
    path.write_text(json.dumps({"algo": "dqn", "gamma": 0.5}), encoding="utf-8")

    document = config.load(path)
    assert document == {"algo": "dqn", "gamma": 0.5}


def test_an_empty_file_is_refused(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    with pytest.raises(config.ConfigError) as refusal:
        config.load(path)
    assert "empty" in str(refusal.value)


def test_a_root_that_is_not_an_object_is_refused(tmp_path):
    path = tmp_path / "list.yaml"
    path.write_text("- one\n- two\n", encoding="utf-8")
    with pytest.raises(config.ConfigError) as refusal:
        config.load(path)
    assert "mapping" in str(refusal.value)


def test_a_bad_json_file_says_so(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{\n", encoding="utf-8")
    with pytest.raises(config.ConfigError) as refusal:
        config.load(path)
    assert "JSON" in str(refusal.value)
