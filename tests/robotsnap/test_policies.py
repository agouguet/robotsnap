"""Where a training run writes the policy it learned.

The rule under test is small and worth pinning: an unnamed ``--save`` goes to
the ``policy/`` directory under a name that carries the run and the moment, a
path the user typed is honoured as typed, and ``--no-save`` writes nothing.
The last test walks the command line itself, because the file name has to say
which method or which rule wrote it, and only the command knows both.
"""

from datetime import datetime
from pathlib import Path

import pytest

from robotsnap import cli, policy_files, runs

# A fixed moment, so the generated name is a value the test can assert on.
MOMENT = datetime(2026, 10, 6, 18, 15, 0)


def test_the_default_directory_is_the_policy_folder():
    """``policy/`` ships with the repository: it is the default, not a temp path."""
    assert policy_files.DEFAULT_DIRECTORY_NAME == "policy"
    assert policy_files.default_directory() == Path("policy")


def test_an_unnamed_save_names_the_run_and_the_moment():
    """Two runs of the same command must not overwrite each other."""
    written = policy_files.resolve_save_path(
        None, label="cadrl", suffix=".pt", moment=MOMENT
    )

    assert written == Path("policy/cadrl-20261006-181500.pt")


def test_a_named_file_is_written_exactly_there(tmp_path):
    """A path is a decision, so it is not decorated with anything."""
    chosen = tmp_path / "runs" / "today.pt"

    written = policy_files.resolve_save_path(
        chosen, label="ppo", suffix=".zip", moment=MOMENT
    )

    assert written == chosen
    assert written.parent.is_dir(), "the directory the file goes into is created"


def test_a_directory_receives_a_generated_name(tmp_path):
    """Both spellings of 'a directory': the one that exists, the one with a slash."""
    existing = tmp_path / "already-there"
    existing.mkdir()

    inside_existing = policy_files.resolve_save_path(
        str(existing), label="sarl", suffix=".pt", moment=MOMENT
    )
    with_slash = policy_files.resolve_save_path(
        str(tmp_path / "not-yet") + "/", label="sarl", suffix=".pt", moment=MOMENT
    )

    assert inside_existing == existing / "sarl-20261006-181500.pt"
    assert with_slash == tmp_path / "not-yet" / "sarl-20261006-181500.pt"
    assert with_slash.parent.is_dir(), "naming a directory creates it"


def test_the_suffix_is_the_writers_own():
    """This package writes a torch document; Stable-Baselines3 writes its zip."""
    assert policy_files.suffix_for(algorithm="reinforce") == ".pt"
    assert policy_files.suffix_for(method="cadrl") == ".pt"
    assert policy_files.suffix_for(method="rgl") == ".pt"
    for rule in ("ppo", "a2c", "sac", "dqn"):
        assert policy_files.suffix_for(algorithm=rule) == ".zip"


def test_the_train_command_saves_into_the_policy_directory(monkeypatch, tmp_path):
    """The default reaches the trainer as a path; --no-save reaches it as None."""
    captured = {}

    def fake(**kwargs):
        captured.clear()
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(runs, "run_training", fake)
    monkeypatch.setattr(runs, "run_sb3_training", fake)
    monkeypatch.setattr(runs, "run_social_training", fake)
    monkeypatch.chdir(tmp_path)

    assert cli.main(["train", "--episodes", "1", "--algo", "ppo"]) == 0
    default = captured["save"]
    assert default.parent == Path("policy")
    assert default.name.startswith("ppo-") and default.suffix == ".zip"
    assert default.parent.is_dir(), "the run can write where it says it will"

    assert cli.main(["train", "--episodes", "1", "--method", "cadrl"]) == 0
    assert captured["save"].parent == Path("policy")
    assert captured["save"].name.startswith("cadrl-")
    assert captured["save"].suffix == ".pt"

    assert cli.main(["train", "--episodes", "1", "--algo", "ppo", "--no-save"]) == 0
    assert captured["save"] is None

    assert (
        cli.main(["train", "--episodes", "1", "--algo", "ppo", "--save", "mine.zip"]) == 0
    )
    assert captured["save"] == Path("mine.zip")


def test_a_run_cannot_be_told_both_to_save_and_not_to():
    """Two answers to one question are refused by the parser, not resolved silently."""
    with pytest.raises(SystemExit):
        cli.main(["train", "--episodes", "1", "--save", "a.zip", "--no-save"])
