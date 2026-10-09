"""Where a training run writes the policy it learned.

``train`` writes one checkpoint when it ends. The command line says where: a
file, a directory, or nothing at all - and then the run writes into ``policy/``,
the directory that ships with the repository, under a name that carries the run
and the moment:

    policy/cadrl-20261006-181500.pt

Two runs of the same command therefore never overwrite each other. The
directory is versioned; what training writes into it is not.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

#: The directory a run writes into when the command line names none.
DEFAULT_DIRECTORY_NAME = "policy"

#: The suffix is the writer's own: the package's trainers write a torch
#: document, Stable-Baselines3 writes its own zip archive.
TORCH_SUFFIX = ".pt"
STABLE_BASELINES_SUFFIX = ".zip"

#: The one learning rule that is this package's own loop rather than a
#: Stable-Baselines3 algorithm, and so writes a torch document.
OWN_LOOP_RULE = "reinforce"


def default_directory() -> Path:
    """The directory an unnamed ``--save`` writes into, relative to the caller."""
    return Path(DEFAULT_DIRECTORY_NAME)


def suffix_for(*, algorithm: str | None = None, method: str | None = None) -> str:
    """The suffix the checkpoint of this run carries, which is its writer's own.

    A method of the literature and this package's own REINFORCE loop are
    written by the package, so they carry ``.pt``; the learning rules that
    Stable-Baselines3 trains carry the ``.zip`` it writes itself.

    ``algorithm`` is ``None`` when a method is what was asked for, because a
    method is not a learning rule and the command line never names both.
    """
    if method is not None or algorithm == OWN_LOOP_RULE:
        return TORCH_SUFFIX
    return STABLE_BASELINES_SUFFIX


def generated_name(label: str, *, suffix: str, moment: datetime | None = None) -> str:
    """``<label>-<YYYYmmdd-HHMMSS><suffix>``: the name of a run that was not told one."""
    stamp = (moment or datetime.now()).strftime("%Y%m%d-%H%M%S")
    return f"{label}-{stamp}{suffix}"


def resolve_save_path(
    target: str | os.PathLike[str] | None,
    *,
    label: str,
    suffix: str,
    moment: datetime | None = None,
) -> Path:
    """Turn a ``--save`` argument into the file a run writes.

    ``target`` is what the command line holds, or ``None`` when it holds
    nothing. A target that names a directory - one that exists, or one written
    with a trailing separator - receives a generated file name; anything else
    is taken as the file itself.

    The parent directory is created, so naming a file inside a directory that
    does not exist yet still works. ``label`` names the run in the generated
    file name: the method, or the learning rule.
    """
    if target is None:
        path = default_directory() / generated_name(label, suffix=suffix, moment=moment)
    else:
        written = os.fspath(target)
        candidate = Path(written).expanduser()
        if written.endswith(("/", os.sep)) or candidate.is_dir():
            path = candidate / generated_name(label, suffix=suffix, moment=moment)
        else:
            path = candidate

    path.parent.mkdir(parents=True, exist_ok=True)
    return path
