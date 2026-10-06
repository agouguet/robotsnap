"""Named configuration files for the command line.

A run has dozens of options and a study wants to name one it can repeat:
``--config ppo`` should mean the hyper-parameters a PPO run starts from, and a
flag typed on the same line should still win. This package is what turns a name
or a path into the mapping of options a sub-command applies, and what keeps the
two levels apart - a *method* of the literature (CADRL, SARL) and a *learning
rule* (PPO, DQN) - by living them in different folders of the same tree.

Nothing here imports a framework: the module is argv-shaped data handling, so
``import robotsnap.config`` stays usable in an interpreter that has only the
base package. YAML is read through PyYAML, imported inside the reader that
needs it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

__all__ = [
    "CONFIG_DIRECTORY",
    "ConfigError",
    "apply_to",
    "describe",
    "find",
    "known",
]

#: The ``configs/`` tree of the checkout. This module lives at
#: ``src/robotsnap/config/``, so the repository root is three parents up and the
#: tree a caller names is never the package's own installation directory.
CONFIG_DIRECTORY: Path = Path(__file__).resolve().parents[3] / "configs"

#: The sub-trees a simple name is looked up in, in the order it is tried. The
#: order is what makes ``--config template`` unambiguous when two folders hold
#: a file of that name: the learning rules come first, then the methods.
_SEARCH_DIRECTORIES: tuple[str, ...] = (
    "algorithms",
    "methods",
    "curriculum",
    "benchmarks",
)

#: The suffixes a configuration may carry, in the order a bare name is tried.
_SUFFIXES: tuple[str, ...] = (".yaml", ".yml", ".json")

#: Command spellings a caller may use that are not themselves registered names.
#: ``benchmark`` is the natural word for ``bench`` and was used in scripts
#: before the command was shortened; both spellings mean the same sub-parser.
_COMMAND_ALIASES: dict[str, str] = {"benchmark": "bench", "infer": "play"}


class ConfigError(ValueError):
    """A configuration could not be found, read, or applied."""


# ---------------------------------------------------------------------------
# What exists, and what a name resolves to.
# ---------------------------------------------------------------------------


def _configuration_files(directory: Path) -> list[Path]:
    """Every readable configuration directly under ``directory``, sorted."""
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in _SUFFIXES
    )


def _names_in(directory: Path) -> tuple[str, ...]:
    """The stems of the configurations under ``directory``, sorted and unique."""
    return tuple(dict.fromkeys(path.stem for path in _configuration_files(directory)))


def known() -> dict[str, tuple[str, ...]]:
    """The configuration names present on disk, per tree, at the time of call.

    Read from the filesystem rather than kept as a table so that adding a
    configuration is dropping a file in a folder: the list a refusal prints is
    the list a caller can actually use, in the session that asks.
    """
    return {
        name: _names_in(CONFIG_DIRECTORY / name) for name in _SEARCH_DIRECTORIES
    }


def _available() -> str:
    """The one-line inventory a "not found" refusal ends with."""
    parts = [
        f"{name} ({', '.join(names)})"
        for name, names in known().items()
        if names
    ]
    if not parts:
        return f"{CONFIG_DIRECTORY} holds no configuration file"
    return "available: " + "; ".join(parts)


def _looks_like_a_path(spec: str | Path) -> bool:
    """Whether ``spec`` names a location rather than a configuration name.

    An absolute path, or one with a separator in it, can only be a path. A bare
    word is a name; a bare file name that exists in the working directory is
    that file, which is what makes ``--config ppo.yaml`` work from a folder
    holding the files it names.
    """
    if isinstance(spec, Path):
        return spec.is_absolute() or len(spec.parts) > 1
    text = str(spec).strip()
    if not text:
        return False
    candidate = Path(text)
    return candidate.is_absolute() or len(candidate.parts) > 1 or candidate.exists()


def _has_suffix(name: str) -> bool:
    return name.lower().endswith(_SUFFIXES)


def find(spec: str | Path) -> Path:
    """Resolve ``spec`` to the configuration file it names.

    ``spec`` is either a path - absolute, or relative to the working directory -
    or a bare name, which is looked up in the four trees of
    :data:`CONFIG_DIRECTORY` in order and then anywhere under the tree. A name
    that answers to nothing raises :class:`ConfigError` naming every
    configuration that does exist, because a typo is the common case and the
    fix is picking from the list.
    """
    if _looks_like_a_path(spec):
        path = Path(spec).expanduser()
        if path.is_file():
            return path.resolve()
        raise ConfigError(f"no configuration file at {path}: {_available()}")

    name = str(spec).strip()
    if not name:
        raise ConfigError(f"an empty configuration name was given: {_available()}")
    filenames = [name] if _has_suffix(name) else [f"{name}{suffix}" for suffix in _SUFFIXES]

    for directory in _SEARCH_DIRECTORIES:
        for filename in filenames:
            candidate = CONFIG_DIRECTORY / directory / filename
            if candidate.is_file():
                return candidate.resolve()

    for filename in filenames:
        for path in sorted(CONFIG_DIRECTORY.rglob(filename)):
            if path.is_file():
                return path.resolve()

    raise ConfigError(f"no configuration named {spec!r}: {_available()}")


# ---------------------------------------------------------------------------
# Reading a file.
# ---------------------------------------------------------------------------


def _yaml():
    """PyYAML, or a refusal that says what to install.

    Imported here rather than at module import so that a session which only
    reads a JSON configuration, or only lists the names, never pays for the
    dependency. PyYAML ships with ``rosbags``, which this package already
    requires, so the missing case is worth a precise message rather than an
    ``ImportError`` out of ``yaml.safe_load``.
    """
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ConfigError(
            "reading a YAML configuration needs PyYAML, which is not installed: "
            "install PyYAML (it comes with rosbags), or pass a .json file instead"
        ) from exc
    return yaml


def load(spec: str | Path) -> dict[str, Any]:
    """Read the configuration ``spec`` names and return it as a mapping.

    A file that is empty, whose root is not an object, or whose keys are not
    strings is refused here rather than halfway through ``apply_to``, because
    the message a caller can act on belongs next to the file it describes.
    """
    path = find(spec)
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix == ".json":
        try:
            document = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"{path} is not valid JSON: {exc}") from exc
    elif suffix in (".yaml", ".yml"):
        yaml = _yaml()
        try:
            document = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ConfigError(f"{path} is not valid YAML: {exc}") from exc
    else:
        raise ConfigError(f"{path}: a configuration is a .yaml, .yml or .json file")

    if document is None:
        raise ConfigError(f"{path} is empty")
    if not isinstance(document, Mapping):
        raise ConfigError(
            f"{path} holds a {type(document).__name__}; a configuration is a "
            "mapping of option names to values"
        )
    bad = [key for key in document if not isinstance(key, str)]
    if bad:
        raise ConfigError(f"{path}: configuration keys must be strings, got {bad!r}")
    return dict(document)


# ---------------------------------------------------------------------------
# Applying a configuration to a parser.
# ---------------------------------------------------------------------------


def apply_to(parser: Any, command: str, document: Mapping[str, Any]) -> list[str]:
    """Set ``document``'s values as defaults on ``command``'s parser.

    Every key is checked against the options the command really defines before
    one of them is set, so a document that names an option this command does
    not have is refused in one message instead of half-applied. The table is
    read off the parser itself - the map from a configuration key to an
    ``argparse`` destination is the command's own actions - which is what keeps
    the two from drifting as options are added.

    The defaults are what a flag overrides, so a run that passes ``--gamma``
    keeps it and the configuration supplies only what the line left out.
    ``parser`` may be the top-level parser, in which case ``command``'s
    sub-parser is found and used, or the sub-parser itself.
    """
    if not isinstance(document, Mapping):
        raise ConfigError(
            f"a configuration is a mapping of option names to values, got "
            f"{type(document).__name__}"
        )
    bad = [key for key in document if not isinstance(key, str)]
    if bad:
        raise ConfigError(f"configuration keys must be strings, got {bad!r}")

    target, destinations = _command_destinations(parser, command)
    unknown = sorted(key for key in document if key not in destinations)
    if unknown:
        named = ", ".join(repr(key) for key in unknown)
        accepted = ", ".join(sorted(destinations))
        raise ConfigError(f"{command!r} has no option {named}; it accepts {accepted}")

    defaults = {destinations[key]: value for key, value in document.items()}
    target.set_defaults(**defaults)
    return sorted(defaults)


def _is_parser(value: Any) -> bool:
    return hasattr(value, "_actions") and hasattr(value, "set_defaults")


def _sub_parsers(parser: Any):
    """The sub-parser table of ``parser``, or ``None`` when it has none."""
    for action in getattr(parser, "_actions", ()):
        choices = getattr(action, "choices", None)
        if (
            isinstance(choices, dict)
            and choices
            and all(_is_parser(value) for value in choices.values())
        ):
            return action
    return None


def _command_destinations(parser: Any, command: str) -> tuple[Any, dict[str, str]]:
    """``(parser to set, {configuration key: argparse destination})``.

    Reading the destinations off the command's own parser is the only table
    that cannot disagree with it: today every configuration key is the
    destination itself - ``observation_params`` above ``--observation-params`` -
    and a flag that ever maps two spellings onto one destination is covered
    without a second list to keep in step.
    """
    sub_parsers = _sub_parsers(parser)
    if sub_parsers is None:
        return parser, _option_destinations(parser)

    name = str(command).strip()
    if name not in sub_parsers.choices:
        name = _COMMAND_ALIASES.get(name, name)
    if name not in sub_parsers.choices:
        known = ", ".join(sorted(sub_parsers.choices))
        raise ConfigError(f"{command!r} is not a command of this parser; it has {known}")
    target = sub_parsers.choices[name]
    return target, _option_destinations(target)


def _option_destinations(parser: Any) -> dict[str, str]:
    """The configuration keys a parser accepts, and the destination of each.

    Only real options count: the help flag and any destination a command sets
    without an option of its own - the handler it dispatches to - are not
    values a configuration has any business carrying.
    """
    destinations: dict[str, str] = {}
    for action in getattr(parser, "_actions", ()):
        option_strings = getattr(action, "option_strings", ())
        dest = getattr(action, "dest", None)
        if not option_strings or not dest or dest in ("help", "==SUPPRESS=="):
            continue
        destinations[str(dest)] = str(dest)
    return destinations


# ---------------------------------------------------------------------------
# Showing a configuration back.
# ---------------------------------------------------------------------------


def describe(document: Mapping[str, Any]) -> str:
    """One ``key = value`` line per entry, sorted, for a ``--print-config``.

    A mapping or a sequence is shown as its JSON form so a nest stays on one
    line and stays copyable; everything else keeps Python's own spelling.
    """
    if not isinstance(document, Mapping):
        raise ConfigError(
            f"a configuration is a mapping of option names to values, got "
            f"{type(document).__name__}"
        )
    return "\n".join(
        f"{key} = {_render(document[key])}" for key in sorted(document, key=str)
    )


def _render(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(value, sort_keys=False, default=str)
    return repr(value)
