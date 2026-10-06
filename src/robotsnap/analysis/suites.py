"""The named scenario suites a benchmark campaign runs, and how a caller names one.

A campaign is a list of scenarios times a number of episodes; a *suite* is that
list of scenarios, so a caller who does not want to type seven ids every time
names ``"basic"`` and lets the default do the rest. Three spellings are
accepted, because the three callers this serves spell it three ways: the name of
a suite the package ships, a path to a YAML file a person wrote, and a plain
comma-separated list that a shell command line hands over as one argument.

Nothing here needs Unity or the environment: the module is a resolver over
strings. Its one optional dependency - ``yaml``, to read a suite file - is
imported inside the function that reads the file, so importing
``robotsnap.analysis`` stays free of it.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

__all__ = [
    "DEFAULT_SUITES",
    "DEFAULT_SUITE",
    "SuitesError",
    "suite_names",
    "scenarios_for",
    "load_suite",
]

#: Extensions a suite file is recognised by even before it is read.
_SUITE_EXTENSIONS = (".yaml", ".yml")

#: The suites the package ships. ``basic`` is the social-navigation set the
#: campaign command runs when the caller names nothing; ``basic_short`` is the
#: same list cut to six scenarios, the one a smoke test runs.
DEFAULT_SUITES: dict[str, tuple[str, ...]] = {
    "basic": (
        "front_approach",
        "door_passing",
        "intersection",
        "circle_crowd",
        "perpendicular_traffic",
        "corner",
        "crowd",
    ),
    "basic_short": (
        "front_approach",
        "door_passing",
        "intersection",
        "circle_crowd",
        "perpendicular_traffic",
        "corner",
    ),
}

#: The suite a caller gets by naming none.
DEFAULT_SUITE = "basic"


class SuitesError(ValueError):
    """A suite spec that names nothing this module can resolve to scenarios."""


def suite_names() -> tuple[str, ...]:
    """The names of the suites the package ships."""
    return tuple(DEFAULT_SUITES)


def scenarios_for(spec: str) -> tuple[str, ...]:
    """The scenario ids a suite spec resolves to.

    ``spec`` is a built-in suite name, a path to a suite YAML file, or a
    comma-separated list of scenario ids. Anything else raises
    :class:`SuitesError` rather than returning an empty campaign - a typo in a
    scenario name is worth a real error, not a run that measured nothing.
    """
    text = str(spec).strip()
    if not text:
        raise SuitesError("a suite spec must not be empty")
    if text in DEFAULT_SUITES:
        return DEFAULT_SUITES[text]

    candidate = Path(text)
    if candidate.suffix.lower() in _SUITE_EXTENSIONS or candidate.is_file():
        return load_suite(candidate)["scenarios"]

    if "," in text:
        names = tuple(part.strip() for part in text.split(",") if part.strip())
        if not names:
            raise SuitesError(f"suite spec {spec!r} names no scenario")
        return names

    raise SuitesError(
        f"unknown suite {spec!r}: name one of {', '.join(suite_names())}, a suite YAML path, "
        "or a comma-separated list of scenario ids"
    )


def load_suite(path: str | Path) -> dict[str, Any]:
    """Read a suite YAML file into the document the campaign command reads.

    The result is a plain dict with ``name``, ``description``, ``scenarios`` and
    ``episodes_per_scenario``. ``scenarios`` is required and non-empty - a suite
    with no scenario is a mistake worth reporting - while ``description`` and
    ``episodes_per_scenario`` are optional and default to ``None``.
    """
    target = Path(path)
    try:
        import yaml
    except ImportError as error:  # pragma: no cover - exercised on a bare install
        raise SuitesError(
            "reading a suite file needs PyYAML; install it or name a built-in suite"
        ) from error

    if not target.is_file():
        raise SuitesError(f"suite file not found: {target}")
    try:
        document = yaml.safe_load(target.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise SuitesError(f"{target} is not readable YAML: {error}") from error
    if not isinstance(document, Mapping):
        raise SuitesError(f"{target} does not hold a YAML mapping")

    return {
        "name": str(document.get("name") or target.stem),
        "description": document.get("description"),
        "scenarios": _scenarios_of(document, source=str(target)),
        "episodes_per_scenario": _episodes_per_scenario(
            document.get("episodes_per_scenario"), source=str(target)
        ),
    }


def _scenarios_of(document: Mapping[str, Any], source: str) -> tuple[str, ...]:
    """The scenario ids of a suite document, insisting on a non-empty list of names."""
    scenarios = document.get("scenarios")
    if not isinstance(scenarios, (list, tuple)) or not scenarios:
        raise SuitesError(f"{source} needs a non-empty 'scenarios' list")
    names = tuple(str(name).strip() for name in scenarios if str(name).strip())
    if len(names) != len(scenarios):
        raise SuitesError(f"{source} has a blank scenario id")
    return names


def _episodes_per_scenario(value: Any, source: str) -> int | None:
    """The episode budget of a suite document, or ``None`` when it does not name one."""
    if value is None:
        return None
    # ``True`` is an ``int`` in Python and is almost certainly a typo for 1
    # written by hand, so it is refused rather than silently read as one episode.
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SuitesError(f"{source} needs a positive integer 'episodes_per_scenario'")
    return value
