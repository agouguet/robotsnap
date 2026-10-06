"""A curriculum: the ordered ladder of stages a training run climbs.

A training run on the social-navigation task starts too hard: a policy dropped
straight into a crowd has almost nothing to learn from, because every episode
ends in a collision before the reward can say anything about the goal. A
curriculum answers that by making the task easier first and letting the run
promote itself once the easy version is solved - the staged-training recipe,
kept here as its own small object so neither :mod:`robotsnap.envs` nor a
training loop has to know about it.

What lives here is only the *decision*: which rung we are on, whether the
success rate over the last few episodes clears its bar, and when the ladder has
been climbed. Applying a stage to a live session is the job of
:mod:`robotsnap.rl.curriculum_env`, which needs nothing from here beyond the
object itself; a subclass of the environment can use the same object directly
if it would rather not wrap.

Nothing here imports ``gymnasium``, ``numpy`` or ``torch``: a stage is a plain
description, and ``yaml`` is pulled in only when a document is actually loaded,
so the command line stays importable without the training extras.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "DEFAULT_CURRICULUM_DIRECTORY",
    "Curriculum",
    "CurriculumError",
    "Stage",
]

#: The repository root, three directories above this module
#: (``src/robotsnap/rl/curriculum.py``). Curriculum names resolve under it.
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

#: Where a curriculum written in YAML is looked up when a caller passes a name
#: rather than a path. The shipped examples live in ``configs/curriculum/``.
DEFAULT_CURRICULUM_DIRECTORY = REPOSITORY_ROOT / "configs" / "curriculum"

#: The keys a stage may carry. Anything else is a typo, and is refused rather
#: than silently ignored.
_STAGE_KEYS = frozenset(
    {
        "name",
        "scenario",
        "scenario_fields",
        "episodes",
        "success_threshold",
        "window",
        "min_episodes",
        "options",
    }
)


class CurriculumError(ValueError):
    """A curriculum document, or the way it is addressed, is not usable."""


@dataclass(frozen=True)
class Stage:
    """One rung: a world to run, and the rule that leaves it.

    ``scenario`` names the Unity scenario to launch, ``scenario_fields`` the
    fields to rewrite in that scenario's YAML first, and ``options`` anything
    else the reset should be given - they are all optional, so a stage may only
    tune a difficulty knob without touching the scene at all.

    ``episodes`` is a budget; ``success_threshold`` over a trailing ``window``
    of episodes is an early exit, honoured only once ``min_episodes`` have been
    played. A stage with *neither* never ends by itself - it is the fixed final
    rung, and a run is allowed to sit on it for ever.
    """

    name: str
    scenario: str | None = None
    scenario_fields: Mapping[str, Any] | None = None
    episodes: int | None = None
    success_threshold: float | None = None
    window: int = 20
    min_episodes: int = 0
    options: Mapping[str, Any] = field(default_factory=dict)


class Curriculum:
    """An ordered ladder of stages, with the counters of the current rung.

    The object is the whole state of a curriculum run: the stage it is on and
    how the last few episodes on that stage went. :meth:`observe` is called once
    per finished episode and is the only thing that moves the ladder on.
    """

    def __init__(self, stages: Sequence[Stage]) -> None:
        stages = tuple(stages)
        if not stages:
            raise CurriculumError("a curriculum needs at least one stage")
        for index, stage in enumerate(stages):
            if not isinstance(stage, Stage):
                raise CurriculumError(f"stage {index} is not a Stage")
            if not stage.name:
                raise CurriculumError(f"stage {index} has no name")
        self.stages: tuple[Stage, ...] = stages
        self._index = 0
        self._seen: list[bool] = []
        self._complete = False

    # -- reading the ladder ------------------------------------------------

    @property
    def current(self) -> Stage:
        """The stage being played."""
        return self.stages[self._index]

    @property
    def index(self) -> int:
        """Zero-based index of the stage being played."""
        return self._index

    @property
    def finished(self) -> bool:
        """Whether every stage has been passed."""
        return self._complete

    def reset_options(self) -> dict[str, Any]:
        """The options to hand the environment's ``reset`` for this stage.

        The scenario is the stage's, and the stage's own options are copied
        underneath it, so a stage can carry extra reset switches. Scenario
        *fields* are not part of this - the environment writes them from its own
        configuration, and :class:`~robotsnap.rl.curriculum_env.CurriculumEnv`
        copies the stage's on before it resets.
        """
        options = dict(self.current.options)
        if self.current.scenario is not None:
            options["scenario"] = self.current.scenario
        return options

    def state(self) -> dict[str, Any]:
        """The line of logging a training loop writes each episode.

        ``episodes`` and ``successes`` count the current stage only, and
        ``success_rate`` is their ratio (the pass rule reads a trailing window
        of the same list, which is why the two can disagree near a promotion).
        """
        episodes = len(self._seen)
        successes = sum(1 for success in self._seen if success)
        return {
            "stage": self._index,
            "name": self.current.name,
            "episodes": episodes,
            "successes": successes,
            "success_rate": (successes / episodes) if episodes else 0.0,
            "finished": self._complete,
        }

    def summary(self) -> str:
        """One readable line for the training log."""
        state = self.state()
        return (
            f"curriculum {state['stage'] + 1}/{len(self.stages)} "
            f"{state['name']!r} episodes={state['episodes']} "
            f"successes={state['successes']} "
            f"success_rate={state['success_rate']:.2f} "
            f"{'finished' if state['finished'] else 'running'}"
        )

    # -- moving the ladder -------------------------------------------------

    def observe(
        self, *, success: bool, info: Mapping[str, Any] | None = None
    ) -> bool:
        """Record one finished episode; return True when the stage is passed.

        The stage is passed when its episode budget is spent, or when enough
        episodes have been played and the success rate over the last ``window``
        of them reaches ``success_threshold`` - whichever comes first. A stage
        that gives neither never passes. ``info`` is accepted for symmetry with
        the wrapper and is not read here; the rule is the boolean.
        """
        if self._complete:
            return False
        self._seen.append(bool(success))
        if not self._passed(self.current):
            return False
        return self._promote()

    def advance(self) -> bool:
        """Move to the next stage by hand; False if already on the last."""
        if self._complete or self._index >= len(self.stages) - 1:
            return False
        self._index += 1
        self._seen = []
        return True

    def _passed(self, stage: Stage) -> bool:
        """Whether the current counters satisfy the stage's exit rule."""
        episodes = len(self._seen)
        if stage.episodes is not None and episodes >= stage.episodes:
            return True
        if stage.success_threshold is None:
            return False
        if episodes < stage.window or episodes < stage.min_episodes:
            return False
        recent = self._seen[-stage.window :]
        rate = sum(1 for success in recent if success) / len(recent)
        return rate >= stage.success_threshold

    def _promote(self) -> bool:
        """Leave the passed stage: next rung, or the top of the ladder."""
        if self._index >= len(self.stages) - 1:
            self._complete = True
            return True
        self._index += 1
        self._seen = []
        return True

    # -- loading -----------------------------------------------------------

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> "Curriculum":
        """Build a curriculum from an already-parsed mapping."""
        if not isinstance(document, Mapping):
            raise CurriculumError("a curriculum must be a mapping")
        raw = document.get("stages")
        if raw is None:
            raise CurriculumError("the curriculum has no 'stages'")
        if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
            raise CurriculumError("'stages' must be a non-empty list")
        if not raw:
            raise CurriculumError("'stages' must be a non-empty list")
        stages = tuple(
            _stage_from_mapping(item, index=index) for index, item in enumerate(raw)
        )
        return cls(stages)

    @classmethod
    def load(cls, spec: str | Path) -> "Curriculum":
        """Load a curriculum by name under ``configs/curriculum`` or by path."""
        path = _resolve_spec(spec)
        document = _read_yaml(path)
        return cls.from_document(document)


def _stage_from_mapping(item: Any, *, index: int) -> Stage:
    """Validate one stage mapping and turn it into a :class:`Stage`."""
    if not isinstance(item, Mapping):
        raise CurriculumError(f"stage {index} is not a mapping")
    unknown = set(item) - _STAGE_KEYS
    if unknown:
        names = ", ".join(sorted(str(key) for key in unknown))
        raise CurriculumError(f"stage {index} has unknown key(s): {names}")
    name = item.get("name")
    if not isinstance(name, str) or not name.strip():
        raise CurriculumError(f"stage {index} needs a non-empty name")
    scenario = item.get("scenario")
    if scenario is not None and not isinstance(scenario, str):
        raise CurriculumError(f"stage {name!r} scenario must be a string")
    scenario_fields = item.get("scenario_fields")
    if scenario_fields is not None and not isinstance(scenario_fields, Mapping):
        raise CurriculumError(f"stage {name!r} scenario_fields must be a mapping")
    options = item.get("options")
    if options is not None and not isinstance(options, Mapping):
        raise CurriculumError(f"stage {name!r} options must be a mapping")
    window = _count(item.get("window", 20), "window", name, minimum=1)
    min_episodes = _count(item.get("min_episodes", 0), "min_episodes", name, minimum=0)
    episodes = _count(item.get("episodes"), "episodes", name, minimum=0)
    threshold = item.get("success_threshold")
    if threshold is not None:
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise CurriculumError(f"stage {name!r} success_threshold must be a number")
        if not 0.0 <= float(threshold) <= 1.0:
            raise CurriculumError(
                f"stage {name!r} success_threshold must be within [0, 1]"
            )
    return Stage(
        name=name,
        scenario=scenario,
        scenario_fields=None if scenario_fields is None else dict(scenario_fields),
        episodes=episodes,
        success_threshold=None if threshold is None else float(threshold),
        window=window,
        min_episodes=min_episodes,
        options={} if options is None else dict(options),
    )


def _count(value: Any, key: str, name: str, *, minimum: int) -> int | None:
    """A whole number at least ``minimum``, or ``None`` when absent."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise CurriculumError(f"stage {name!r} {key} must be a whole number")
    if value < minimum:
        raise CurriculumError(f"stage {name!r} {key} must be at least {minimum}")
    return value


def _resolve_spec(spec: str | Path) -> Path:
    """Find the file a name or path refers to, without importing the package."""
    text = str(spec)
    # A path that exists as typed - absolute, or relative to the caller's cwd.
    candidate = Path(text)
    if candidate.is_file():
        return candidate
    # A relative path typed from somewhere else, still meant from the repo.
    rooted = REPOSITORY_ROOT / candidate
    if not candidate.is_absolute() and rooted.is_file():
        return rooted
    # A bare name: look for <name>.yaml / <name>.yml under the curriculum dir.
    if len(candidate.parts) == 1:
        stem = candidate.stem if candidate.suffix else candidate.name
        for suffix in (".yaml", ".yml"):
            named = DEFAULT_CURRICULUM_DIRECTORY / f"{stem}{suffix}"
            if named.is_file():
                return named
    # A bridge to robotsnap.config when that module exists, used loosely: it is
    # another agent's surface, and this module must resolve names without it.
    try:
        from robotsnap.config import find  # type: ignore
    except Exception:  # pragma: no cover - optional integration
        find = None
    if find is not None:
        try:
            found = Path(find(text))
        except Exception:  # pragma: no cover - optional integration
            found = None
        if found is not None and found.is_file():
            return found
    raise CurriculumError(
        f"no curriculum named {text!r} under {DEFAULT_CURRICULUM_DIRECTORY}"
    )


def _read_yaml(path: Path) -> Any:
    """Parse a YAML document, refusing anything a reader cannot use."""
    import yaml  # imported here so the module stays importable without it

    try:
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    except yaml.YAMLError as error:
        raise CurriculumError(f"invalid YAML in {path}: {error}") from error
    except OSError as error:
        raise CurriculumError(f"cannot read curriculum {path}: {error}") from error
