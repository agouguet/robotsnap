"""Running a benchmark campaign, saving it, and comparing two of them.

A campaign runs each scenario of a suite for a fixed number of episodes, keeps
every episode document Unity produced, averages them with
:func:`robotsnap.analysis.metrics.summarize`, and writes the lot to one JSON file
a second campaign can be compared against.

The environment belongs to the caller: this module never builds, opens or closes
one, and reads an episode through an injected callable rather than a hard-wired
client. That is what lets the same loop drive a live Unity session, a replayed
client or a fake environment in a test.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from robotsnap.analysis.metrics import (
    EPISODE_KEYS,
    OUTCOMES,
    _SCALAR_METRICS,
    is_episode,
    summarize,
)

__all__ = [
    "CampaignError",
    "ScenarioResult",
    "Campaign",
    "run_campaign",
    "save_campaign",
    "load_campaign",
    "compare_campaigns",
    "format_comparison",
    "format_summary",
]

#: The top-level rates a comparison reports. ``goal_rate`` is left out because it
#: is ``success_rate`` under another name; the rest are the failure rates.
_RATE_KEYS: tuple[str, ...] = ("success_rate",) + tuple(
    f"{outcome}_rate" for outcome in OUTCOMES if outcome != "goal"
)


class CampaignError(ValueError):
    """A campaign that cannot be run, or a document that cannot be read back."""


@dataclass
class ScenarioResult:
    """One scenario of a suite: the episodes it produced and their averages."""

    scenario: str
    episodes: list[dict[str, Any]]
    summary: dict[str, Any]


@dataclass
class Campaign:
    """A whole benchmark run: the suite, the policy, and what each scenario did."""

    name: str
    suite: str | None
    scenarios: tuple[str, ...]
    episodes_per_scenario: int
    policy: str | None
    started_at: str
    finished_at: str
    results: list[ScenarioResult]

    @property
    def episodes(self) -> list[dict[str, Any]]:
        """Every episode of every scenario, in the order the campaign played them."""
        return [episode for result in self.results for episode in result.episodes]

    @property
    def summary(self) -> dict[str, Any]:
        """The whole campaign aggregated, as if it were one long session."""
        return summarize(self.episodes)

    def to_document(self) -> dict[str, Any]:
        """The campaign as a JSON-serialisable document, in a stable key order."""
        return {
            "name": self.name,
            "suite": self.suite,
            "scenarios": list(self.scenarios),
            "episodes_per_scenario": self.episodes_per_scenario,
            "policy": self.policy,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "results": [
                {
                    "scenario": result.scenario,
                    "summary": result.summary,
                    "episodes": result.episodes,
                }
                for result in self.results
            ],
        }

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> "Campaign":
        """Rebuild a campaign from the document :meth:`to_document` writes."""
        if not isinstance(document, Mapping):
            raise CampaignError("a campaign document must be a mapping")
        try:
            return cls(
                name=str(document["name"]),
                suite=None if document.get("suite") is None else str(document["suite"]),
                scenarios=tuple(str(name) for name in document["scenarios"]),
                episodes_per_scenario=int(document["episodes_per_scenario"]),
                policy=None if document.get("policy") is None else str(document["policy"]),
                started_at=str(document["started_at"]),
                finished_at=str(document["finished_at"]),
                results=[_scenario_result(item) for item in document["results"]],
            )
        except (KeyError, TypeError, ValueError) as error:
            raise CampaignError(f"not a campaign document: {error}") from error


def _scenario_result(item: Any) -> ScenarioResult:
    """One entry of a campaign document's result list, rebuilt."""
    if not isinstance(item, Mapping):
        raise CampaignError("a campaign result must be a mapping")
    episodes = item.get("episodes")
    summary = item.get("summary")
    if not isinstance(episodes, list) or not isinstance(summary, Mapping):
        raise CampaignError("a campaign result needs an 'episodes' list and a 'summary'")
    return ScenarioResult(
        scenario=str(item.get("scenario", "")),
        episodes=[dict(episode) for episode in episodes],
        summary=dict(summary),
    )


def run_campaign(
    *,
    environment: Any,
    scenarios: Sequence[str],
    episodes_per_scenario: int,
    act: Callable[[Any, Mapping[str, Any]], Any],
    name: str = "campaign",
    suite: str | None = None,
    policy: str | None = None,
    seed: int | None = None,
    max_steps: int = 1000,
    on_episode: Callable[[int, str, str], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    episode_reader: Callable[[str, int], Mapping[str, Any] | None] | None = None,
) -> Campaign:
    """Play ``scenarios`` x ``episodes_per_scenario`` episodes and collect the metrics.

    Every scenario is applied with ``reset(options={"scenario": ..., "launch":
    True})``, so a campaign over one long-lived environment still sees each
    scenario's own world. ``act`` is called with ``(observation, info)`` at every
    step; an episode ends on ``terminated``, ``truncated`` or after ``max_steps``.

    The episode document comes from ``episode_reader(scenario, index)`` when one
    is given, and otherwise from the last episode of
    ``environment.client.episodes()``. An episode that produced no document is
    counted as a failure - the run keeps going and the count is warned about at
    the end - because a run we cannot measure must widen the sample it lowers.

    ``on_episode`` is called after each episode with its 1-based index.
    ``should_stop`` is polled before each episode, so a Ctrl-C handler stops
    between episodes rather than in the middle of one.
    """
    scenario_ids = tuple(str(scenario) for scenario in scenarios)
    if not scenario_ids:
        raise CampaignError("a campaign needs at least one scenario")
    if int(episodes_per_scenario) < 1:
        raise CampaignError("a campaign needs at least one episode per scenario")
    if int(max_steps) < 1:
        raise CampaignError("max_steps must be at least 1")

    started_at = _now_iso()
    results: list[ScenarioResult] = []
    missing: list[tuple[str, int]] = []
    episode_offset = 0
    stopped = False

    for scenario in scenario_ids:
        episodes: list[dict[str, Any]] = []
        for index in range(1, int(episodes_per_scenario) + 1):
            if should_stop is not None and should_stop():
                stopped = True
                break
            observation, info = _reset(
                environment, scenario, None if seed is None else seed + episode_offset
            )
            episode_offset += 1
            steps = 0
            terminated = truncated = False
            while True:
                observation, _, terminated, truncated, info = environment.step(
                    act(observation, info)
                )
                steps += 1
                if terminated or truncated or steps >= int(max_steps):
                    break

            document = _episode_document(environment, episode_reader, scenario, index)
            if document is None:
                missing.append((scenario, index))
                document = _missing_episode(scenario, index)
            episodes.append(document)
            if on_episode is not None:
                on_episode(index, scenario, _episode_line(scenario, index, episodes_per_scenario, document))

        results.append(
            ScenarioResult(scenario=scenario, episodes=episodes, summary=summarize(episodes))
        )
        if stopped:
            break

    campaign = Campaign(
        name=name,
        suite=suite,
        scenarios=scenario_ids,
        episodes_per_scenario=int(episodes_per_scenario),
        policy=policy,
        started_at=started_at,
        finished_at=_now_iso(),
        results=results,
    )
    if missing:
        warnings.warn(
            f"{len(missing)} episode(s) produced no metrics document and were counted as "
            "failures: "
            + ", ".join(f"{scenario}#{index}" for scenario, index in missing),
            RuntimeWarning,
            stacklevel=2,
        )
    return campaign


def _reset(environment: Any, scenario: str, seed: int | None) -> Any:
    """Reset onto ``scenario``, forcing a full Unity reload, with a seed when asked."""
    options = {"scenario": scenario, "launch": True}
    if seed is None:
        return environment.reset(options=options)
    return environment.reset(seed=seed, options=options)


def _episode_document(
    environment: Any,
    episode_reader: Callable[[str, int], Mapping[str, Any] | None] | None,
    scenario: str,
    index: int,
) -> dict[str, Any] | None:
    """The document of the episode that just finished, or ``None``.

    A reader's answer only counts when it is a whole episode document: a partial
    one would be averaged over a different denominator than the one it was
    counted in, so it is treated the same as no answer at all.
    """
    document = (
        episode_reader(scenario, index)
        if episode_reader is not None
        else _client_episode(environment)
    )
    if isinstance(document, Mapping) and is_episode(document):
        return dict(document)
    return None


def _client_episode(environment: Any) -> Any:
    """The last episode a connected client has finished, or ``None``.

    A client that cannot answer is a missing episode, not a broken campaign:
    raising here would throw away the episodes that did come back.
    """
    client = getattr(environment, "client", None)
    finished = getattr(client, "episodes", None)
    if not callable(finished):
        return None
    try:
        episodes = finished()
    except Exception:  # any client failure means "no document"
        return None
    return episodes[-1] if episodes else None


def _missing_episode(scenario: str, index: int) -> dict[str, Any]:
    """An episode-shaped placeholder for one that produced no metrics document.

    It carries every key of an episode with the ``unknown`` outcome and no
    numbers, so :func:`summarize` counts it in the denominator and never as a
    success - the failure rate a reader sees includes it instead of the sample
    quietly shrinking.
    """
    placeholder = {key: None for key in EPISODE_KEYS}
    placeholder.update(
        {
            "id": f"{scenario}:missing:{index}",
            "scenario": scenario,
            "robot": "",
            "started_at": "",
            "outcome": "unknown",
            "trajectories": {},
        }
    )
    return placeholder


def _episode_line(scenario: str, index: int, total: int, episode: Mapping[str, Any]) -> str:
    """One human line about an episode, for a caller's progress output."""
    outcome = str(episode.get("outcome", "unknown"))
    return (
        f"{scenario} {index}/{total}: {outcome}"
        f" | {_seconds(episode.get('world_seconds'))} s"
        f" | closest {_metres(episode.get('min_human_distance_m'))}"
        f" | path {_metres(episode.get('path_length_m'))}"
    )


def format_summary(summary: Mapping[str, Any]) -> str:
    """One line for a summary document: the numbers a benchmark reports first.

    A campaign prints one of these per scenario and one over the whole run, so a
    reader sees where a policy wins and where it does not without opening the
    JSON. The keys are the ones :func:`~robotsnap.analysis.metrics.summarize`
    writes, and a series the summary does not carry reads ``n/a`` rather than
    being reported as zero.
    """
    episodes = summary.get("episodes", 0)
    success = _percent(summary.get("success_rate"))
    collisions = _percent(summary.get("collision_rate"))
    distance = _metres(_metric_mean(summary, "min_human_distance_m"))
    world = _seconds(_metric_mean(summary, "world_seconds"))
    return (
        f"{episodes} episodes | success {success} | collisions {collisions} | "
        f"min distance {distance} | world time {world}"
    )


def save_campaign(campaign: Campaign, path: str | Path) -> Path:
    """Write ``campaign`` as indented JSON and return the file it wrote."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(campaign.to_document(), indent=2) + "\n", encoding="utf-8")
    return target


def load_campaign(path: str | Path) -> Campaign:
    """Read back a campaign written by :func:`save_campaign`."""
    source = Path(path)
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CampaignError(f"cannot read campaign {source}: {error}") from error
    return Campaign.from_document(document)


def compare_campaigns(a: Campaign, b: Campaign) -> dict[str, Any]:
    """What changed between two campaigns, metric by metric.

    Every rate and the mean of every scalar metric both summaries carry becomes
    ``{"name", "a", "b", "delta", "change"}``: ``delta`` is ``b - a``, and
    ``change`` its fraction of ``a`` - ``None`` when ``a`` is missing or zero, so
    a caller prints "n/a" instead of dividing by zero. A metric only one campaign
    has is named in ``only_in_a`` or ``only_in_b``, never compared against
    nothing.
    """
    a_summary = a.summary
    b_summary = b.summary
    metrics: list[dict[str, Any]] = []

    for rate in _RATE_KEYS:
        metrics.append(_entry(rate, _value(a_summary, rate), _value(b_summary, rate)))

    only_in_a: list[str] = []
    only_in_b: list[str] = []
    for key in _SCALAR_METRICS:
        name = f"{key}.mean"
        a_value = _metric_mean(a_summary, key)
        b_value = _metric_mean(b_summary, key)
        if a_value is None and b_value is None:
            continue
        if a_value is None:
            only_in_b.append(name)
        elif b_value is None:
            only_in_a.append(name)
        else:
            metrics.append(_entry(name, a_value, b_value))

    return {
        "a": a.name,
        "b": b.name,
        "episodes": _entry("episodes", a_summary.get("episodes"), b_summary.get("episodes")),
        "metrics": metrics,
        "only_in_a": only_in_a,
        "only_in_b": only_in_b,
    }


def format_comparison(comparison: Mapping[str, Any]) -> str:
    """Render :func:`compare_campaigns` as an aligned, human-readable table."""
    a_name = str(comparison.get("a", "A"))
    b_name = str(comparison.get("b", "B"))

    rows: list[tuple[str, str, str, str, str]] = []
    episodes = comparison.get("episodes") or {}
    rows.append(
        (
            "episodes",
            _cell(episodes.get("a")),
            _cell(episodes.get("b")),
            _signed(episodes.get("delta")),
            "",
        )
    )
    for entry in comparison.get("metrics", []):
        if not isinstance(entry, Mapping):
            continue
        rows.append(
            (
                str(entry.get("name", "")),
                _cell(entry.get("a")),
                _cell(entry.get("b")),
                _signed(entry.get("delta")),
                _percent(entry.get("change")),
            )
        )

    header = ("metric", a_name, b_name, "delta", "change")
    widths = [max(len(row[column]) for row in (header, *rows)) for column in range(len(header))]

    def line(cells: Sequence[str]) -> str:
        return "  ".join(cell.ljust(widths[column]) for column, cell in enumerate(cells)).rstrip()

    lines = [line(header), "  ".join("-" * width for width in widths)]
    lines.extend(line(row) for row in rows)

    only_a = comparison.get("only_in_a") or []
    only_b = comparison.get("only_in_b") or []
    if only_a:
        lines.append(f"only in {a_name}: " + ", ".join(str(name) for name in only_a))
    if only_b:
        lines.append(f"only in {b_name}: " + ", ".join(str(name) for name in only_b))
    return "\n".join(lines)


def _entry(name: str, a_value: Any, b_value: Any) -> dict[str, Any]:
    """One compared row: the two values, their difference and the fractional change."""
    delta = change = None
    if _is_number(a_value) and _is_number(b_value):
        delta = b_value - a_value
        if a_value != 0:
            change = (b_value - a_value) / a_value
    return {"name": name, "a": a_value, "b": b_value, "delta": delta, "change": change}


def _value(summary: Mapping[str, Any], key: str) -> float | None:
    """A named number of a summary, or ``None`` when it is absent or not a number."""
    value = summary.get(key)
    return float(value) if _is_number(value) else None


def _metric_mean(summary: Mapping[str, Any], key: str) -> float | None:
    """The ``mean`` of a scalar metric's ``{"mean", "min", "max"}`` sub-document."""
    series = summary.get(key)
    return _value(series, "mean") if isinstance(series, Mapping) else None


def _is_number(value: Any) -> bool:
    """Whether ``value`` is a real number; ``bool`` is a flag, not a measurement."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _cell(value: Any) -> str:
    """A table cell for one value: integers as written, floats shortened."""
    if value is None:
        return "n/a"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return f"{value:.4g}" if isinstance(value, float) else str(value)


def _signed(value: Any) -> str:
    """A table cell for one difference, always with its sign."""
    if not _is_number(value):
        return "n/a"
    return f"{value:+d}" if isinstance(value, int) else f"{value:+.4g}"


def _percent(value: Any) -> str:
    """A table cell for one fractional change, as a percentage."""
    return f"{value:+.1%}" if _is_number(value) else "n/a"


def _seconds(value: Any) -> str:
    return f"{value:.1f}" if _is_number(value) else "n/a"


def _metres(value: Any) -> str:
    return f"{value:.2f} m" if _is_number(value) else "n/a"


def _now_iso() -> str:
    """The current UTC instant, ISO-8601 - the stamp a campaign opens and closes on."""
    return datetime.now(timezone.utc).isoformat()
