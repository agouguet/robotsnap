"""Reading a session's episodes back, and averaging them.

This is the analysis half of the package: it owns the shape of the episode
document a Unity session produces and the arithmetic a benchmarking campaign
runs over a list of them. Nothing here needs Unity, ROS or a socket - the live
stream is read through :class:`~robotsnap.client.RobotSNAPClient`, and the
exports written to disk are read here, from a directory a caller names.

The map:

- :mod:`robotsnap.analysis.metrics` - the episode document, its reader and its
  summary: the outcome, the timings, the social numbers and the trajectories.
- :mod:`robotsnap.analysis.suites` - the scenario suites a campaign names, and
  how a name, a YAML path or a comma-separated list resolves to scenario ids.
- :mod:`robotsnap.analysis.campaign` - running a suite into a benchmark campaign,
  saving it as JSON, and comparing two of them.

Every public name of those modules is re-exported here, so
``from robotsnap.analysis import summarize`` works as well as
``from robotsnap.analysis.metrics import summarize``. Nothing heavy is imported
to make that so: the campaign and suite modules pull ``yaml`` and the
environment in only when they are actually used, so ``import
robotsnap.analysis`` stays free of gymnasium, numpy and torch.
"""

from robotsnap.analysis import metrics
from robotsnap.analysis import campaign, suites
from robotsnap.analysis.campaign import (
    Campaign,
    CampaignError,
    ScenarioResult,
    compare_campaigns,
    format_comparison,
    format_summary,
    load_campaign,
    run_campaign,
    save_campaign,
)
from robotsnap.analysis.metrics import (
    EPISODE_KEYS,
    METRICS_TOPIC,
    NO_HUMAN_DISTANCE,
    OPTIONAL_EPISODE_KEYS,
    OUTCOMES,
    episode_trajectories,
    episodes_from_session,
    is_episode,
    load_export,
    load_trajectories,
    read_catalogue,
    read_index,
    read_session,
    read_sessions,
    session_files,
    summarize,
)
from robotsnap.analysis.suites import (
    DEFAULT_SUITE,
    DEFAULT_SUITES,
    SuitesError,
    load_suite,
    scenarios_for,
    suite_names,
)

__all__ = [
    "metrics",
    "campaign",
    "suites",
    "METRICS_TOPIC",
    "EPISODE_KEYS",
    "OPTIONAL_EPISODE_KEYS",
    "OUTCOMES",
    "NO_HUMAN_DISTANCE",
    "is_episode",
    "episode_trajectories",
    "summarize",
    "read_session",
    "episodes_from_session",
    "read_index",
    "load_export",
    "session_files",
    "read_catalogue",
    "read_sessions",
    "load_trajectories",
    "Campaign",
    "CampaignError",
    "ScenarioResult",
    "run_campaign",
    "save_campaign",
    "load_campaign",
    "compare_campaigns",
    "format_comparison",
    "format_summary",
    "DEFAULT_SUITES",
    "DEFAULT_SUITE",
    "SuitesError",
    "suite_names",
    "scenarios_for",
    "load_suite",
]
