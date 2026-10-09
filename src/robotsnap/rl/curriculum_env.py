"""A Gymnasium wrapper that applies a curriculum to a RobotSNAP environment.

The environment itself stays ignorant of curricula: this wrapper, which may
import ``gymnasium`` at module level, is the one place a
:class:`~robotsnap.rl.curriculum.Curriculum` meets a live session. On every
reset it hands the environment the current stage's options - above all the
scenario to launch - and on every finished episode it tells the curriculum
whether that episode succeeded, so the ladder can move itself on.

The success of an episode is read out of ``info`` defensively. The simulator's
own ``RobotSNAPEnv`` reports ``goal_reached``, but the same task exists under a
handful of names across builds and forks, so several plausible keys are tried
before the fallback: an episode that ended on ``terminated`` with no recorded
failure has nothing left to have ended on but the goal.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import gymnasium

from robotsnap.rl.curriculum import Curriculum, Stage

__all__ = ["CurriculumEnv", "episode_succeeded"]

#: Keys whose value, when the episode ends, decides success on its own.
_SUCCESS_KEYS = ("success", "goal_reached", "reached_goal", "goal")

#: Keys whose truth means the episode did *not* succeed, read only by the
#: fallback that otherwise trusts ``terminated``.
_FAILURE_KEYS = ("collision", "out_of_bounds", "timeout", "stalled")

#: Words an ``outcome``-style status is compared against, lower-cased.
_SUCCESS_WORDS = frozenset({"1", "true", "yes", "goal", "success", "reached"})


def episode_succeeded(
    info: Mapping[str, Any] | None, *, terminated: bool = False
) -> bool:
    """Whether an episode that just ended counts as a success.

    A boolean under any of :data:`_SUCCESS_KEYS` decides outright; ``outcome``
    is read as a word. When none is present, ``terminated`` without a recorded
    failure is taken as success, since the goal is the only ending left.
    """
    data = info if isinstance(info, Mapping) else {}
    for key in _SUCCESS_KEYS:
        decided = _as_bool(data.get(key))
        if decided is not None:
            return decided
    outcome = data.get("outcome")
    if isinstance(outcome, str):
        return outcome.strip().lower() in _SUCCESS_WORDS
    if not terminated:
        return False
    for key in _FAILURE_KEYS:
        if _as_bool(data.get(key)) is True:
            return False
    return True


def _as_bool(value: Any) -> bool | None:
    """A boolean reading of a value, or ``None`` when it does not say one.

    Arrays and other objects are left alone rather than coerced, so a ``goal``
    key that happens to hold a position is skipped instead of being asked for a
    truth value it cannot give.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in _SUCCESS_WORDS
    return None


class CurriculumEnv(gymnasium.Wrapper):
    """Wrap a RobotSNAP environment so a curriculum drives its resets.

    ``options`` handed to :meth:`reset` are merged *over* the stage's own, so a
    caller can still override one field for a special episode without giving up
    the scenario the stage asked for.
    """

    def __init__(self, env: Any, curriculum: Curriculum) -> None:
        super().__init__(env)
        if not isinstance(curriculum, Curriculum):
            raise TypeError("curriculum must be a Curriculum")
        self.curriculum = curriculum

    def __getattr__(self, name: str) -> Any:
        """Forward what the wrapper does not own to the environment underneath.

        A run keeps reading the session off the environment it was handed -
        ``client`` to stop it at the end, the observation names and the reward
        weights to announce what is being trained - and a curriculum sits
        between the two. Without this, wrapping hides them: the stop raises
        ``AttributeError`` and the announcement quietly loses its details.

        A name starting with an underscore is the wrapper's own business and is
        never forwarded, which is also what keeps ``__getattr__`` from
        recursing into itself through ``self.env``.
        """
        if name.startswith("_"):
            raise AttributeError(name)
        env = self.__dict__.get("env")
        if env is None:
            raise AttributeError(name)
        return getattr(env, name)

    @property
    def curriculum_state(self) -> dict[str, Any]:
        """The current stage's counters, for a logger or a callback."""
        return self.curriculum.state()

    def reset(
        self, *, seed: int | None = None, options: Mapping[str, Any] | None = None
    ) -> Any:
        """Reset onto the current stage, scenario and options included."""
        merged = self.curriculum.reset_options()
        merged.update(dict(options or {}))
        self._apply_scenario_fields(self.curriculum.current)
        return self.env.reset(seed=seed, options=merged)

    def step(self, action: Any) -> Any:
        """Step the environment, then let the curriculum read a finished one."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            self.curriculum.observe(
                success=episode_succeeded(info, terminated=terminated), info=info
            )
        return observation, reward, terminated, truncated, info

    def _apply_scenario_fields(self, stage: Stage) -> None:
        """Copy a stage's scenario fields onto the environment before a reset.

        The session writes a scenario's fields to disk once and then trusts the
        cache; clearing that latch is what makes a stage that rewrites the same
        scenario's fields heard. Environments without the attribute are left
        alone - a wrapper around a plain Gymnasium env has nothing to rewrite.
        """
        if stage.scenario_fields is None or not hasattr(self.env, "scenario_fields"):
            return
        self.env.scenario_fields = dict(stage.scenario_fields)
        if hasattr(self.env, "_scenario_written"):
            self.env._scenario_written = False
