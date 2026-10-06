"""Stable-Baselines3 algorithms on the RobotSNAP task.

``robotsnap.envs.RobotSNAPEnv`` is already a Gymnasium environment, so a
Stable-Baselines3 algorithm can drive it as it stands; what this module adds is
the two things the algorithms need and the environment does not carry.

The first is a discrete action set. PPO, A2C and SAC take the continuous
``[linear_x, angular_z]`` action of the environment as it is, but DQN and every
other value-based method need a finite set of actions to put a value on, so
:class:`DiscreteCommands` turns an integer into one of a documented handful of
velocity commands - and the wrapper keeps the action space and the command in
one place, so a policy can never be replayed through a different set.

The second is a factory. A run here drives one simulator session, so the
vectorised environment has one worker by default; more workers would mean more
sessions, which is a different thing to ask of a Unity scene. ``total_timesteps``
is therefore in control steps: a run of 600 steps at a control period of two
seconds covers twenty minutes of the world, whatever the time scale is.

Nothing here imports ``stable_baselines3``, ``torch`` or ``numpy`` at module
import: all three live behind the factories below, so ``import robotsnap.rl.sb3``
- and the command line, ``--help`` included - stays usable in an interpreter
that has only the base package.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

__all__ = [
    "ALGORITHMS",
    "DISCRETE_COMMANDS",
    "DiscreteCommands",
    "algorithm_class",
    "check_environment",
    "load_model",
    "make_env",
    "make_vec_env",
    "play",
    "train",
]

#: The names ``--algo`` accepts, and which of them want a discrete action space.
#: ``sac`` and the two policy-gradient methods take the continuous action of the
#: environment unchanged; ``dqn`` only exists with a finite action set.
ALGORITHMS: Mapping[str, bool] = {
    "ppo": False,
    "a2c": False,
    "sac": False,
    "dqn": True,
}

#: The commands a discrete action picks from, as ``(linear_x, angular_z)``.
#: Speed first, then turn, and the zero command is index 0 so a policy that
#: learns to stand still has one integer to pick. The speeds and turn rates are
#: fractions of the environment's own limits, so the same table means the same
#: motion on a robot with different limits.
DISCRETE_COMMANDS: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),
    (1.0, 0.0), (0.5, 0.0), (-0.5, 0.0),
    (0.5, 1.0), (0.5, -1.0),
    (0.0, 1.0), (0.0, -1.0),
    (-0.5, 1.0), (-0.5, -1.0),
)

#: The entry a checkpoint written here carries, naming the algorithm that wrote
#: it. Stable-Baselines3's own archive does not record that - a PPO and an A2C
#: checkpoint name the same policy class - so the only reliable answer for a
#: replay is the one the writer leaves behind.
ALGORITHM_ENTRY = "robotsnap_algorithm"


class DiscreteCommands:
    """Turn an integer into one of a fixed set of ``(linear_x, angular_z)`` commands.

    The fractions in :data:`DISCRETE_COMMANDS` are scaled by the environment's
    own limits, so a table written once gives the same behaviour to a Jackal and
    to a robot whose controller is limited more tightly. Negative linear speeds
    are kept - a policy that has to back away from a pedestrian needs them - and
    they are a minority of the table, because a navigating robot rarely wants
    them.
    """

    def __init__(
        self,
        commands: Sequence[tuple[float, float]] = DISCRETE_COMMANDS,
        *,
        max_linear: float = 1.0,
        max_angular: float = 1.0,
    ):
        if not commands:
            raise ValueError("the command table cannot be empty")
        self.commands = tuple((float(linear), float(angular)) for linear, angular in commands)
        self.max_linear = float(max_linear)
        self.max_angular = float(max_angular)

    def __len__(self) -> int:
        return len(self.commands)

    def command(self, index: Any) -> np.ndarray:
        """The action the environment takes for ``index``, clipped to its limits."""
        import numpy as np

        position = int(np.asarray(index).reshape(-1)[0])
        position = max(0, min(position, len(self.commands) - 1))
        linear, angular = self.commands[position]
        return np.array(
            [
                float(np.clip(linear * self.max_linear, -self.max_linear, self.max_linear)),
                float(np.clip(angular * self.max_angular, -self.max_angular, self.max_angular)),
            ],
            dtype=np.float32,
        )

    def describe(self) -> str:
        """One line naming the commands, for a help message or a log."""
        return ", ".join(
            f"{index}:{linear * self.max_linear:+.2f},{angular * self.max_angular:+.2f}"
            for index, (linear, angular) in enumerate(self.commands)
        )


def _discrete_wrapper_class():
    """The ``gymnasium.ActionWrapper`` for the table above, built on first use."""
    import gymnasium

    class DiscreteActionWrapper(gymnasium.ActionWrapper):
        """An integer action on top of the environment's continuous one.

        The wrapper replaces the action space, not the environment: the policy
        emits an index, the index becomes a command, and the environment still
        sees the ``[linear_x, angular_z]`` it always saw. ``info["action"]``
        therefore keeps reporting the command that was actually applied, which
        is what a log or a replay needs.
        """

        def __init__(self, env, commands: DiscreteCommands):
            super().__init__(env)
            self.commands = commands
            self.action_space = gymnasium.spaces.Discrete(len(commands))

        def action(self, action):
            return self.commands.command(action)

    return DiscreteActionWrapper


def make_env(
    *,
    environment: type | None = None,
    discrete: bool = False,
    seed: int | None = None,
    **options: Any,
):
    """Build one environment, optionally with the discrete action set.

    ``environment`` is the class to build - ``RobotSNAPEnv`` by default, and a
    subclass (a task of your own, or the social task of
    :class:`robotsnap.models.social.SocialNavEnv`) is the reason this is a
    factory rather than a call. ``options`` go to that class unchanged, so
    everything the environment takes is available here.
    """
    if environment is None:
        from robotsnap.envs import RobotSNAPEnv

        environment = RobotSNAPEnv

    env = environment(**options)
    if discrete:
        env = with_discrete_actions(env)
    if seed is not None:
        env.reset(seed=int(seed))
    return env


def with_discrete_actions(env):
    """``env`` with :class:`DiscreteCommands` on top, unless it already has them.

    An environment whose action space is already ``Discrete`` - the social
    navigation environments carry their own table - is left alone: wrapping it
    would put one integer table on top of another and quietly change what an
    action means.
    """
    import gymnasium

    if isinstance(env.action_space, gymnasium.spaces.Discrete):
        return env
    limits = DiscreteCommands(
        max_linear=float(getattr(env, "max_linear", 1.0) or 1.0),
        max_angular=float(getattr(env, "max_angular", 1.0) or 1.0),
    )
    return _discrete_wrapper_class()(env, limits)


def make_vec_env(
    *,
    environment: type | None = None,
    discrete: bool = False,
    seed: int | None = None,
    workers: int = 1,
    **options: Any,
):
    """One or more environments for ``model.learn``, wrapped the way SB3 wants.

    ``workers`` above one builds several environments over the *same* session,
    which is only meaningful once more than one robot is driven at a time; the
    default of one is what a single robot against the crowd wants.
    """
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    def build(index: int):
        def factory():
            return make_env(
                environment=environment,
                discrete=discrete,
                seed=None if seed is None else seed + index,
                **options,
            )

        return factory

    factories = [build(index) for index in range(max(1, int(workers)))]
    if len(factories) == 1:
        return DummyVecEnv(factories)
    return SubprocVecEnv(factories)


def algorithm_class(name: str):
    """The Stable-Baselines3 class for ``name``, imported on first use."""
    key = str(name).strip().lower()
    if key not in ALGORITHMS:
        raise ValueError(
            f"algorithm is one of {', '.join(sorted(ALGORITHMS))}, got {name!r}"
        )
    try:
        from stable_baselines3 import A2C, DQN, PPO, SAC
    except ImportError as error:  # pragma: no cover - depends on the install
        raise RuntimeError(
            "the algorithms need stable-baselines3: pip install -e '.[sb3]'"
        ) from error
    return {"ppo": PPO, "a2c": A2C, "sac": SAC, "dqn": DQN}[key]


def check_environment(env) -> None:
    """Run SB3's own contract check, so a mismatch is reported where it happens.

    ``check_env`` exercises ``reset``/``step`` against the declared spaces; it
    is the cheapest way to be told that an observation is the wrong dtype or an
    action falls outside its space, rather than seeing it as a training run that
    quietly learns nothing.
    """
    from stable_baselines3.common.env_checker import check_env as sb3_check_env

    sb3_check_env(env, warn=True, skip_render_check=True)


def train(
    *,
    algo: str = "ppo",
    timesteps: int = 20_000,
    save: str | Path | None = None,
    seed: int | None = None,
    environment: type | None = None,
    env: Any = None,
    workers: int = 1,
    log_interval: int = 1,
    progress: bool = False,
    policy_kwargs: Mapping[str, Any] | None = None,
    algo_kwargs: Mapping[str, Any] | None = None,
    callback: Any = None,
    **options: Any,
) -> Any:
    """Train ``algo`` on the task and return the model.

    ``options`` are the environment's own (``scenario``, ``control_period``,
    ``pacing``, ``time_scale``, ``observations``, ``seconds``...), so the
    command line passes them through unchanged. The learning rate, the network
    width and the rest are Stable-Baselines3's own keyword arguments and are
    handed over as ``algo_kwargs``.

    An already-built ``env`` is used as it stands - a run that owns the session
    wants to hand over the one environment it will close itself, rather than
    have a second one built over the same simulator.
    """
    key = str(algo).strip().lower()
    discrete = ALGORITHMS.get(key)
    if discrete is None:
        raise ValueError(
            f"algorithm is one of {', '.join(sorted(ALGORITHMS))}, got {algo!r}"
        )

    model_class = algorithm_class(key)
    if env is None:
        env = make_vec_env(
            environment=environment,
            discrete=bool(discrete),
            seed=seed,
            workers=workers,
            **options,
        )
    elif discrete:
        env = with_discrete_actions(env)

    model = model_class(
        "MlpPolicy",
        env,
        seed=seed,
        verbose=1,
        policy_kwargs=dict(policy_kwargs) if policy_kwargs else None,
        **dict(algo_kwargs or {}),
    )
    model.learn(
        total_timesteps=int(timesteps),
        log_interval=log_interval,
        progress_bar=progress,
        callback=callback,
    )
    if save is not None:
        path = Path(save)
        # A checkpoint is a zip: a name without a suffix gets one, and a name
        # that already carries another one is left alone rather than renamed
        # behind the caller's back.
        if not path.suffix:
            path = path.with_suffix(".zip")
        model.save(str(path))
        stamp_algorithm(path, key)
    return model


def stamp_algorithm(path: str | Path, algo: str) -> None:
    """Write the algorithm's name into the checkpoint, beside SB3's own files."""
    import zipfile

    with zipfile.ZipFile(path, "a", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(ALGORITHM_ENTRY, str(algo).strip().lower())


def load_model(path: str | Path, env=None, algo: str | None = None):
    """Read a checkpoint written by :func:`train` - or by ``model.save``."""
    key = _algorithm_from_checkpoint(path, algo=algo)
    model_class = algorithm_class(key)
    return model_class.load(str(path), env=env)


def _algorithm_from_checkpoint(path: str | Path, *, algo: str | None = None) -> str:
    """The algorithm a ``.zip`` checkpoint was written by.

    The marker :func:`stamp_algorithm` leaves is the answer for anything this
    package wrote. A checkpoint from elsewhere is read through Stable-
    Baselines3's own loader and judged on the policy class it names, which
    identifies the value-based algorithms and SAC; a plain actor-critic policy
    belongs to both PPO and A2C, so that case is refused rather than guessed -
    loading a policy into the wrong algorithm gives wrong shapes and a silent
    non-answer, which is worse than being asked.
    """
    import zipfile

    if algo is not None and str(algo).strip().lower() != "auto":
        key = str(algo).strip().lower()
        if key not in ALGORITHMS:
            raise ValueError(
                f"algorithm is one of {', '.join(sorted(ALGORITHMS))}, got {algo!r}"
            )
        return key

    try:
        with zipfile.ZipFile(path) as archive:
            if ALGORITHM_ENTRY in archive.namelist():
                written = archive.read(ALGORITHM_ENTRY).decode("utf-8", "replace").strip().lower()
                if written in ALGORITHMS:
                    return written
            names = archive.namelist()
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise ValueError(
            f"{path} is not a readable Stable-Baselines3 checkpoint: {error}"
        ) from error

    if "data" not in names:
        raise ValueError(f"{path} is not a Stable-Baselines3 checkpoint: it carries no 'data'")

    policy_name = _policy_class_name(path)
    for candidate, suffix in (("dqn", "DQNPolicy"), ("sac", "SACPolicy")):
        if policy_name.endswith(suffix):
            return candidate
    raise ValueError(
        f"{path} was not written by this package and names the policy {policy_name or 'nothing'}, "
        "which more than one algorithm uses: pass --algo to say which one wrote it"
    )


def _policy_class_name(path: str | Path) -> str:
    """The policy class a checkpoint names, read through Stable-Baselines3's loader."""
    try:
        from stable_baselines3.common.save_util import load_from_zip_file

        data, _, _ = load_from_zip_file(str(path), load_data=True)
    except Exception as error:  # pragma: no cover - a broken archive of another shape
        raise ValueError(f"{path} cannot be read as a checkpoint: {error}") from error
    if not isinstance(data, Mapping):
        return ""
    return str(getattr(data.get("policy_class"), "__name__", "") or "")


def play(
    path: str | Path,
    env=None,
    *,
    episodes: int = 1,
    deterministic: bool = True,
    algo: str | None = None,
    on_step: Callable[[int, float], None] | None = None,
    **options: Any,
):
    """Run a saved policy, without training, and report what the episodes did.

    Returns one dict per episode: the reward the policy collected, the world
    seconds it took, the number of steps, and how the episode ended.
    """
    key = _algorithm_from_checkpoint(path, algo=algo)
    discrete = bool(ALGORITHMS[key])
    owned = env is None
    if env is None:
        env = make_env(discrete=discrete, **options)
    elif discrete:
        env = with_discrete_actions(env)
    model = load_model(path, env=env, algo=key)

    history = []
    try:
        for index in range(max(1, int(episodes))):
            observation, _ = env.reset()
            total = 0.0
            steps = 0
            while True:
                action, _ = model.predict(observation, deterministic=deterministic)
                observation, reward, terminated, truncated, info = env.step(action)
                total += float(reward)
                steps += 1
                if on_step is not None:
                    on_step(steps, total)
                if terminated or truncated:
                    break
            history.append(
                {
                    "episode": index,
                    "reward": total,
                    "steps": steps,
                    "seconds": float(info.get("episode_seconds") or 0.0),
                    "wall_seconds": float(info.get("wall_seconds") or 0.0),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "stalled": bool(info.get("stalled")),
                }
            )
    finally:
        if owned:
            env.close()
    return history
