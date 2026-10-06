"""The learned policy, its running normaliser, and the checkpoint they travel in.

``robotsnap.runs`` trains one of these and hands it to a file; ``robotsnap.rl.policy``
is the one place that knows the shape of the network, so a run that only loads a
checkpoint does not have to guess it. A checkpoint carries the weights *and* the
statistics the observations were normalised by - without the second, the network
would be replayed on observations at a different scale than the ones it learned
on.

Nothing here imports ``torch`` or ``numpy`` at module import: both live behind
the factories below, so ``import robotsnap.rl.policy`` - and the command line,
``--help`` included - stays usable in an interpreter that has only the base
package.
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = [
    "FORMAT",
    "policy_class",
    "running_norm_class",
    "save_policy",
    "load_policy",
]

#: The tag a checkpoint carries, so a reader knows at once what it is holding.
FORMAT = "robotsnap-policy-1"

_POLICY = None
_NORM = None


def policy_class():
    """The Gaussian policy class, built once on the first call.

    A ``torch.nn.Module`` subclass, so this is the call that pulls PyTorch in;
    ``import robotsnap.rl.policy`` on its own does not.
    """
    global _POLICY
    if _POLICY is None:
        import torch
        import torch.nn as nn

        class Policy(nn.Module):
            """A Gaussian policy: an MLP for the mean, a learned spread per action."""

            def __init__(self, observations: int, actions: int = 2, hidden: int = 64):
                super().__init__()
                self.body = nn.Sequential(
                    nn.Linear(observations, hidden),
                    nn.Tanh(),
                    nn.Linear(hidden, hidden),
                    nn.Tanh(),
                )
                self.mean = nn.Linear(hidden, actions)
                self.log_std = nn.Parameter(torch.full((actions,), -0.5))

            def forward(self, observation):
                features = self.body(observation)
                # The mean is a tanh: the action space is a box of the controller's
                # own limits, so the network cannot ask for more than the robot can do.
                return (
                    torch.tanh(self.mean(features)),
                    self.log_std.clamp(-3.0, 0.5).exp(),
                )

            def distribution(self, observation):
                mean, std = self.forward(observation)
                return torch.distributions.Normal(mean, std)

        _POLICY = Policy
    return _POLICY


def running_norm_class():
    """The running normaliser class, built once on the first call.

    Only ``numpy`` is needed, but it is deferred here too so the module stays
    importable without the environment extra.
    """
    global _NORM
    if _NORM is None:
        import numpy as np

        class RunningNorm:
            """A running mean and variance, so a network sees observations near unit scale."""

            def __init__(self, size: int):
                self.mean = np.zeros(size, dtype=np.float64)
                self.square = np.zeros(size, dtype=np.float64)
                self.count = 1e-4

            def __call__(self, observation):
                value = np.asarray(observation, dtype=np.float64)
                self.count += 1.0
                delta = value - self.mean
                self.mean += delta / self.count
                self.square += delta * (value - self.mean)
                std = np.sqrt(np.maximum(self.square / self.count, 1e-8))
                return ((value - self.mean) / std).astype(np.float32)

            def normalise(self, observation):
                """Apply the statistics already collected, without moving them.

                Training updates the statistics on every observation it sees;
                an episode played back from a checkpoint must not, or the
                normalisation would drift away from the one the network learned
                on.
                """
                value = np.asarray(observation, dtype=np.float64)
                std = np.sqrt(np.maximum(self.square / self.count, 1e-8))
                return ((value - self.mean) / std).astype(np.float32)

        _NORM = RunningNorm
    return _NORM


def save_policy(
    path,
    policy,
    norm,
    *,
    observation_size,
    hidden,
    observation_names: Sequence[str] = (),
    actions: int = 2,
):
    """Write ``policy`` and ``norm`` to ``path``, with the shape to rebuild them.

    The state dict alone is not enough to use a policy later: it does not say
    how wide the network is, and it does not carry the statistics the
    observations were normalised by. Both travel here, so :func:`load_policy`
    rebuilds the pair without being told anything.
    """
    import torch

    document = {
        "format": FORMAT,
        "state_dict": policy.state_dict(),
        "observation_size": int(observation_size),
        "observation_names": [str(name) for name in observation_names],
        "hidden": int(hidden),
        "actions": int(actions),
        "normaliser": {
            "mean": [float(value) for value in norm.mean],
            "square": [float(value) for value in norm.square],
            "count": float(norm.count),
        },
    }
    torch.save(document, path)
    return path


def load_policy(path, *, observation_size=None, hidden=None, device="cpu"):
    """Rebuild ``(policy, normaliser, meta)`` from the checkpoint at ``path``.

    A checkpoint written by :func:`save_policy` carries everything needed. A
    bare ``state_dict`` - what a training run used to write - does not, so it
    needs ``observation_size`` and ``hidden``, and the normaliser comes back as
    the identity. ``meta`` always says which case it was: ``"legacy"`` is true
    when nothing could be recovered beyond the weights, and ``"normaliser"`` is
    ``None`` when no statistics were stored.
    """
    import torch

    document = torch.load(path, map_location=device)

    if isinstance(document, dict) and document.get("format") == FORMAT:
        return _load_document(document, observation_size, hidden, device)
    return _load_legacy(document, observation_size, hidden, device)


def _check_shape(size, depth, observation_size, hidden):
    """The shape to build with, refusing a hint that contradicts the file."""
    if observation_size is not None and int(observation_size) != size:
        raise ValueError(
            f"observation_size {int(observation_size)} does not match the "
            f"checkpoint's {size}"
        )
    if hidden is not None and int(hidden) != depth:
        raise ValueError(
            f"hidden {int(hidden)} does not match the checkpoint's {depth}"
        )
    return size, depth


def _load_document(document, observation_size, hidden, device):
    """Rebuild a checkpoint that carries its own metadata."""
    import numpy as np

    size = int(document["observation_size"])
    depth = int(document["hidden"])
    actions = int(document.get("actions", 2))
    size, depth = _check_shape(size, depth, observation_size, hidden)

    policy = policy_class()(size, actions, depth).to(device)
    policy.load_state_dict(document["state_dict"])
    policy.eval()

    norm = running_norm_class()(size)
    normaliser = document.get("normaliser")
    if normaliser is not None:
        norm.mean = np.asarray(normaliser["mean"], dtype=np.float64)
        norm.square = np.asarray(normaliser["square"], dtype=np.float64)
        norm.count = float(normaliser["count"])

    meta = {
        "format": document.get("format"),
        "observation_size": size,
        "observation_names": [
            str(name) for name in document.get("observation_names", ())
        ],
        "hidden": depth,
        "actions": actions,
        "legacy": False,
        "normaliser": normaliser,
    }
    return policy, norm, meta


def _load_legacy(state_dict, observation_size, hidden, device):
    """Rebuild a bare state dict, which needs its shape spelled out."""
    import numpy as np

    if observation_size is None or hidden is None:
        raise ValueError(
            "this checkpoint is a bare state_dict, without the metadata a "
            "RobotSNAP checkpoint carries: pass observation_size and hidden to "
            "load_policy() so the network can be rebuilt. No normalisation was "
            "saved either, so the observations are fed to it raw."
        )

    size = int(observation_size)
    depth = int(hidden)
    actions = _infer_actions(state_dict)

    policy = policy_class()(size, actions, depth).to(device)
    policy.load_state_dict(state_dict)
    policy.eval()

    # A fresh running normaliser scales by 1/count, which is not the identity:
    # for a legacy checkpoint the statistics were never saved, so the honest
    # normaliser is the one that leaves the observation alone.
    norm = running_norm_class()(size)
    norm.mean = np.zeros(size, dtype=np.float64)
    norm.square = np.full(size, 1.0, dtype=np.float64)
    norm.count = 1.0

    meta = {
        "format": None,
        "observation_size": size,
        "observation_names": [],
        "hidden": depth,
        "actions": actions,
        "legacy": True,
        "normaliser": None,
    }
    return policy, norm, meta


def _infer_actions(state_dict) -> int:
    """The number of actions the state dict's output layer was built for."""
    weight = state_dict.get("mean.weight") if hasattr(state_dict, "get") else None
    if weight is not None and getattr(weight, "shape", None):
        return int(weight.shape[0])
    return 2
