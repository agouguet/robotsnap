"""How the core reaches a transport it does not ship, by name.

The core does not know ROS2: it knows a group of entry points (``robotsnap.transports``) and asks
for one by name. A package that registers itself there - ``robotsnap-ros2`` answers ``ros2`` - is
loaded on demand; a name nobody registered is refused with the install line that provides it, so
the core stays importable and usable on a machine that never installs a transport.
"""

from __future__ import annotations

import importlib.metadata

__all__ = [
    "ENTRY_POINT_GROUP",
    "Ros2Unavailable",
    "TransportUnavailable",
    "load_ros2",
    "load_transport",
]

#: The entry-point group a transport registers itself in. The core knows this name and no other.
ENTRY_POINT_GROUP = "robotsnap.transports"

#: The name the ROS2 transport registers under, and the name the callers that catch a missing ROS2
#: install ask for.
_ROS2 = "ros2"


class TransportUnavailable(RuntimeError):
    """Raised when a transport the caller asked for is not installed."""


class Ros2Unavailable(TransportUnavailable):
    """Raised when a ROS2 session was asked for and no ROS2 transport is installed.

    A subclass of :class:`TransportUnavailable`, so a caller that catches the general failure and
    one that catches the ROS2 one both hear the same refusal.
    """


def load_transport(name: str):
    """The module registered as ``name`` in ``robotsnap.transports``, or a :class:`TransportUnavailable`.

    The transport's own code is imported here and nowhere else, so a run that never asks for it
    never pays for it: the core imports this function, not ``rclpy``.
    """
    entries = importlib.metadata.entry_points(group=ENTRY_POINT_GROUP, name=name)
    entry = next(iter(entries), None)
    if entry is None:
        raise TransportUnavailable(
            f"the transport {name!r} is not installed: pip install robotsnap-{name}"
        )
    return entry.load()


def load_ros2():
    """``load_transport("ros2")``, with the name the ROS2 callers of the core catch.

    The CLI and the viewer catch :class:`Ros2Unavailable`; naming it here keeps that catch working
    whether the transport is missing entirely or is a ROS2 install this interpreter cannot use.
    """
    try:
        return load_transport(_ROS2)
    except TransportUnavailable as exc:
        raise Ros2Unavailable(str(exc)) from exc
