"""Placing a stream at the instant it was measured, not when it arrived.

The scan is measured at one time and arrives later, so this mixin keeps the
stamped robot poses, holds each scan until the odometry sample that closes its
own interval has arrived, and then interpolates the pose to draw it with. It is
mixed into :class:`robotsnap.viewer.window.Viewer`; it owns no window state of
its own beyond the pose and scan deques the class sets up.
"""

from __future__ import annotations

import math
import time
from collections import deque

from robotsnap.viewer.colours import PENDING_SCANS
from robotsnap.viewer.geometry import _header_stamp


def _scan_hold_seconds():
    """The current :data:`SCAN_HOLD_SECONDS` of the viewer package.

    Read through the package namespace rather than a copied import, so patching
    ``robotsnap.viewer.SCAN_HOLD_SECONDS`` still changes what this reads: the
    value used to be a module global of the single viewer module.
    """
    from robotsnap import viewer

    return viewer.SCAN_HOLD_SECONDS


class _Placement:
    """The pose history and the held scans the window places a stream with."""

    # -- placing a stream at the instant it was measured ---------------------

    def _remember(self, odom, pose) -> None:
        """Keep one stamped pose per odometry message, newest last, without duplicates."""
        if odom is None or pose is None:
            return
        stamp = _header_stamp(odom)
        if stamp is None:
            return
        if self._poses and self._poses[-1][0] == stamp:
            return
        self._poses.append((stamp, pose[0], pose[1], pose[2]))

    def _queue_scan(self, scan) -> None:
        """Hold a scan until the pose of the instant it measured is known.

        The odometry sample that comes *after* a scan's stamp is published at the end of the interval the
        scan was measured in, so it arrives a period late. Waiting for it is the whole difference between
        placing the scan where the robot was and placing it where the robot had already left: a robot
        turning at a radian a second is a tenth of a radian away after one period, which is a third of a
        metre on a wall three metres out.
        """
        if scan is None or scan.stamp is None:
            return
        if self._held and self._held[-1][0].stamp == scan.stamp:
            return
        self._held.append((scan, time.monotonic()))

    def _collect_scans(self, fallback):
        """The ``(scan, pose)`` pairs to draw now, oldest first.

        A scan whose poses are not known yet stays held until they are, or until it has waited
        :data:`SCAN_HOLD_SECONDS`, which is what a stream that stopped publishing odometry costs instead of
        a backlog that never empties.
        """
        now = time.monotonic()
        ready = []
        held: deque = deque(maxlen=PENDING_SCANS)
        for held_scan, arrived in self._held:
            placement = self._scan_pose(held_scan)
            if placement is not None or now - arrived >= _scan_hold_seconds():
                ready.append(
                    (held_scan, placement if placement is not None else fallback)
                )
            else:
                held.append((held_scan, arrived))
        self._held = held
        return ready

    def _scan_pose(self, scan):
        """The pose to place ``scan`` with, or ``None`` while the sample after its stamp is missing."""
        if not self._poses:
            return None
        if scan.stamp <= self._poses[0][0]:
            # Older than every pose kept: the oldest is the closest answer, and none of them comes after.
            return self._poses[0][1:]
        return self._pose_at(scan.stamp)

    def _pose_at(self, stamp):
        """The robot pose at ``stamp``, interpolated between the two poses around it, or ``None``.

        A scan measures the world at one instant and arrives later; a pose read when it arrives is the
        pose of the robot *since* then, so drawing the scan with it turns the whole cloud by however much
        the robot has turned in between - the faster it turns, the further the walls slide. Placing the
        scan at its own stamp removes that. The yaw is interpolated the short way round, so a robot
        crossing +-pi does not send its points the long way.
        """
        if stamp is None or len(self._poses) < 2:
            return None
        history = list(self._poses)
        if stamp <= history[0][0] or stamp >= history[-1][0]:
            # Outside the window: the newest pose is the closest answer, and interpolating past the ends
            # would invent a pose the robot was never in.
            return None
        for index in range(1, len(history)):
            before, after = history[index - 1], history[index]
            if not before[0] <= stamp <= after[0]:
                continue
            span = after[0] - before[0]
            if span <= 0.0:
                return (after[1], after[2], after[3])
            fraction = (stamp - before[0]) / span
            turn = math.atan2(
                math.sin(after[3] - before[3]), math.cos(after[3] - before[3])
            )
            return (
                before[1] + fraction * (after[1] - before[1]),
                before[2] + fraction * (after[2] - before[2]),
                before[3] + fraction * turn,
            )
        return None
