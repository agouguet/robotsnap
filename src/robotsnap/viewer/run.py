"""Live 2D view of a RobotSNAP session, fed by :mod:`robotsnap.client`.

The viewer is a script, not a library: it owns the bridge the way a user script
would, starts the TCP endpoint Unity dials, and redraws the world from the
topics the client has already decoded.

    python -m robotsnap.viewer --port 10000 --rate 30

With ``--ros2`` it reads the same topics off a ROS2 graph instead, which is the
case when the session is already there - because the navigation method under
test runs on it, in a container or not - and ``ros_tcp_endpoint`` is the one
serving Unity. Nothing else changes: the same window draws the same session.

    source /opt/ros/humble/setup.bash
    python -m robotsnap.viewer --ros2

``pygame`` is imported by :func:`main`, never at module import, so importing
this module never drags a display dependency into the package.

Frames: ``/odom``, ``/scan`` and ``/map`` are in the ROS world frame (x
forward, y left, z up), the scan being expressed in the robot frame once the
robot pose is applied, and the crowd is drawn from the ``humans`` of the state
snapshot, which are in that same world frame. A stream that measures the world
at one instant is placed at *that* instant: the scan is turned by the robot pose
interpolated at its own stamp, not by the pose of the moment it arrived. The
sample that closes that interpolation arrives a period after the scan does, so a
scan is held for it rather than placed with the newest pose to hand: a robot
spinning on the spot would otherwise drag the walls round with it by one odometry
period of turn.
Every read tolerates a topic that has not arrived yet: an empty world draws an
empty window and the HUD says what is missing.
"""

from __future__ import annotations

import argparse
import sys
import traceback

from robotsnap.client import RobotSNAPClient
from robotsnap.viewer.colours import DEFAULT_RATE
from robotsnap.viewer.window import Viewer


def main(argv: list[str] | None = None) -> int:
    """Run the viewer until the window is closed, then stop the bridge."""
    args = _parse_args(argv)

    # The session is opened before the window is: a machine that asked for a ROS2 viewer without a
    # ROS2 install has to be told that, and telling it after a pygame banner buries the one line that
    # matters under the one that does not.
    if args.ros2:
        # Loaded here, like pygame: a machine without a ROS2 transport runs every other viewer, and
        # the message it gets when it asks for a ROS2 one says how to install what is missing.
        from robotsnap.bridge.transport import Ros2Unavailable, load_ros2

        try:
            client = load_ros2().Ros2Client()
        except Ros2Unavailable as exc:
            print(f"the ROS2 viewer is unavailable: {exc}", file=sys.stderr)
            return 1
    else:
        try:
            client = RobotSNAPClient(host=args.host, port=args.port)
        except Exception as exc:
            print(f"cannot listen on {args.host}:{args.port}: {exc}", file=sys.stderr)
            return 1

    try:
        import pygame
    except ImportError as exc:
        client.stop()
        print(f"the viewer needs pygame ({exc}): pip install pygame", file=sys.stderr)
        return 1

    try:
        pygame.init()
        viewer = Viewer(pygame, client)
        clock = pygame.time.Clock()
        print(
            f"reading {viewer.session_label} - waiting for Unity "
            "(start the scene; nothing is drawn until the session speaks)"
        )
        # A session that ended underneath the reader - a ROS2 graph that went away - closes the window:
        # there is nothing left to redraw, and waiting for a peer that will never come would only spin.
        while viewer.handle_events() and not getattr(client, "closed", False):
            viewer.update()
            clock.tick(args.rate)
    except KeyboardInterrupt:
        print()
    except Exception as exc:  # pygame.error when there is no display
        print(f"viewer stopped: {exc}", file=sys.stderr)
        # No display is an expected way to stop on a headless host; anything else is a bug, and its
        # traceback is what says where it is.
        if not isinstance(exc, pygame.error):
            traceback.print_exc()
        return 1
    finally:
        client.stop()
        pygame.quit()

    print(f"last error: {client.last_error or 'none'}")
    return 0


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        prog="python -m robotsnap.viewer",
        description="Draw a live 2D view of a RobotSNAP session over the Python bridge.",
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="address to listen on (default: %(default)s)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=10000,
        help="port Unity dials, 0 picks a free one (default: %(default)s)",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=DEFAULT_RATE,
        help="redraws per second (default: %(default)s)",
    )
    parser.add_argument(
        "--ros2",
        action="store_true",
        help=(
            "read the session off a ROS2 graph instead of owning the bridge: for a setup where "
            "ros_tcp_endpoint serves Unity and the navigation methods run on the graph "
            "(needs a sourced ROS2 install)"
        ),
    )
    args = parser.parse_args(argv)
    if args.rate <= 0:
        parser.error("--rate must be greater than zero")
    return args
