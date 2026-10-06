"""The ``bridge`` command: serve Unity, printing the state, optionally mirrored on ROS2.

``robotsnap bridge`` and ``python -m robotsnap.bridge`` are one runner, so the
single entry point covers every run of the package, the bridge included.
Whether a ROS2 mirror stands beside the session is the bridge's own option, and
neither the environment nor the viewer hears about it - the ``--viewer`` window
here is the bridge's own way to draw the session it already serves.
"""

from __future__ import annotations

import argparse

NAME = "bridge"
ALIASES = ()
HELP = "serve Unity and print the incoming state, optionally mirrored on ROS2"
DESCRIPTION = (
    "Own the port Unity dials and print a summary line per tick. With --ros2 the same "
    "session is mirrored on a ROS2 graph, so a navigation method that is a ROS2 node sees "
    "Unity's streams and drives the robot without ros_tcp_endpoint running."
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
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
        "--rate", type=float, default=2.0, help="summary lines per second (default: %(default)s)"
    )
    parser.add_argument(
        "--ros2",
        action="store_true",
        help=(
            "mirror the session on a ROS2 graph: a ROS2 navigation method reads the streams and "
            "writes /cmd_vel, while this bridge keeps serving Unity"
        ),
    )
    parser.add_argument(
        "--ros2-node-name",
        default="robotsnap_gateway",
        help="name the mirror's ROS2 node answers to (default: %(default)s)",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help=(
            "draw the session in the package's 2D window instead of printing a line per "
            "tick, on the bridge this command already serves"
        ),
    )


def run(args: argparse.Namespace) -> int:
    from robotsnap.bridge.__main__ import main as bridge_main

    forwarded = ["--host", args.host, "--port", str(args.port), "--rate", str(args.rate)]
    if args.ros2:
        forwarded.extend(["--ros2", "--ros2-node-name", args.ros2_node_name])
    if args.viewer:
        forwarded.append("--viewer")
    return bridge_main(forwarded)
