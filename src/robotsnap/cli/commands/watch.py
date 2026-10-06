"""The ``watch`` command: draw a live 2D view of a session.

It owns the bridge Unity dials, or reads a ROS2 graph with ``--ros2``. It is
named ``watch`` because it drives nothing; ``viewer`` is kept as an alias so a
command line written before the rename keeps working.
"""

from __future__ import annotations

import argparse

NAME = "watch"
ALIASES = ("viewer",)
HELP = "draw a live 2D view of a session"
DESCRIPTION = "Draw a live 2D view of a RobotSNAP session over the Python bridge."


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """The bridge the viewer opens, and the two ways it can read a session."""
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
        "--rate", type=float, default=30.0, help="redraws per second (default: %(default)s)"
    )
    parser.add_argument(
        "--ros2",
        action="store_true",
        help=(
            "read the session off a ROS2 graph instead of owning the bridge, for a setup where "
            "ros_tcp_endpoint serves Unity and the navigation methods run on the graph"
        ),
    )


def run(args: argparse.Namespace) -> int:
    """Hand the viewer its own arguments; it owns the window and the session."""
    from robotsnap.viewer import main as viewer_main

    forwarded = ["--host", args.host, "--port", str(args.port), "--rate", str(args.rate)]
    if args.ros2:
        forwarded.append("--ros2")
    return viewer_main(forwarded)
