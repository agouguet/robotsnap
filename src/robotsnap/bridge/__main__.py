"""Diagnostic runner for the RobotSNAP Unity bridge.

``python -m robotsnap.bridge --host 0.0.0.0 --port 10000 --rate 2`` starts the
bridge, waits for Unity and then prints one compact summary line per tick so a
human can compare the Python view with the Unity display. Ctrl+C stops cleanly.

``--ros2`` runs the same bridge with a ROS2 mirror beside it: Unity still dials
this process, and every stream the session publishes is put on a ROS2 graph for
a navigation method to read, while whatever the graph publishes on the session's
command topics is written back to Unity. That is the job ``ros_tcp_endpoint``
does, done by this package, and it is what lets a ROS2 node drive the session
while the Gymnasium environment and the viewer keep reading the same bridge.

``--viewer`` replaces the lines with the package's own window on the same session, so the bridge can
be the only thing a user runs while watching Unity.

No ROS, matplotlib or pygame is needed for the printed modes: this tool is a read-only window on the
messages Unity sends, the mirror is only asked for with ``--ros2``, and pygame only with ``--viewer``.
"""

from __future__ import annotations

import argparse
import sys
import time

from robotsnap.bridge.server import RobotSNAPBridge


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        prog="python -m robotsnap.bridge",
        description="Run the RobotSNAP Unity bridge and print the incoming state.",
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="address to listen on (default: %(default)s)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=10000,
        help="port to listen on, 0 picks a free one (default: %(default)s)",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=2.0,
        help="summary lines per second (default: %(default)s)",
    )
    parser.add_argument(
        "--ros2",
        action="store_true",
        help=(
            "mirror the session on a ROS2 graph as well: a navigation method that is a ROS2 node "
            "sees Unity's streams and drives /cmd_vel without ros_tcp_endpoint running"
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
            "draw the session in a window instead of printing a line per tick: the same view the "
            "environment and the viewer command draw, on the session this bridge already serves "
            "(needs pygame)"
        ),
    )
    args = parser.parse_args(argv)
    if args.rate <= 0:
        parser.error("--rate must be greater than zero")
    return args


def _summary(bridge) -> str:
    snapshot = bridge.snapshot()
    if snapshot.robot is None:
        pose = "robot=---"
    else:
        robot = snapshot.robot
        pose = f"robot x={robot.x:+.2f} z={robot.z:+.2f} yaw={robot.yaw:+.2f}"
    if snapshot.laser is None:
        laser = "---"
    else:
        laser = str(len(snapshot.laser.points_2d()))
    state = "connected" if bridge.is_connected else "waiting"
    map_seen = "yes" if snapshot.map_received else "no"
    scenario = snapshot.simulation.scenario_id if snapshot.simulation is not None else None
    return (
        f"{state} | {pose} | agents={len(snapshot.agents)} | laser_pts={laser} | "
        f"map={map_seen} | humans={len(snapshot.humans)} | scenario={scenario or '---'}"
    )


def _print_ticks(bridge, gateway, *, rate: float) -> None:
    """Print one compact line per tick until Ctrl+C, naming each topic the first time it speaks."""
    period = 1.0 / rate
    seen: set[str] = set()
    next_tick = time.monotonic()
    while True:
        types = bridge.topic_types()
        for topic in sorted(bridge.topic_counts()):
            if topic not in seen:
                seen.add(topic)
                print(f"new topic {topic} ({types.get(topic, '?')})")
        print(_summary(bridge))
        if gateway is not None and gateway.last_error:
            print(f"mirror: {gateway.last_error}")
        next_tick += period
        delay = next_tick - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        else:
            next_tick = time.monotonic()


def _draw_ticks(bridge, *, rate: float) -> None:
    """Draw the session this bridge serves in a window until it is closed.

    The window is the one the rest of the package draws - the very class an environment uses for
    ``--viewer`` - so a bridge in one terminal and a training run in another show the same picture of
    the same session. It opens no session of its own: the bridge is already the one Unity dials, and
    the client wrapped around it only reads. ``rate`` redraws per second here, where the printed mode
    counts lines.

    Imported inside the function on purpose: a machine without pygame runs every other mode of this
    runner, and only a caller that asked for the window is told it needs one.
    """
    try:
        import pygame
    except ImportError as exc:
        print(f"--viewer needs pygame ({exc}): pip install pygame", file=sys.stderr)
        return

    from robotsnap.client import RobotSNAPClient
    from robotsnap.viewer import Viewer

    client = RobotSNAPClient(bridge=bridge)
    pygame.init()
    try:
        viewer = Viewer(pygame, client)
        clock = pygame.time.Clock()
        while viewer.handle_events():
            viewer.update()
            clock.tick(rate)
    except KeyboardInterrupt:
        print()
    finally:
        pygame.quit()


def main(argv=None) -> int:
    args = _parse_args(argv)
    bridge = RobotSNAPBridge(host=args.host, port=args.port)
    try:
        bridge.start()
    except Exception as exc:
        print(f"failed to start the bridge: {exc}", file=sys.stderr)
        return 1

    gateway = None
    if args.ros2:
        # The transport is loaded through the entry point: the core knows the name ``ros2``, not the
        # code behind it. A missing plugin is a clean refusal whose line already names the install.
        from robotsnap.bridge.transport import TransportUnavailable, load_ros2

        try:
            module = load_ros2()
        except TransportUnavailable as exc:
            print(f"--ros2 needs a ROS2 install: {exc}", file=sys.stderr)
            bridge.stop()
            return 1

        reason = module.unavailable_reason()
        if reason is not None:
            print(f"--ros2 needs a ROS2 install: {reason}", file=sys.stderr)
            bridge.stop()
            return 1
        try:
            gateway = module.Ros2Gateway(bridge, node_name=args.ros2_node_name)
            gateway.start()
        except Exception as exc:
            print(f"failed to start the ROS2 mirror: {exc}", file=sys.stderr)
            bridge.stop()
            return 1

    print(f"listening on {args.host}:{bridge.port}")
    if gateway is not None:
        print("mirroring the session on a ROS2 graph")
    try:
        print("waiting for Unity to connect...")
        bridge.wait_for_connection()
        if args.viewer:
            _draw_ticks(bridge, rate=args.rate)
        else:
            _print_ticks(bridge, gateway, rate=args.rate)
    except KeyboardInterrupt:
        print()
    finally:
        if gateway is not None:
            gateway.stop()
        bridge.stop()

    counts = bridge.topic_counts()
    print("recap:")
    if counts:
        for topic in sorted(counts):
            print(f"  {topic}: {counts[topic]} message(s)")
    else:
        print("  no topics seen")
    if gateway is not None:
        stats = gateway.stats()
        print(
            "  ROS2 mirror: "
            f"{stats['published']} published, {stats['received']} received, "
            f"{stats['skipped']} topic(s) skipped"
        )
        for topic, reason in sorted(gateway.skipped().items()):
            print(f"    {topic}: {reason}")
    print(f"  last error: {bridge.last_error or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
