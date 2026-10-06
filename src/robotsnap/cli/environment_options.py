"""The environment surface, the group of options every environment command adds.

A file of its own because it is long: the observation, the sizes of its
components, the controller limits, the reward weights and the limits of an
episode. :func:`_environment_kwargs` turns them into the names
:mod:`robotsnap.runs` expects, and :func:`_session_client` opens the ROS2 client
a run needs when the transport is not TCP.
"""

from __future__ import annotations

import argparse


def _add_environment_options(parser: argparse.ArgumentParser) -> None:
    """The options shared by every sub-command that builds an environment."""
    # Imported here rather than at module import: the two argument types live in
    # ``options``, which re-exports this module, so a top-level import in either
    # direction would be a cycle.
    from robotsnap.cli.options import _json_object, _observation_spec

    group = parser.add_argument_group("environment options")
    group.add_argument(
        "--observations",
        type=_observation_spec,
        default=None,
        help=(
            "observation components, comma-separated or JSON, e.g. 'pose,goal,lidar' "
            "or '{\"lidar\": {\"bins\": 24}}' (default: the task's own observation, "
            "the built-in vector for every command but `goal`)"
        ),
    )
    group.add_argument(
        "--observation-params",
        type=_json_object,
        default=None,
        metavar="JSON",
        help="per-component parameters, e.g. '{\"lidar\": {\"bins\": 24}}'",
    )
    group.add_argument(
        "--observation-structure",
        choices=("flat", "dict"),
        default="flat",
        help="flat array or a dict of named arrays (default: %(default)s)",
    )
    group.add_argument(
        "--max-agents",
        type=int,
        default=8,
        help="neighbours kept in the observation (default: %(default)s)",
    )
    group.add_argument(
        "--lidar-bins",
        type=int,
        default=48,
        help="slices of the laser sweep (default: %(default)s)",
    )
    group.add_argument(
        "--goal-radius",
        type=float,
        default=0.5,
        help="distance that counts as arrival, in m (default: %(default)s)",
    )
    group.add_argument(
        "--collision-distance",
        type=float,
        default=0.25,
        help="lidar range that counts as a crash, in m (default: %(default)s)",
    )
    group.add_argument(
        "--occupancy-threshold",
        type=int,
        default=50,
        help="occupancy value that counts as a wall (default: %(default)s)",
    )
    group.add_argument(
        "--max-linear",
        type=float,
        default=1.0,
        help="linear speed limit of the controller, in m/s (default: %(default)s)",
    )
    group.add_argument(
        "--max-angular",
        type=float,
        default=1.0,
        help="angular speed limit of the controller, in rad/s (default: %(default)s)",
    )
    group.add_argument(
        "--reload-on-reset",
        action="store_true",
        help="reload the scenario on every reset instead of re-applying it",
    )
    group.add_argument(
        "--control-mode",
        default="ros",
        help="control mode of the session (default: %(default)s)",
    )
    group.add_argument(
        "--robot",
        default=None,
        help="id of the robot every command goes to (default: the session's primary robot)",
    )
    group.add_argument(
        "--max-humans",
        type=int,
        default=8,
        help="crowd members kept in the observation (default: %(default)s)",
    )
    group.add_argument(
        "--wait-timeout",
        type=float,
        default=30.0,
        help="seconds to wait for the world before giving up (default: %(default)s)",
    )
    group.add_argument(
        "--goal-reward",
        type=float,
        default=10.0,
        help="reward for reaching the goal (default: %(default)s)",
    )
    group.add_argument(
        "--progress-reward",
        type=float,
        default=1.0,
        help="reward per metre closer to the goal (default: %(default)s)",
    )
    group.add_argument(
        "--collision-penalty",
        type=float,
        default=10.0,
        help="penalty for a collision (default: %(default)s)",
    )
    group.add_argument(
        "--out-of-bounds-penalty",
        type=float,
        default=5.0,
        help="penalty for leaving the map (default: %(default)s)",
    )
    group.add_argument(
        "--time-penalty",
        type=float,
        default=0.1,
        help="penalty per simulated second (default: %(default)s)",
    )


def _environment_kwargs(args: argparse.Namespace) -> dict:
    """The environment options of ``args``, as ``runs`` names them."""
    return dict(
        client=_session_client(args),
        observations=args.observations,
        observation_params=args.observation_params,
        observation_structure=args.observation_structure,
        max_agents=args.max_agents,
        max_humans=args.max_humans,
        lidar_bins=args.lidar_bins,
        wait_timeout=args.wait_timeout,
        goal_radius=args.goal_radius,
        collision_distance=args.collision_distance,
        occupancy_threshold=args.occupancy_threshold,
        max_linear=args.max_linear,
        max_angular=args.max_angular,
        robot=args.robot,
        reload_on_reset=args.reload_on_reset,
        control_mode=args.control_mode,
        goal_reward=args.goal_reward,
        progress_reward=args.progress_reward,
        collision_penalty=args.collision_penalty,
        out_of_bounds_penalty=args.out_of_bounds_penalty,
        time_penalty=args.time_penalty,
    )


def _session_client(args: argparse.Namespace):
    """The client a run is handed, or ``None`` so the environment builds its own.

    ``tcp`` is the default and the environment already knows how to build it: it binds the port and
    waits for Unity. Only a graph is opened here, because the environment would otherwise build a
    socket client - and a run that asked for ROS2 while quietly serving a port is the one failure a
    caller could not see from the outside.
    """
    transport = getattr(args, "transport", "tcp")
    if transport == "tcp":
        return None

    from robotsnap.client import open_client
    from robotsnap.bridge.transport import Ros2Unavailable

    try:
        return open_client(transport=transport, host=args.host, port=args.port)
    except Ros2Unavailable as exc:
        # Same exit as any other command line the run cannot obey, and the same line the reader needs:
        # a shell that never sourced a ROS2 install is what this almost always is.
        raise SystemExit(f"--transport ros2: {exc}") from None
