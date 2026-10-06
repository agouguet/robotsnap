"""Turning published data into pixels and text: the grid and the HUD.

Two jobs live here because both are pure functions of a message. The occupancy
grid is packed into one RGB surface instead of a rectangle per cell, and the
status band is rendered as three short lines. Neither needs the window, only
the message and the palette.
"""

from __future__ import annotations

import math

from robotsnap.topics import MAP, base
from robotsnap.viewer.colours import MAP_FREE, MAP_UNKNOWN, MAP_WALL
from robotsnap.viewer.geometry import _as_float, _map_bounds


def _map_count(client) -> int:
    """How many ``/map`` messages the bridge has decoded, for the draw cache."""
    return int(client.bridge.topic_counts().get(base(MAP), 0))


def _map_image(pygame, message, rect):
    """Render a grid as one RGB surface scaled to ``rect``.

    One rectangle per cell would be far too slow on a real grid, so the cells
    are packed into a byte buffer and handed to ``pygame.image.frombuffer``.
    The grid is the ROS one Unity publishes: ``index = row * width + column``,
    a column counting along +x from the published origin and a row along +y.
    """
    bounds = _map_bounds(message)
    info = getattr(message, "info", None)
    data = getattr(message, "data", None)
    if bounds is None or data is None:
        return None
    width = int(_as_float(getattr(info, "width", None)) or 0)
    height = int(_as_float(getattr(info, "height", None)) or 0)
    cells = width * height
    if cells <= 0 or len(data) < cells:
        return None

    palette = _palette()
    unknown = palette[-1]
    try:
        # Row 0 of the grid is its lowest y and the picture is drawn with +y upwards, so the rows are
        # packed from the last to the first: the top of the surface is then the highest y, which is where
        # the robot, the crowd and the lidar returns are plotted.
        pixels = b"".join(
            palette.get(int(data[row * width + column]), unknown)
            for row in range(height - 1, -1, -1)
            for column in range(width)
        )
    except (TypeError, ValueError):
        return None

    surface = pygame.image.frombuffer(pixels, (width, height), "RGB")
    size = (max(1, rect.width), max(1, rect.height))
    return pygame.transform.scale(surface, size)


def _palette() -> dict[int, bytes]:
    """One RGB triple per occupancy value: free 0, walls 100, unknown -1."""
    return {value: bytes(_occupancy_colour(value)) for value in range(-1, 101)}


def _occupancy_colour(value: int) -> tuple[int, int, int]:
    """Colour of one occupancy value, darkening with the probability."""
    if value < 0:
        return MAP_UNKNOWN
    if value == 0:
        return MAP_FREE
    if value >= 100:
        return MAP_WALL
    ratio = value / 100.0
    return tuple(
        int(round(free + (wall - free) * ratio))
        for free, wall in zip(MAP_FREE, MAP_WALL)
    )


# -- HUD --------------------------------------------------------------------


def _hud_lines(
    snapshot: dict,
    pose,
    agent_count: int,
    laser_count: int,
    speed,
    map_message,
) -> list[str]:
    """The three status lines: run state, clock and speed, then the world."""
    state = _text(snapshot.get("simulation_state"))
    scenario = _text(snapshot.get("scenario_name"))
    bounds = _map_bounds(map_message)
    if bounds is None:
        map_text = "map waiting"
    else:
        map_text = f"map {bounds[2] - bounds[0]:.1f} x {bounds[3] - bounds[1]:.1f} m"
    if pose is None:
        robot_text = "robot waiting"
    else:
        robot_text = (
            f"robot {pose[0]:+.2f},{pose[1]:+.2f} yaw {math.degrees(pose[2]):+.0f}deg"
        )
    return [
        f"simulation {state or 'no state yet'} | scenario {scenario or 'none'}",
        f"sim time {_format_seconds(snapshot.get('sim_time_seconds'))} | "
        f"time scale {_format_scale(snapshot.get('time_scale'))} | "
        f"speed {_format_speed(speed)}",
        f"agents {agent_count} | lidar {laser_count} pts | {map_text} | {robot_text}",
    ]


def _text(value) -> str | None:
    """``value`` when it is a string, else ``None``."""
    return value if isinstance(value, str) else None


def _format_seconds(value) -> str:
    seconds = _as_float(value)
    return "---" if seconds is None else f"{seconds:.1f}s"


def _format_scale(value) -> str:
    scale = _as_float(value)
    return "x?" if scale is None else f"x{scale:.2f}"


def _format_speed(value) -> str:
    speed = _as_float(value)
    return "---" if speed is None else f"{speed:.2f} m/s"
