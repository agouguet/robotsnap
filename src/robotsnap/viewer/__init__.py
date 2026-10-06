"""Live 2D view of a RobotSNAP session, fed by :mod:`robotsnap.client`.

Run it as ``python -m robotsnap.viewer`` or through
``python -m robotsnap viewer``. This package used to be a single module; the
public surface it had is kept here, and the code now lives in one module per
responsibility:

- :mod:`~robotsnap.viewer.colours` - the palette and every tunable constant.
- :mod:`~robotsnap.viewer.geometry` - poses, frames, and points into the world.
- :mod:`~robotsnap.viewer.session` - the client read as robot, crowd and lidar rows.
- :mod:`~robotsnap.viewer.placement` - placing a scan at the instant it was measured.
- :mod:`~robotsnap.viewer.scene` - the occupancy grid and the HUD text.
- :mod:`~robotsnap.viewer.window` - the ``Viewer`` window: events and drawing.
- :mod:`~robotsnap.viewer.run` - the ``main`` command line entry point.
"""

from __future__ import annotations

from robotsnap.viewer.colours import (
    AGENT_HIDDEN,
    AGENT_OUTLINE,
    AGENT_VISIBLE,
    BACKGROUND,
    DEFAULT_RATE,
    DEFAULT_VIEW_SPAN,
    GOAL,
    GRID,
    HUD_BACKGROUND,
    HUD_HEIGHT,
    LASER,
    MAP_FREE,
    MAP_MARGIN,
    MAP_UNKNOWN,
    MAP_WALL,
    PENDING_SCANS,
    POSE_HISTORY,
    ROBOT_BODY,
    ROBOT_FRAME,
    ROBOT_OUTLINE,
    ROBOT_PALETTE,
    ROBOT_RADIUS,
    SCAN_HOLD_SECONDS,
    TEXT,
    WAITING,
    WINDOW_SIZE,
    WORLD_FRAME,
)
from robotsnap.viewer.geometry import (
    _as_float,
    _ground_y,
    _header_stamp,
    _map_bounds,
    _place,
    _robot_pose,
    _state_pose,
    _to_world,
    _twist_speed,
    _world_point,
    _yaw,
)
from robotsnap.viewer.run import _parse_args, main
from robotsnap.viewer.scene import (
    _format_scale,
    _format_seconds,
    _format_speed,
    _hud_lines,
    _map_count,
    _map_image,
    _occupancy_colour,
    _palette,
    _text,
)
from robotsnap.viewer.session import (
    _agent_entries,
    _agent_frame,
    _agent_visibility,
    _agents,
    _laser_points,
    _laser_scan,
    _robots,
)
from robotsnap.viewer.window import Viewer

__all__ = ["Viewer", "main"]
