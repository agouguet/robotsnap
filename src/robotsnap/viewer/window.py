"""The pygame window: its events, and every drawing call it makes.

``Viewer`` owns the surface, folds in the window events, and redraws once per
``update()``. All of its drawing is a function of whatever the client currently
holds, so a disconnected peer, a missing topic and an empty map all render.
The placement of a stream at the instant it was measured lives in
:class:`robotsnap.viewer.placement._Placement`, which this class mixes in.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

from robotsnap.topics import MAP, ODOM
from robotsnap.viewer.colours import (
    AGENT_HIDDEN,
    AGENT_OUTLINE,
    AGENT_VISIBLE,
    BACKGROUND,
    DEFAULT_VIEW_SPAN,
    GOAL,
    GRID,
    HUD_BACKGROUND,
    HUD_HEIGHT,
    LASER,
    MAP_MARGIN,
    MAP_UNKNOWN,
    PENDING_SCANS,
    POSE_HISTORY,
    ROBOT_FRAME,
    ROBOT_OUTLINE,
    ROBOT_RADIUS,
    TEXT,
    WAITING,
    WINDOW_SIZE,
)
from robotsnap.viewer.geometry import (
    _map_bounds,
    _place,
    _robot_pose,
    _state_pose,
    _twist_speed,
    _world_point,
)
from robotsnap.viewer.placement import _Placement
from robotsnap.viewer.scene import _hud_lines, _map_count, _map_image
from robotsnap.viewer.session import _agents, _laser_points, _laser_scan, _robots


class Viewer(_Placement):
    """One pygame window showing one RobotSNAP session.

    Every drawing call is a function of whatever the client currently holds, so
    a disconnected peer, a missing topic and an empty map all render.
    """

    def __init__(self, pygame, client, size: tuple[int, int] = WINDOW_SIZE):
        self.pygame = pygame
        self.client = client
        self.width, self.height = size
        self.screen = pygame.display.set_mode(size)
        pygame.display.set_caption("RobotSNAP - Unity session")
        self.font = pygame.font.SysFont("dejavusansmono,couriernew,monospace", 14)
        self.port = client.bridge.port
        # What the status lines call the session: a port Unity dials, or the graph a ROS2 session
        # already lives on. The viewer never reads it for anything but saying where it is listening.
        self.session_label = getattr(client, "session_label", f"port {self.port}")
        self._closed = False
        self._connection_announced: bool | None = None
        self._map_key: Any = None
        self._map_image = None
        # Stamped poses of the primary robot, newest last: a scan is placed at the instant it was measured,
        # not at the instant it arrived.
        self._poses: deque = deque(maxlen=POSE_HISTORY)
        # Scans that arrived before the pose of their own instant did, oldest first.
        self._held: deque = deque(maxlen=PENDING_SCANS)

    @property
    def closed(self) -> bool:
        """True once the window was closed or Esc was pressed."""
        return self._closed

    def handle_events(self) -> bool:
        """Fold in the pending window events, returning False to stop the loop."""
        for event in self.pygame.event.get():
            if event.type == self.pygame.QUIT:
                self._closed = True
            elif event.type == self.pygame.KEYDOWN and event.key in (
                self.pygame.K_ESCAPE,
                self.pygame.K_q,
            ):
                self._closed = True
        return not self._closed

    def update(self) -> None:
        """Read the session once and redraw the window."""
        snapshot = self._snapshot()
        odom = self.client.last_message(ODOM)
        pose = _robot_pose(odom)
        if pose is None:
            pose = _state_pose(snapshot.get("robot"))
        self._remember(odom, pose)
        robots = _robots(self.client, snapshot, pose)
        map_message = self.client.last_message(MAP)
        scan = _laser_scan(self.client)
        self._queue_scan(scan)
        placements = self._collect_scans(pose)
        agents, frame = _agents(self.client, snapshot)
        transform = self._view_transform(map_message, pose)

        self.screen.fill(BACKGROUND)
        if map_message is None:
            self._draw_grid(transform)
        else:
            self._draw_map(map_message, transform)
        laser_count = 0
        for held_scan, placement in placements:
            points = _laser_points(held_scan)
            laser_count += len(points)
            self._draw_laser(points, placement, transform)
        self._draw_agents(agents, frame, pose, transform)
        self._draw_goal(snapshot, transform)
        self._draw_robots(robots, transform)
        self._draw_hud(
            _hud_lines(
                snapshot=snapshot,
                pose=pose,
                agent_count=len(agents),
                laser_count=laser_count,
                speed=_twist_speed(odom),
                map_message=map_message,
            )
        )
        if not self._is_connected():
            self._draw_waiting()
        self.pygame.display.flip()

    # -- reading the session ------------------------------------------------

    def _snapshot(self) -> dict:
        """The last ``/simulation/state`` body, or an empty dict."""
        state = self.client.snapshot()
        return state if isinstance(state, dict) else {}

    def _is_connected(self) -> bool:
        """True while Unity holds the socket; announced once per change."""
        connected = bool(self.client.is_connected)
        if connected != self._connection_announced:
            self._connection_announced = connected
            print(
                f"Unity connected on {self.session_label}."
                if connected
                else f"no Unity peer yet: still listening on {self.session_label}."
            )
        return connected

    # -- view geometry ------------------------------------------------------

    def _view_transform(self, map_message, pose):
        """``(pixels per metre, screen x of world 0, screen y of world 0)``.

        A map, once it has arrived, is fitted whole into the window so the
        walls, the robot and the crowd always share one frame. Without one the
        view is centred on the robot at a fixed span.
        """
        bounds = _map_bounds(map_message)
        if bounds is None:
            scale = min(self.width, self.height) / DEFAULT_VIEW_SPAN
            centre_x, centre_y = pose[:2] if pose is not None else (0.0, 0.0)
        else:
            min_x, min_y, max_x, max_y = bounds
            span_x = max(max_x - min_x, 1e-6)
            span_y = max(max_y - min_y, 1e-6)
            scale = (
                min(self.width / span_x, self.height / span_y)
                * (1.0 - 2.0 * MAP_MARGIN)
            )
            centre_x = 0.5 * (min_x + max_x)
            centre_y = 0.5 * (min_y + max_y)
        return (
            scale,
            self.width / 2.0 - centre_x * scale,
            self.height / 2.0 + centre_y * scale,
        )

    def _to_screen(self, transform, x: float, y: float) -> tuple[int, int]:
        """World metres to window pixels."""
        scale, offset_x, offset_y = transform
        return int(round(offset_x + x * scale)), int(round(offset_y - y * scale))

    # -- drawing ------------------------------------------------------------

    def _draw_grid(self, transform) -> None:
        """A one-metre grid, so distances read off before the map arrives."""
        scale, offset_x, offset_y = transform
        if scale <= 0.0:
            return
        x = offset_x % scale
        while x < self.width:
            self.pygame.draw.line(self.screen, GRID, (x, 0), (x, self.height), 1)
            x += scale
        y = offset_y % scale
        while y < self.height:
            self.pygame.draw.line(self.screen, GRID, (0, y), (self.width, y), 1)
            y += scale

    def _draw_map(self, map_message, transform) -> None:
        """Draw the occupancy grid, cached until a newer one arrives."""
        bounds = _map_bounds(map_message)
        if bounds is None:
            return
        min_x, min_y, max_x, max_y = bounds
        top_left = self._to_screen(transform, min_x, max_y)
        bottom_right = self._to_screen(transform, max_x, min_y)
        rect = self.pygame.Rect(
            top_left,
            (
                max(1, bottom_right[0] - top_left[0]),
                max(1, bottom_right[1] - top_left[1]),
            ),
        )
        key = (_map_count(self.client), id(map_message.data), rect.size)
        if key != self._map_key:
            self._map_key = key
            self._map_image = _map_image(self.pygame, map_message, rect)
        if self._map_image is None:
            self.pygame.draw.rect(self.screen, MAP_UNKNOWN, rect, 1)
        else:
            self.screen.blit(self._map_image, rect.topleft)

    def _draw_laser(self, points, pose, transform) -> None:
        """Draw the lidar returns, placed in the world with the robot pose.

        The scan frame is fixed to the robot, so the points are always placed
        with the robot pose, whatever frame the agents are drawn in.
        """
        for point in points:
            world = _place(point, ROBOT_FRAME, pose)
            if world is None:
                continue
            px, py = self._to_screen(transform, world[0], world[1])
            if 0 <= px < self.width and 0 <= py < self.height:
                self.screen.set_at((px, py), LASER)

    def _draw_agents(self, agents, frame, pose, transform) -> None:
        """Draw every agent, dimmed when the lidar cannot see it."""
        for entry in agents:
            world = _place(_world_point(entry), frame, pose)
            if world is None:
                continue
            px, py = self._to_screen(transform, world[0], world[1])
            colour = AGENT_VISIBLE if entry.get("visible", True) else AGENT_HIDDEN
            rect = self.pygame.Rect(px - 5, py - 5, 10, 10)
            self.pygame.draw.rect(self.screen, colour, rect)
            self.pygame.draw.rect(self.screen, AGENT_OUTLINE, rect, 1)
            label = self.font.render(str(entry.get("id", "?")), True, TEXT)
            # The id sits on a dark patch, so it reads over a light map too.
            box = label.get_rect(topleft=(px + 8, py - 9))
            self.screen.fill(HUD_BACKGROUND, box)
            self.screen.blit(label, box)

    def _draw_goal(self, snapshot, transform) -> None:
        """Draw the robot goal, when the snapshot carries one."""
        if snapshot.get("robot_has_goal") is False:
            return
        point = _world_point(snapshot.get("robot_goal"))
        if point is None:
            return
        px, py = self._to_screen(transform, point[0], point[1])
        for radius in (10, 16, 22):
            self.pygame.draw.circle(self.screen, GOAL, (px, py), radius, 1)
        self.pygame.draw.line(self.screen, GOAL, (px - 26, py), (px + 26, py), 1)
        self.pygame.draw.line(self.screen, GOAL, (px, py - 26), (px, py + 26), 1)

    def _draw_robots(self, robots, transform) -> None:
        """Draw every robot of the roster, labelled with its id."""
        for robot_id, pose, colour in robots:
            if pose is None:
                continue
            px, py = self._to_screen(transform, pose[0], pose[1])
            length = ROBOT_RADIUS * 1.7
            head = (
                int(round(px + math.cos(pose[2]) * length)),
                int(round(py - math.sin(pose[2]) * length)),
            )
            self.pygame.draw.circle(self.screen, colour, (px, py), ROBOT_RADIUS)
            self.pygame.draw.circle(
                self.screen, ROBOT_OUTLINE, (px, py), ROBOT_RADIUS, 2
            )
            self.pygame.draw.line(self.screen, ROBOT_OUTLINE, (px, py), head, 3)
            self.pygame.draw.circle(self.screen, ROBOT_OUTLINE, head, 3)
            if robot_id is None:
                continue
            label = self.font.render(robot_id, True, TEXT)
            box = label.get_rect(topleft=(px + 8, py + 6))
            self.screen.fill(HUD_BACKGROUND, box)
            self.screen.blit(label, box)

    def _draw_hud(self, lines) -> None:
        """Draw the status band across the top of the window."""
        band = (0, 0, self.width, HUD_HEIGHT)
        self.pygame.draw.rect(self.screen, HUD_BACKGROUND, band)
        self.pygame.draw.line(
            self.screen, GRID, (0, HUD_HEIGHT), (self.width, HUD_HEIGHT), 1
        )
        for index, line in enumerate(lines):
            self.screen.blit(self.font.render(line, True, TEXT), (12, 8 + index * 17))

    def _draw_waiting(self) -> None:
        """Say what the viewer waits for while nothing is connected."""
        port = "?" if self.port is None else self.port
        centre = (self.width // 2, self.height // 2)
        self._blit_centred(
            f"waiting for Unity to connect on port {port}", centre, WAITING
        )
        self._blit_centred(
            "start the scene, or check that the bridge port matches",
            (centre[0], centre[1] + 22),
            TEXT,
        )

    def _blit_centred(self, message: str, centre, colour) -> None:
        """Draw one line of text centred on ``centre``."""
        text = self.font.render(message, True, colour)
        self.screen.blit(text, text.get_rect(center=centre))
