"""The viewer's frame arithmetic and its reading of the published grid."""

from __future__ import annotations

import collections
import math
from types import SimpleNamespace

import pytest

from robotsnap import viewer


class _FakePygame:
    """Just enough pygame for :func:`viewer._map_image`: it packs and returns the pixels."""

    class image:
        @staticmethod
        def frombuffer(pixels, size, kind):
            return {"pixels": pixels, "size": size, "kind": kind}

    class transform:
        @staticmethod
        def scale(surface, size):
            return surface


class _FakeRect:
    def __init__(self, position, size=None):
        if size is None:
            self.x, self.y, self.width, self.height = position
        else:
            self.x, self.y = position
            self.width, self.height = size
        self.topleft = (self.x, self.y)
        self.size = (self.width, self.height)

    @property
    def center(self):
        return self.x + self.width / 2.0, self.y + self.height / 2.0


class _FakeLabel:
    def __init__(self, text):
        self.text = text

    def get_rect(self, **kwargs):
        rect = _FakeRect((0, 0, 10 * len(self.text), 12))
        for name, value in kwargs.items():
            setattr(rect, name, value)
        return rect

    def get_width(self):
        return 10 * len(self.text)


class _FakeFont:
    @staticmethod
    def render(text, antialias, colour):
        return _FakeLabel(str(text))


class _FakeScreen:
    """Records every pixel the viewer plots, so a placement can be checked."""

    def __init__(self, size):
        self.size = size
        self.pixels = []

    def fill(self, colour, rect=None):
        pass

    def blit(self, surface, position):
        pass

    def set_at(self, position, colour):
        self.pixels.append((position, colour))


class _FakePygameWindow(_FakePygame):
    """The whole pygame surface the Viewer touches, not just the image helpers."""

    Rect = _FakeRect
    KEYDOWN = 2
    QUIT = 256
    K_ESCAPE = 27
    K_q = 113

    def __init__(self):
        self.screen = None

    def init(self):
        return None

    def quit(self):
        return None

    class display:
        screen = None

        @classmethod
        def set_mode(cls, size):
            cls.screen = _FakeScreen(size)
            return cls.screen

        @staticmethod
        def set_caption(caption):
            pass

        @staticmethod
        def flip():
            pass

    class font:
        @staticmethod
        def SysFont(name, size):
            return _FakeFont()

    class event:
        @staticmethod
        def get():
            return []

    class draw:
        @staticmethod
        def rect(*args, **kwargs):
            pass

        @staticmethod
        def line(*args, **kwargs):
            pass

        @staticmethod
        def circle(*args, **kwargs):
            pass


def _map_message(width, height, resolution, origin_x, origin_y, data):
    return SimpleNamespace(
        info=SimpleNamespace(
            width=width,
            height=height,
            resolution=resolution,
            origin=SimpleNamespace(position=SimpleNamespace(x=origin_x, y=origin_y)),
        ),
        data=list(data),
    )


def test_the_grid_is_packed_with_its_lowest_row_at_the_bottom():
    """Row 0 of a ROS grid is its lowest y, and the picture is drawn with +y upwards.

    The two rows of this grid differ on purpose: packing them in published order
    would put the walls at the top of the surface, where the viewer plots the
    highest y.
    """
    walls, free = 100, 0
    message = _map_message(2, 2, 0.5, 0.0, 0.0, [walls, walls, free, free])
    surface = viewer._map_image(_FakePygame, message, SimpleNamespace(width=40, height=40))

    assert surface["size"] == (2, 2)
    assert surface["kind"] == "RGB"
    assert surface["pixels"][:3] == bytes(viewer.MAP_FREE)
    assert surface["pixels"][6:9] == bytes(viewer.MAP_WALL)


def test_the_map_bounds_are_the_published_origin_and_size():
    message = _map_message(4, 2, 0.5, -1.0, -0.25, [0] * 8)
    assert viewer._map_bounds(message) == (-1.0, -0.25, 1.0, 0.75)


@pytest.mark.parametrize(
    "yaw, expected",
    [
        (0.0, (1.0, 0.0)),
        (math.pi / 2.0, (0.0, 1.0)),
        (math.pi, (-1.0, 0.0)),
        (-math.pi / 2.0, (0.0, -1.0)),
    ],
)
def test_a_robot_frame_point_is_turned_by_the_robot_yaw(yaw, expected):
    """The robot frame is ROS too, so a point 1 m ahead follows the robot's own yaw."""
    world = viewer._place((1.0, 0.0), viewer.ROBOT_FRAME, (0.0, 0.0, yaw))
    assert world == pytest.approx(expected, abs=1e-9)


# --- placing a stream at the instant it was measured ------------------------


def _viewer_with_poses(poses):
    """A Viewer with only the pose history filled in: no pygame, no client."""
    instance = object.__new__(viewer.Viewer)
    instance._poses = collections.deque(poses, maxlen=viewer.POSE_HISTORY)
    return instance


def test_a_scan_is_placed_at_its_own_stamp():
    """Between two stamped poses the scan gets the pose the robot held, not the newest one."""
    instance = _viewer_with_poses(
        [(10.0, 1.0, 0.0, 0.0), (11.0, 3.0, 2.0, math.pi / 2.0)]
    )

    assert instance._pose_at(10.5) == pytest.approx((2.0, 1.0, math.pi / 4.0))
    assert instance._pose_at(10.0) is None, "the ends are not extrapolated"
    assert instance._pose_at(11.0) is None
    assert instance._pose_at(None) is None


def test_a_single_pose_is_not_extrapolated_from():
    """One odometry message cannot say where the robot was before or after it."""
    instance = _viewer_with_poses([(10.0, 1.0, 0.0, 0.0)])
    assert instance._pose_at(10.5) is None


def test_the_interpolated_yaw_takes_the_short_way_round():
    """A robot crossing +-pi turns a little, not almost the whole circle."""
    instance = _viewer_with_poses(
        [(0.0, 0.0, 0.0, 3.0), (1.0, 0.0, 0.0, 3.0 + 0.4 - 2.0 * math.pi)]
    )

    yaw = instance._pose_at(0.5)[2]
    assert math.atan2(math.sin(yaw - 3.2), math.cos(yaw - 3.2)) == pytest.approx(0.0, abs=1e-9)


# --- the crowd ---------------------------------------------------------------


class _FakeClient:
    def __init__(self, payload):
        self._payload = payload

    def agents(self):
        return self._payload


def test_the_crowd_comes_from_the_snapshot_and_its_flag_from_the_agent_stream():
    """The snapshot carries the crowd and the robot of one instant, so it cannot lag behind the pose."""
    client = _FakeClient(
        {"agents": [{"id": "0", "x": 1.0, "y": 2.0, "visible": False}], "frame": "robot"}
    )
    snapshot = {"humans": [{"id": 0, "x": 5.0, "y": 6.0}, {"id": 1, "x": 7.0, "y": 8.0}]}

    entries, frame = viewer._agents(client, snapshot)

    assert frame == viewer.WORLD_FRAME
    assert [entry["x"] for entry in entries] == [5.0, 7.0]
    assert entries[0]["visible"] is False
    assert entries[1].get("visible") is None


def test_the_robot_frame_stream_is_used_while_no_snapshot_has_arrived():
    client = _FakeClient(
        {"agents": [{"id": "0", "x": 1.0, "y": 2.0, "visible": True}], "frame": "robot"}
    )

    entries, frame = viewer._agents(client, {})

    assert frame == viewer.ROBOT_FRAME
    assert [entry["x"] for entry in entries] == [1.0]


# --- the whole redraw --------------------------------------------------------


def _odom(stamp, x, y, yaw):
    half = yaw / 2.0
    return SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(sec=int(stamp), nanosec=int((stamp % 1) * 1e9))),
        pose=SimpleNamespace(
            pose=SimpleNamespace(
                position=SimpleNamespace(x=x, y=y, z=0.0),
                orientation=SimpleNamespace(
                    x=0.0, y=0.0, z=math.sin(half), w=math.cos(half)
                ),
            )
        ),
        twist=SimpleNamespace(twist=SimpleNamespace(linear=SimpleNamespace(x=0.0, y=0.0))),
    )


def _scan(stamp, ranges, angle_min=-1.0, angle_increment=0.5):
    from robotsnap.bridge.state import LaserScanState

    return LaserScanState(
        angle_min=angle_min,
        angle_max=angle_min + angle_increment * (len(ranges) - 1),
        angle_increment=angle_increment,
        range_min=0.1,
        range_max=10.0,
        ranges=tuple(float(r) for r in ranges),
        frame_id="laser",
        stamp=stamp,
    )


class _ViewerClient:
    """A connected session with nothing in it until a test puts something in it."""

    class _Bridge:
        port = 10000

        def __init__(self, scan):
            self._scan = scan

        def snapshot(self):
            return SimpleNamespace(laser=self._scan)

        def topic_counts(self):
            return {}

    def __init__(self, scans=(), messages=None, snapshot=None, agents=None):
        # ``scans`` is the sequence of /scan states the bridge answers with, one per read.
        self._scans = list(scans)
        self._messages = messages or {}
        self._snapshot = snapshot or {}
        self._agents = agents or {"agents": [], "frame": "robot"}
        self.bridge = self._Bridge(self._scans[0] if self._scans else None)

    @property
    def is_connected(self):
        return True

    def last_message(self, topic):
        return self._messages.get(topic)

    def snapshot(self):
        return self._snapshot

    def agents(self):
        return self._agents


def test_an_empty_session_redraws_without_raising():
    """Nothing has arrived: the window still draws, and says what it waits for."""
    pygame = _FakePygameWindow()
    instance = viewer.Viewer(pygame, _ViewerClient())

    instance.update()

    assert instance.closed is False


def test_a_scan_is_drawn_at_the_pose_of_its_own_stamp():
    """The whole redraw path, with a scan that arrived after the robot had turned on."""
    from robotsnap.topics import MAP, ODOM

    # The robot turned a quarter turn between the two odometry samples, one second apart, and the scan
    # was measured half way through it: it must be drawn at the half-way pose, not at either end.
    # The map is wider than the scan, so every beam of it lands inside the window.
    messages = {
        ODOM: _odom(3.0, 0.0, 0.0, math.pi / 2.0),
        MAP: _map_message(40, 40, 0.5, -10.0, -10.0, [100] * 1600),
    }
    pygame = _FakePygameWindow()
    client = _ViewerClient(
        scans=[_scan(2.5, [2.0, 2.0, 2.0])],
        messages=messages,
        snapshot={"robot": {"x": 0.0, "y": 0.0, "yaw": math.pi / 2.0}},
    )
    instance = viewer.Viewer(pygame, client)
    # One earlier pose, so the stamp of the scan falls between two samples.
    instance._remember(_odom(2.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

    instance.update()

    expected = viewer._place((2.0, 0.0), viewer.ROBOT_FRAME, (0.0, 0.0, math.pi / 4.0))
    drawn = [position for position, _ in pygame.display.screen.pixels]
    scale, offset_x, offset_y = instance._view_transform(
        messages[MAP], (0.0, 0.0, math.pi / 2.0)
    )
    want = (int(round(offset_x + expected[0] * scale)), int(round(offset_y - expected[1] * scale)))
    assert any(abs(x - want[0]) <= 1 and abs(y - want[1]) <= 1 for x, y in drawn), (
        f"no beam plotted at the interpolated pose {want}: {drawn[:10]}"
    )


def test_a_scan_waits_for_the_odometry_sample_that_closes_its_interval():
    """A scan is held while the pose after its stamp is missing, then drawn at the interpolated one.

    This is the case every live read is in: the odometry sample that comes after a scan's stamp is published
    at the end of the very interval the scan was measured in, so it always arrives a period late. Drawn at
    once with the newest pose, the cloud of a turning robot lands a period of turn away from the walls.
    """
    from robotsnap.topics import MAP, ODOM

    messages = {
        ODOM: _odom(2.0, 0.0, 0.0, 0.0),
        MAP: _map_message(40, 40, 0.5, -10.0, -10.0, [100] * 1600),
    }
    pygame = _FakePygameWindow()
    client = _ViewerClient(
        scans=[_scan(2.5, [2.0, 2.0, 2.0], angle_min=0.0)],
        messages=messages,
        snapshot={"robot": {"x": 0.0, "y": 0.0, "yaw": 0.0}},
    )
    instance = viewer.Viewer(pygame, client)

    instance.update()
    assert pygame.display.screen.pixels == [], (
        "the scan was drawn with a pose the robot had already left instead of waiting for the one that "
        "closes its interval"
    )

    # The next odometry sample turns a quarter turn over the second that contains the scan.
    messages[ODOM] = _odom(3.0, 0.0, 0.0, math.pi / 2.0)
    instance.update()

    expected = viewer._place((2.0, 0.0), viewer.ROBOT_FRAME, (0.0, 0.0, math.pi / 4.0))
    scale, offset_x, offset_y = instance._view_transform(
        messages[MAP], (0.0, 0.0, math.pi / 2.0)
    )
    want = (
        int(round(offset_x + expected[0] * scale)),
        int(round(offset_y - expected[1] * scale)),
    )
    drawn = [position for position, _ in pygame.display.screen.pixels]
    assert any(abs(x - want[0]) <= 1 and abs(y - want[1]) <= 1 for x, y in drawn), (
        f"the held scan was not drawn at the interpolated pose {want}: {drawn[:10]}"
    )


def test_a_scan_that_waits_too_long_is_drawn_with_the_newest_pose(monkeypatch):
    """A stream that stopped publishing odometry must not hole the scan for ever."""
    from robotsnap.topics import MAP, ODOM

    monkeypatch.setattr(viewer, "SCAN_HOLD_SECONDS", 0.0)
    messages = {
        ODOM: _odom(2.0, 0.0, 0.0, 0.0),
        MAP: _map_message(40, 40, 0.5, -10.0, -10.0, [100] * 1600),
    }
    pygame = _FakePygameWindow()
    client = _ViewerClient(
        scans=[_scan(2.5, [2.0, 2.0, 2.0], angle_min=0.0)],
        messages=messages,
        snapshot={"robot": {"x": 0.0, "y": 0.0, "yaw": 0.0}},
    )
    instance = viewer.Viewer(pygame, client)

    instance.update()

    expected = viewer._place((2.0, 0.0), viewer.ROBOT_FRAME, (0.0, 0.0, 0.0))
    scale, offset_x, offset_y = instance._view_transform(messages[MAP], (0.0, 0.0, 0.0))
    want = (
        int(round(offset_x + expected[0] * scale)),
        int(round(offset_y - expected[1] * scale)),
    )
    drawn = [position for position, _ in pygame.display.screen.pixels]
    assert any(abs(x - want[0]) <= 1 and abs(y - want[1]) <= 1 for x, y in drawn), (
        f"an expired scan was dropped instead of drawn with the newest pose: {drawn[:10]}"
    )
