"""``robotsnap bridge --viewer``: the bridge draws the session it already serves.

The runner has two modes and they are alternatives: a line per tick on stdout, or the package's own
window. This checks the window one without a display - pygame is injected, and so is the ``Viewer``
class - because what matters here is not the drawing but the wiring: the viewer must be built on a
client wrapped around *this* bridge, so it reads the session Unity dials rather than opening one.
"""

from __future__ import annotations

import sys
import types

from robotsnap.bridge import __main__ as runner


class _FakeClock:
    def __init__(self):
        self.ticks: list[float] = []

    def tick(self, rate):
        self.ticks.append(rate)


class _FakePygame(types.ModuleType):
    def __init__(self):
        super().__init__("pygame")
        self.initialized = 0
        self.quit_calls = 0
        self.clock = _FakeClock()
        # ``pygame.time.Clock()`` is what the runner asks for, so ``time`` is a namespace of its own
        # rather than a method wearing the submodule's name.
        self.time = types.SimpleNamespace(Clock=lambda: self.clock)

    def init(self):
        self.initialized += 1

    def quit(self):
        self.quit_calls += 1


class _FakeViewer:
    """Draws three frames then closes itself, which is what a window closed by hand looks like."""

    made: list[tuple] = []
    frames = 0

    def __init__(self, pygame, client):
        self.pygame = pygame
        self.client = client
        self.updates = 0
        type(self).made.append((pygame, client))

    def handle_events(self):
        return self.updates < 3

    def update(self):
        self.updates += 1


class _Bridge:
    """Just enough bridge for the client the runner wraps around it."""

    is_running = True
    is_connected = True
    last_error = None

    def start(self):
        raise AssertionError("the runner's bridge is already running; nothing should start it twice")


def test_the_viewer_mode_draws_the_bridge_the_runner_already_serves(monkeypatch):
    pygame = _FakePygame()
    monkeypatch.setitem(sys.modules, "pygame", pygame)
    monkeypatch.setattr("robotsnap.viewer.Viewer", _FakeViewer)
    _FakeViewer.made.clear()
    bridge = _Bridge()

    runner._draw_ticks(bridge, rate=12.0)

    assert pygame.initialized == 1 and pygame.quit_calls == 1
    (made_pygame, client), = _FakeViewer.made
    assert made_pygame is pygame
    assert client.bridge is bridge, "the window reads the session this bridge serves"
    assert pygame.clock.ticks == [12.0, 12.0, 12.0], "one redraw per frame, at the asked rate"


def test_the_viewer_mode_says_what_is_missing_instead_of_crashing(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "pygame", None)

    runner._draw_ticks(_Bridge(), rate=12.0)

    assert "pygame" in capsys.readouterr().err
