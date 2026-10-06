"""The one way the core reaches a transport it does not ship.

The core imports no ROS2 code: it reads the ``robotsnap.transports`` entry-point group and refuses
with one actionable line when nothing is registered there. These tests pin that refusal without a
plugin present, which is what keeps the core importable and testable on its own.
"""

from __future__ import annotations

import importlib.metadata

import pytest

from robotsnap.bridge.transport import (
    ENTRY_POINT_GROUP,
    Ros2Unavailable,
    TransportUnavailable,
    load_ros2,
    load_transport,
)


def _nothing_registered(*, group, name):
    """What ``entry_points(group=..., name=...)`` answers on a machine with no transport installed."""
    assert group == ENTRY_POINT_GROUP
    return ()


def test_load_transport_says_what_to_install_when_nothing_is_registered(monkeypatch):
    monkeypatch.setattr(importlib.metadata, "entry_points", _nothing_registered)

    with pytest.raises(TransportUnavailable) as raised:
        load_transport("ros2")

    assert "'ros2'" in str(raised.value)
    assert "pip install robotsnap-ros2" in str(raised.value), "the refusal has to say the way out"


def test_load_ros2_raises_the_name_the_cli_and_viewer_catch(monkeypatch):
    monkeypatch.setattr(importlib.metadata, "entry_points", _nothing_registered)

    with pytest.raises(Ros2Unavailable) as raised:
        load_ros2()

    # The subclass is what lets ``except Ros2Unavailable`` catch a transport that was never installed
    # and one that is installed but unusable, through the same handler.
    assert isinstance(raised.value, TransportUnavailable)
    assert "pip install robotsnap-ros2" in str(raised.value)


def test_a_registered_transport_is_loaded_and_handed_back(monkeypatch):
    class _EntryPoint:
        def load(self):
            return "the-ros2-module"

    seen: dict[str, str] = {}

    def _registered(*, group, name):
        seen["group"], seen["name"] = group, name
        return (_EntryPoint(),)

    monkeypatch.setattr(importlib.metadata, "entry_points", _registered)

    assert load_transport("ros2") == "the-ros2-module"
    assert seen == {"group": ENTRY_POINT_GROUP, "name": "ros2"}


_WITHOUT_THE_PLUGIN = """
import importlib.abc, importlib.metadata, sys

class _Hidden(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in {"rclpy", "robotsnap_ros2"}:
            raise ImportError(name)
        return None

sys.meta_path.insert(0, _Hidden())
# As if the plugin were never installed: the group answers nothing for any name.
importlib.metadata.entry_points = lambda **params: ()

import robotsnap.bridge.__main__
import robotsnap.client
import robotsnap.viewer.run

assert "rclpy" not in sys.modules, "the core imported rclpy at import time"
assert "robotsnap_ros2" not in sys.modules, "the core imported the plugin at import time"

from robotsnap.bridge.transport import Ros2Unavailable, load_ros2

try:
    load_ros2()
except Ros2Unavailable as exc:
    assert "pip install robotsnap-ros2" in str(exc)
else:
    raise AssertionError("load_ros2 answered without a transport installed")

try:
    robotsnap.client.open_client(transport="ros2")
except Ros2Unavailable as exc:
    assert "pip install robotsnap-ros2" in str(exc)
else:
    raise AssertionError("open_client answered without a transport installed")
"""


def test_the_core_imports_and_refuses_without_the_plugin():
    """Prove it in a fresh interpreter that blocks the plugin and rclpy outright."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c", _WITHOUT_THE_PLUGIN],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
