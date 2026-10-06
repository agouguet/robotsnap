"""Fixtures shared by every suite of the package.

The fake Unity peer itself lives in ``tests/unity_peer.py``: one implementation
of the ROS-TCP-Connector framing means one place to fix when the wire changes.
The peer here registers the streams the bridge and the client suites need; a
suite that needs another one registers it itself, as the environment tests do
with ``/odom``, ``/scan`` and ``/map``.
"""

import pytest

from robotsnap.bridge.server import RobotSNAPBridge
from robotsnap.client import RobotSNAPClient
from unity_peer import CONNECTION_TIMEOUT as _CONNECTION_TIMEOUT
from unity_peer import UnityPeer


@pytest.fixture
def bridge():
    """A bridge bound to a free local port."""
    server = RobotSNAPBridge(host="127.0.0.1", port=0)
    server.start()
    try:
        yield server
    finally:
        server.stop()


@pytest.fixture
def client(bridge):
    """A client driving the fixture bridge, without owning it."""
    facade = RobotSNAPClient(bridge=bridge, autostart=False)
    try:
        yield facade
    finally:
        facade.stop()


@pytest.fixture
def unity(bridge):
    """A connected fake-Unity peer that drains the handshake and registers topics."""
    peer = UnityPeer(bridge.port)
    try:
        assert bridge.wait_for_connection(timeout=_CONNECTION_TIMEOUT) is True
        peer.recv_handshake()
        peer.register_publisher("/simulation/state", "std_msgs/String")
        peer.register_publisher("/simulation/agents", "std_msgs/String")
        peer.register_publisher("/simulation/metrics", "std_msgs/String")
        peer.register_publisher("/simulation/control_result", "std_msgs/String")
        peer.register_publisher("/odom", "nav_msgs/Odometry")
        peer.register_publisher("/scan", "sensor_msgs/LaserScan")
        peer.register_publisher("/map", "nav_msgs/OccupancyGrid")
        peer.register_publisher("/reset_done", "std_msgs/Bool")
        peer.register_subscriber("/simulation/control", "std_msgs/String")
        yield peer
    finally:
        peer.close()


@pytest.fixture
def bare_peer(bridge):
    """A connected fake-Unity peer that has registered nothing yet.

    The window a client has to survive: the socket is up, so ``is_connected``
    is true, but the components that own the streams have not subscribed and a
    command written now would be dispatched to nobody.
    """
    peer = UnityPeer(bridge.port)
    try:
        assert bridge.wait_for_connection(timeout=_CONNECTION_TIMEOUT) is True
        peer.recv_handshake()
        yield peer
    finally:
        peer.close()
