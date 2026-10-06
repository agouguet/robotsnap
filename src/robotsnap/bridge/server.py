"""Runtime server for the RobotSNAP Unity bridge.

Unity runs the ``ros-tcp-connector`` package as a TCP *client* and dials in to
this process, so Python is the server. The endpoint sends ``__handshake`` as the
first bytes on the socket, before reading anything, because Unity's reader
thread requires it. Unity then sends ``__publish`` / ``__subscribe``
registrations and topic payloads over that one socket.

Only one Unity peer is supported at a time: a second connection replaces the
first and the old socket is closed. That is enough for a single simulated
world and keeps reconnection after a Unity domain reload simple.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from typing import Any, Callable

from robotsnap import topics
from robotsnap.bridge import codec, protocol, state
from robotsnap.bridge.state import RobotSNAPState

__all__ = [
    "BridgeError",
    "BridgeSession",
    "RobotSNAPBridge",
    "DEFAULT_TOPIC_TYPES",
    "normalise_msg_type",
]

#: Topic -> rosbags type map, seeded from the fixed surface in
#: :mod:`robotsnap.topics` and keyed by base topic name. Unity announces the
#: same types through ``__publish``, so this only makes the first message of a
#: session decodable.
DEFAULT_TOPIC_TYPES: dict[str, str] = topics.base_types()

_ACCEPT_TIMEOUT = 0.2
_JOIN_TIMEOUT = 1.0
_POLL_INTERVAL = 0.02


class BridgeError(RuntimeError):
    """Raised when the bridge cannot start or a frame cannot be encoded."""


def normalise_msg_type(message_name: str) -> str:
    """Convert a Unity message name to the rosbags ``pkg/msg/Type`` form.

    Unity publishes ``nav_msgs/Odometry``; rosbags expects
    ``nav_msgs/msg/Odometry``. Names that already carry the ROS2 form, or that
    carry an unknown shape, are returned unchanged.
    """
    name = str(message_name).strip()
    if not name or "/msg/" in name or "/srv/" in name:
        return name
    parts = name.split("/")
    if len(parts) == 2:
        return f"{parts[0]}/msg/{parts[1]}"
    return name


def _resolve_key(canonical: str, mapping) -> str | None:
    """Resolve a topic name against a mapping, tolerating prefixes both ways.

    :func:`robotsnap.topics.base` already removed the leading slash. A stored
    ``odom`` matches ``/robot0/odom`` and a stored ``robot0/odom`` matches
    ``/odom``.
    """
    if canonical in mapping:
        return canonical
    parts = canonical.split("/")
    for index in range(1, len(parts)):
        suffix = "/".join(parts[index:])
        if suffix in mapping:
            return suffix
    for key in mapping:
        if key.endswith("/" + canonical):
            return key
    return None


class BridgeSession:
    """One accepted Unity connection.

    The session owns its socket and runs the blocking read loop on its own
    daemon thread. One bad frame is recorded and skipped instead of killing the
    session; only a broken stream ends the loop.
    """

    def __init__(self, sock, bridge, address):
        self._sock = sock
        self._bridge = bridge
        self.address = address
        self._lock = threading.Lock()
        self._closed = False
        self._stopped = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="robotsnap-bridge-session", daemon=True
        )

    @property
    def is_running(self) -> bool:
        return self._thread.is_alive() and not self._stopped.is_set()

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        """Close the socket and let the read loop finish."""
        self._stopped.set()
        self._close()
        if self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(timeout=_JOIN_TIMEOUT)

    def send_frame(self, frame: bytes) -> None:
        """Send one already encoded frame, raising OSError when closed."""
        with self._lock:
            if self._closed:
                raise OSError("session is closed")
            self._sock.sendall(frame)

    def _close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            sock = self._sock
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            sock.close()
        except OSError:
            pass

    def _run(self) -> None:
        sock = self._sock
        reason: Exception | None = None
        try:
            while not self._stopped.is_set():
                try:
                    destination, payload = protocol.read_frame(sock)
                except (protocol.ProtocolError, OSError) as exc:
                    reason = exc
                    break
                try:
                    self._bridge._handle_frame(destination, payload)
                except Exception as exc:  # never let one bad frame kill the session
                    self._bridge._record_error(f"frame handling failed: {exc}")
        finally:
            self._stopped.set()
            self._close()
            self._bridge._on_session_closed(self, reason)


class RobotSNAPBridge:
    """TCP server holding the last decoded message for every Unity topic.

    Every public method is safe to call from another thread while the read loop
    runs; shared state is guarded by one lock and sending by another.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 10000,
        topic_types: dict[str, str] | None = None,
        on_state: Callable[[RobotSNAPState], None] | None = None,
        protocol_name: str = "ROS2",
        version: str = "v0.7.0",
    ):
        self.host = host
        self._requested_port = int(port)
        self._on_state = on_state
        #: Advertised in the handshake Unity receives on connect. Unity checks
        #: ``version`` starts with ``v0.7.`` and ``protocol_name`` equals the
        #: compiled-in protocol (``ROS2`` for this project).
        self.protocol_name = str(protocol_name)
        self.version = str(version)

        self._topic_types: dict[str, str] = dict(DEFAULT_TOPIC_TYPES)
        if topic_types:
            for topic, msg_type in topic_types.items():
                self._topic_types[topics.base(topic)] = normalise_msg_type(msg_type)
        self._subscriptions: dict[str, str] = {}

        self._latest: dict[str, Any] = {}
        self._raw: dict[str, bytes] = {}
        self._counts: dict[str, int] = {}
        self._announced: dict[str, str] = {}
        #: What the peer said it *publishes*, as opposed to what it announced at
        #: all: a topic it only subscribed to is not a stream this process will
        #: ever receive, and a mirror must not offer one on its behalf.
        self._published: dict[str, str] = {}
        self._handshake: dict[str, Any] | None = None
        self._last_error: str | None = None
        #: Mirror objects that read every message as it arrives - a ROS2 gateway, mainly. Kept apart
        #: from ``on_state``, which is called once per message with the whole snapshot and is meant for
        #: a consumer that wants the session rather than the stream.
        self._message_listeners: list[Callable[[str, bytes, str], None]] = []

        self._session: BridgeSession | None = None
        self._port = self._requested_port
        self._running = False
        self._server_sock: socket.socket | None = None
        self._accept_thread: threading.Thread | None = None

        self._lock = threading.RLock()
        self._send_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._connection_event = threading.Event()

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Bind, listen and spawn the accept thread."""
        with self._lock:
            if self._running:
                return
            self._stop_event.clear()
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                server.bind((self.host, self._requested_port))
                server.listen(1)
            except OSError as exc:
                server.close()
                raise BridgeError(
                    f"cannot bind {self.host}:{self._requested_port}: {exc}"
                ) from exc
            server.settimeout(_ACCEPT_TIMEOUT)
            self._server_sock = server
            self._port = int(server.getsockname()[1])
            self._running = True
            thread = threading.Thread(
                target=self._accept_loop, name="robotsnap-bridge-accept", daemon=True
            )
            self._accept_thread = thread
        thread.start()

    def stop(self) -> None:
        """Close every socket and join the threads. Idempotent."""
        with self._lock:
            if (
                not self._running
                and self._server_sock is None
                and self._session is None
                and self._accept_thread is None
            ):
                return
            self._running = False
            self._stop_event.set()
            server = self._server_sock
            self._server_sock = None
            accept_thread = self._accept_thread
            self._accept_thread = None
            session = self._session
            self._session = None
            self._handshake = None
            self._announced.clear()
            self._published.clear()
            self._subscriptions.clear()

        if server is not None:
            try:
                server.close()
            except OSError:
                pass
        if (
            accept_thread is not None
            and accept_thread.is_alive()
            and accept_thread is not threading.current_thread()
        ):
            accept_thread.join(timeout=_JOIN_TIMEOUT)
        if session is not None:
            session.stop()
        self._connection_event.set()

    @property
    def port(self) -> int:
        """The real bound port, needed when the bridge was started with port=0."""
        with self._lock:
            return self._port

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._running

    @property
    def is_connected(self) -> bool:
        # Derived from the bridge's own reference, not from thread liveness, so
        # that "disconnected" and the cleared handshake become visible together.
        with self._lock:
            return self._session is not None

    def wait_for_connection(self, timeout: float | None = None) -> bool:
        """Block until a Unity peer is connected. Returns False on timeout."""
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        while True:
            if self.is_connected:
                return True
            if deadline is None:
                remaining = _POLL_INTERVAL
            else:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
            self._connection_event.wait(min(_POLL_INTERVAL, remaining))
            self._connection_event.clear()

    def wait_for_subscription(
        self, topic: str, timeout: float | None = None
    ) -> bool:
        """Block until the peer has subscribed to ``topic``, at most ``timeout`` seconds.

        The connector drops a message whose topic holds no subscriber *at the
        moment it is dispatched*: it is not queued for a listener that arrives
        later. A write to such a topic therefore disappears without a trace, so
        anything that sends on a topic waits here first. Returns ``True`` when
        the subscription is in place, ``False`` on timeout.

        Unlike :meth:`topic_types`, this is not satisfied by the seed table: it
        reports only what the peer itself said.
        """
        wanted = topics.base(topic)
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        while True:
            if _resolve_key(wanted, self.subscriptions()) is not None:
                return True
            if deadline is None:
                remaining = _POLL_INTERVAL
            else:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
            time.sleep(min(_POLL_INTERVAL, max(0.0, remaining)))

    # -- data access -------------------------------------------------------

    def add_message_listener(self, listener: Callable[[str, bytes, str], None]) -> None:
        """Register a callback called with ``(topic key, payload, message type)`` per message.

        The payload is the bytes Unity sent, untouched, and the key is the canonical topic name
        (:func:`robotsnap.topics.base`). This is what a mirror needs: a gateway that republishes the
        session elsewhere has to hand over what arrived, not a re-encoding of it.
        """
        with self._lock:
            if listener not in self._message_listeners:
                self._message_listeners.append(listener)

    def remove_message_listener(self, listener: Callable[[str, bytes, str], None]) -> None:
        """Stop calling a listener registered by :meth:`add_message_listener`."""
        with self._lock:
            if listener in self._message_listeners:
                self._message_listeners.remove(listener)

    def latest(self, topic: str):
        """Last decoded message for a base topic, or None.

        Accepts ``odom``, ``/odom`` or ``/robot0/odom`` and resolves them to the
        same stored topic.
        """
        canonical = topics.base(topic)
        with self._lock:
            key = _resolve_key(canonical, self._latest)
            if key is None:
                return None
            return self._latest[key]

    def latest_raw(self, topic: str) -> bytes | None:
        """Last raw payload received for a topic, decoded or not."""
        canonical = topics.base(topic)
        with self._lock:
            key = _resolve_key(canonical, self._raw)
            if key is None:
                return None
            return self._raw[key]

    def snapshot(self) -> RobotSNAPState:
        """Aggregate the latest messages into a state snapshot. Never blocks."""
        with self._lock:
            messages = dict(self._latest)
        return state.build_state(messages)

    def handshake_metadata(self) -> dict | None:
        """Metadata sent by Unity in ``__handshake``, or None before it."""
        with self._lock:
            if self._handshake is None:
                return None
            metadata = self._handshake.get("metadata", {})
            return dict(metadata) if isinstance(metadata, dict) else metadata

    def topic_counts(self) -> dict[str, int]:
        """Messages received per canonical topic since the bridge started."""
        with self._lock:
            return dict(self._counts)

    def topic_types(self) -> dict[str, str]:
        """Every topic the bridge can decode, seeded table and peer announcements alike.

        This answers "can a message on this topic be read", not "has the peer
        registered it": the seed makes it non-empty before a single frame
        arrives. Use :meth:`announced_topics` when the question is what the
        peer has actually said.
        """
        with self._lock:
            return dict(self._topic_types)

    def announced_topics(self) -> dict[str, str]:
        """Topics the *peer* has registered, per canonical name.

        Unity sends ``__publish`` and ``__subscribe`` frames from the
        components that own each stream, so this is the first moment the
        bridge knows the peer is not only connected but has finished
        registering. A command written before that can be dispatched to a
        topic nobody listens to yet, which is why a client waits for this
        rather than for the socket. Empty while no peer has registered
        anything, and cleared when the session closes.
        """
        with self._lock:
            return dict(self._announced)

    def subscriptions(self) -> dict[str, str]:
        """Topics the peer subscribed to, per canonical name -> its own spelling.

        These are the streams a client may write to; ``/cmd_vel`` and
        ``/simulation/control`` in the current contract.
        """
        with self._lock:
            return dict(self._subscriptions)

    def published_topics(self) -> dict[str, str]:
        """Topics the peer *publishes*, per canonical name -> the type it named.

        :meth:`announced_topics` is everything the peer registered, which includes
        the topics it only listens to. A mirror that offered a publisher for those
        would be announcing a stream the session never sends - ``/cmd_vel`` above
        all, where a navigation method reading the graph would take the mirror for
        a second author of its own command. This is the narrower question, and the
        one both directions of a mirror need.
        """
        with self._lock:
            return dict(self._published)

    @property
    def last_error(self) -> str | None:
        with self._lock:
            return self._last_error

    # -- outgoing traffic --------------------------------------------------

    def handshake_frame(self) -> bytes:
        """The ``__handshake`` frame Unity expects as the first message.

        Unity's reader thread treats the first frame on the socket as the
        handshake and rejects anything older than ``v0.7``. The payload carries
        ``metadata`` as a JSON string, matching ``{"protocol": protocol_name}``.
        """
        payload = json.dumps(
            {
                "version": self.version,
                "metadata": json.dumps({"protocol": self.protocol_name}),
            }
        ).encode("utf-8")
        return protocol.encode_frame("__handshake", payload)

    def send_cmd_vel(self, linear_x: float, angular_z: float) -> bool:
        """Encode and send a Twist on ``/cmd_vel``. False when disconnected."""
        try:
            payload = codec.encode_twist(linear_x, angular_z)
        except codec.CodecError as exc:
            self._record_error(f"cannot encode cmd_vel: {exc}")
            return False
        return self.send_message(self._cmd_vel_topic(), payload)

    def send_message(self, topic: str, payload: bytes) -> bool:
        """Send raw bytes on a topic. False when nothing is connected."""
        try:
            frame = protocol.encode_frame(topic, payload)
        except (protocol.ProtocolError, TypeError) as exc:
            self._record_error(f"cannot encode frame for {topic}: {exc}")
            return False
        with self._send_lock:
            with self._lock:
                session = self._session
            if session is None:
                return False
            try:
                session.send_frame(frame)
            except OSError as exc:
                self._record_error(f"send failed on {topic}: {exc}")
                self._drop_session(session)
                return False
        return True

    def publish(self, topic: str, msg_type: str, msg) -> bool:
        """Encode a message with the typestore and send it. False on failure."""
        try:
            payload = codec.encode(msg_type, msg)
        except codec.CodecError as exc:
            self._record_error(f"cannot encode {msg_type}: {exc}")
            return False
        return self.send_message(topic, payload)

    def _cmd_vel_topic(self) -> str:
        """The destination a bare velocity command goes to.

        The simulator registers one subscription per robot: the primary robot
        keeps ``/cmd_vel`` and every robot also answers on
        ``/robot_<id>/cmd_vel``. A bare command means the primary robot, so the
        legacy name wins when it is registered, and a namespaced name is only a
        deterministic fallback for a fleet whose primary was renamed. To drive
        one named robot of several, address it explicitly: ``publish`` here, or
        ``RobotSNAPClient.send_cmd_vel(..., robot="robot_2")``.
        """
        return self.destination_for(topics.CMD_VEL)

    def destination_for(self, topic: str) -> str:
        """The name the peer registered for a topic, for a message coming from outside.

        A publisher that is not this bridge - a ROS2 node, a gateway mirroring the graph - holds the
        topic name of its own graph and needs the name Unity actually listens on. The bare name wins
        when the peer registered it, and a namespaced name is the deterministic fallback for a fleet
        whose primary was renamed, the same rule :meth:`_cmd_vel_topic` follows. A topic nobody
        registered is returned unchanged, so a caller still reaches a session that has simply not
        announced it yet.
        """
        wanted = topics.base(topic)
        with self._lock:
            if wanted in self._subscriptions:
                return self._subscriptions[wanted]
            namespaced = sorted(
                name
                for key, name in self._subscriptions.items()
                if key.endswith("/" + wanted)
            )
        return namespaced[0] if namespaced else topic

    # -- accept thread -----------------------------------------------------

    def _accept_loop(self) -> None:
        while not self._stop_event.is_set():
            server = self._server_sock
            if server is None:
                break
            try:
                conn, address = server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                conn.settimeout(None)
            except OSError:
                pass
            # Nagle holds a small write back until the previous one is acknowledged, and a lockstep
            # step writes `cmd_vel` and the release straight after one another: the second one then
            # waits a round trip of the peer's delayed acknowledgement before it leaves, which is tens
            # of milliseconds of wall time on every step. Each message is already framed and sent on
            # its own, so there is nothing to coalesce and the delay buys nothing.
            try:
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except OSError:
                pass

            session = BridgeSession(conn, self, address)
            try:
                # Unity's ReaderThread reads exactly one message on connect and
                # requires it to be the handshake, so it must go out before the
                # read loop starts.
                session.send_frame(self.handshake_frame())
            except OSError as exc:
                self._record_error(f"cannot send handshake: {exc}")
                session.stop()
                continue
            with self._lock:
                previous = self._session
                self._session = session
                self._handshake = None
                # A new peer registers its topics again, so the evidence of the
                # previous one must not let a client skip that wait.
                self._announced.clear()
                self._published.clear()
                self._subscriptions.clear()
            if previous is not None:
                previous.stop()
            session.start()
            self._connection_event.set()

    def _on_session_closed(self, session: BridgeSession, reason=None) -> None:
        with self._lock:
            if self._session is not session:
                return
            self._session = None
            self._handshake = None
            self._announced.clear()
            self._published.clear()
            self._subscriptions.clear()
        if reason is not None and not isinstance(reason, protocol.ProtocolError):
            self._record_error(f"session closed: {reason}")

    def _drop_session(self, session: BridgeSession) -> None:
        with self._lock:
            if self._session is session:
                self._session = None
                self._handshake = None
                self._announced.clear()
                self._published.clear()
                self._subscriptions.clear()
        session.stop()

    # -- frame handling ----------------------------------------------------

    def _handle_frame(self, destination: str, payload: bytes) -> None:
        if destination == "":
            return  # keepalive
        if protocol.is_syscommand(destination):
            self._handle_syscommand(destination, payload)
            return
        self._handle_topic_message(destination, payload)

    def _handle_syscommand(self, command: str, payload: bytes) -> None:
        """Dispatch one system command.

        In ROS2 mode Unity writes the command name as the *frame destination*
        and the JSON body directly as the frame payload, the body carrying a
        trailing NUL because strings include it. The command name is not
        repeated inside the payload.
        """
        try:
            body = payload.rstrip(b"\x00").decode("utf-8")
        except (UnicodeDecodeError, AttributeError) as exc:
            self._record_error(f"{command} payload is not valid utf-8: {exc}")
            return
        if not body:
            self._record_error(f"{command} payload is empty")
            return

        try:
            params: Any = json.loads(body)
        except ValueError as exc:
            self._record_error(f"{command} payload is not valid JSON: {exc}")
            return
        if not isinstance(params, dict):
            self._record_error(f"{command} payload must be a JSON object")
            return

        if command == "__handshake":
            self._handle_handshake(body)
        elif command == "__publish":
            self._handle_registration(params, publishing=True)
        elif command == "__subscribe":
            self._handle_registration(params, publishing=False)
        elif command == "__error":
            self._record_error(str(params.get("text", "")))
        elif command in (
            "__log",
            "__warn",
            "__topic_list",
            "__request",
            "__response",
            "__remove_publisher",
            "__remove_subscriber",
            "__remove_ros_service",
            "__remove_unity_service",
            "__ros_service",
            "__unity_service",
        ):
            pass  # accepted, nothing to do for a read-only mirror
        else:
            self._record_error(f"unsupported system command {command}")

    def _handle_handshake(self, body: str) -> None:
        try:
            handshake = protocol.parse_handshake(body)
        except protocol.ProtocolError as exc:
            self._record_error(f"invalid handshake: {exc}")
            return
        with self._lock:
            self._handshake = handshake

    def _handle_registration(self, params: dict, publishing: bool) -> None:
        topic = params.get("topic")
        if not isinstance(topic, str) or topic == "":
            self._record_error("registration without a topic")
            return
        msg_type = normalise_msg_type(params.get("message_name", ""))
        key = topics.base(topic)
        with self._lock:
            if msg_type:
                self._topic_types[key] = msg_type
                self._announced[key] = msg_type
            else:
                self._announced.setdefault(key, "")
            if publishing:
                self._published[key] = msg_type
            else:
                self._subscriptions[key] = topic
            self._counts.setdefault(key, 0)

    def _handle_topic_message(self, destination: str, payload: bytes) -> None:
        key = topics.base(destination)
        with self._lock:
            msg_type = self._type_for_locked(key)
            self._raw[key] = bytes(payload)
            self._counts[key] = self._counts.get(key, 0) + 1

        if not msg_type:
            self._record_error(f"unknown topic type for {destination}")
            return
        try:
            message = codec.decode(msg_type, payload)
        except codec.CodecError as exc:
            self._record_error(f"decode failed on {key}: {exc}")
            return

        with self._lock:
            self._latest[key] = message
        self._notify_message(key, bytes(payload), msg_type)
        self._notify_state()

    def _type_for_locked(self, key: str) -> str | None:
        found = _resolve_key(key, self._topic_types)
        return None if found is None else self._topic_types[found]

    def _notify_state(self) -> None:
        callback = self._on_state
        if callback is None:
            return
        try:
            callback(self.snapshot())
        except Exception as exc:
            self._record_error(f"on_state callback failed: {exc}")

    def _notify_message(self, key: str, payload: bytes, msg_type: str) -> None:
        """Hand one message to every listener, outside the lock and outside the read path's error.

        The payload goes over as the bytes Unity sent rather than as a re-encoding of the decoded
        message: the two are the same bytes, and a round trip that is not needed is a round trip that
        can differ. A listener that raises is recorded and skipped - a mirror that broke must not stop
        the session it mirrors.
        """
        with self._lock:
            listeners = list(self._message_listeners)
        for listener in listeners:
            try:
                listener(key, payload, msg_type)
            except Exception as exc:
                self._record_error(f"message listener failed on {key}: {exc}")

    def _record_error(self, message: str) -> None:
        with self._lock:
            self._last_error = str(message)
