"""CDR encode/decode helpers for the messages exchanged with Unity.

Message payloads are CDR encoded with the ``rosbags`` ROS2 Humble typestore, so
only the types the simulator actually publishes have a dedicated decoder here;
anything else goes through :func:`decode` and :func:`encode` by type name. The
type names come from :mod:`robotsnap.topics`, the one place that lists them.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from rosbags.typesys import Stores, get_typestore

from robotsnap import topics

__all__ = [
    "TYPESTORE",
    "CodecError",
    "decode",
    "encode",
    "decode_odometry",
    "decode_laserscan",
    "decode_occupancy_grid",
    "decode_bool",
    "decode_string",
    "encode_twist",
    "decode_agents_json",
    "json_object",
    "as_bool",
]

#: Typestore used for every message that crosses the bridge.
TYPESTORE = get_typestore(Stores.ROS2_HUMBLE)

#: Spellings of "true" a JSON body may use, Unity's serializer included.
_TRUE_STRINGS = frozenset({"true", "1", "yes"})


class CodecError(RuntimeError):
    """Raised when a payload cannot be encoded or decoded."""


def decode(msg_type: str, raw: bytes):
    """Decode CDR bytes into a typestore message instance."""
    try:
        return TYPESTORE.deserialize_cdr(bytes(raw), msg_type)
    except Exception as exc:  # rosbags raises plain KeyError/Exception subclasses
        raise CodecError(f"failed to decode {msg_type}: {exc}") from exc


def encode(msg_type: str, msg) -> bytes:
    """Encode a typestore message instance into CDR bytes."""
    try:
        return bytes(TYPESTORE.serialize_cdr(msg, msg_type))
    except Exception as exc:
        raise CodecError(f"failed to encode {msg_type}: {exc}") from exc


def decode_odometry(raw):
    """Decode a ``nav_msgs/msg/Odometry`` payload."""
    return decode(topics.ODOMETRY_TYPE, raw)


def decode_laserscan(raw):
    """Decode a ``sensor_msgs/msg/LaserScan`` payload."""
    return decode(topics.LASER_SCAN_TYPE, raw)


def decode_occupancy_grid(raw):
    """Decode a ``nav_msgs/msg/OccupancyGrid`` payload."""
    return decode(topics.OCCUPANCY_GRID_TYPE, raw)


def decode_bool(raw):
    """Decode a ``std_msgs/msg/Bool`` payload."""
    return decode(topics.BOOL_TYPE, raw)


def decode_string(raw):
    """Decode a ``std_msgs/msg/String`` payload."""
    return decode(topics.STRING_TYPE, raw)


def encode_twist(linear_x: float, angular_z: float) -> bytes:
    """Encode a ``/cmd_vel`` Twist with the given forward and yaw commands."""
    vector = TYPESTORE.types["geometry_msgs/msg/Vector3"]
    twist = TYPESTORE.types[topics.TWIST_TYPE](
        linear=vector(x=float(linear_x), y=0.0, z=0.0),
        angular=vector(x=0.0, y=0.0, z=float(angular_z)),
    )
    return encode(topics.TWIST_TYPE, twist)


def decode_agents_json(payload) -> dict[str, Any]:
    """Parse a ``/simulation/agents`` body.

    The body is ``{"agents": [...], "frame": "robot"}``: every agent the current
    session simulates, in the **robot** frame. Each entry is flat, with ``id``,
    ``x``, ``y``, ``z``, ``vx``, ``vy`` and ``vz``, plus ``visible``.

    ``payload`` is the raw ``std_msgs/String`` body, already-parsed JSON
    included. Returns ``{"agents": [dict, ...], "frame": str}`` with the agent
    fields coerced to their Python types, so a caller never has to index the raw
    body itself. ``frame`` defaults to ``"robot"``, the only frame the contract
    defines. Raises :class:`CodecError` when the body does not have that shape.
    """
    body = _as_json_object(payload)
    entries = body.get("agents")
    if not isinstance(entries, list):
        raise CodecError("agents payload must carry an 'agents' array")

    agents: list[dict[str, Any]] = []
    for index, item in enumerate(entries):
        if not isinstance(item, Mapping):
            raise CodecError(f"agent #{index} must be a JSON object")
        try:
            agent = {
                "id": str(item["id"]),
                "x": float(item["x"]),
                "y": float(item["y"]),
                "z": float(item["z"]),
                "vx": float(item["vx"]),
                "vy": float(item["vy"]),
                "vz": float(item["vz"]),
                "visible": as_bool(item["visible"]),
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise CodecError(f"agent #{index} has an invalid schema: {exc}") from exc
        agents.append(agent)

    return {"agents": agents, "frame": str(body.get("frame", "robot"))}


def json_object(payload) -> Mapping[str, Any] | None:
    """Read a ``std_msgs/String`` body as a JSON object, or ``None``.

    The one reader for the JSON bodies of the contract, so a decoded
    ``std_msgs/String``, its raw ``data`` string, an already-parsed mapping and
    a bytes payload are all accepted the same way. ``None`` means "not an
    object": a missing value, text that is not JSON, or JSON that is not an
    object. Callers that treat that as an error wrap this with their own
    exception, which is what :func:`decode_agents_json` does.
    """
    if payload is None:
        return None
    if isinstance(payload, Mapping):
        return payload
    text = getattr(payload, "data", payload)
    if isinstance(text, (bytes, bytearray)):
        text = bytes(text).decode("utf-8", "replace")
    if not isinstance(text, str):
        return None
    try:
        body = json.loads(text)
    except ValueError:
        return None
    return body if isinstance(body, Mapping) else None


def as_bool(value: Any) -> bool:
    """Coerce a JSON value to a bool, tolerating the string forms Unity may send."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in _TRUE_STRINGS
    return bool(value)


def _as_json_object(payload) -> Mapping[str, Any]:
    """Read a JSON body strictly, raising :class:`CodecError` when it is not one."""
    body = json_object(payload)
    if body is None:
        raise CodecError("payload is not a JSON object")
    return body
