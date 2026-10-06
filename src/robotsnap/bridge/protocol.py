"""Wire protocol shared with the Unity ROS TCP Connector.

Unity connects as a TCP *client* to this Python endpoint. Every frame has the
same layout in both directions::

    [int32 len(destination)][destination utf8][int32 size][size bytes payload]

All integers are little-endian unsigned int32 and strings are UTF-8 prefixed by
their byte length. A system command carries its JSON body directly as the frame
payload.
"""

from __future__ import annotations

import json
import struct
from typing import Any

__all__ = [
    "ProtocolError",
    "SYSCOMMAND_PREFIX",
    "is_syscommand",
    "read_exact",
    "read_int32",
    "read_string",
    "read_frame",
    "encode_int32",
    "encode_string",
    "encode_frame",
    "serialize_command",
    "parse_handshake",
]

SYSCOMMAND_PREFIX = "__"

_INT32 = struct.Struct("<I")
_NUL = "\x00"


class ProtocolError(RuntimeError):
    """Raised when the byte stream does not match the expected framing."""


def is_syscommand(destination: str) -> bool:
    """Return True for system commands, which are prefixed with ``__``."""
    return destination.startswith(SYSCOMMAND_PREFIX)


def read_exact(sock, size: int) -> bytes:
    """Read exactly ``size`` bytes, raising ProtocolError on early EOF."""
    if size < 0:
        raise ProtocolError("read size must not be negative")
    buffer = bytearray()
    while len(buffer) < size:
        chunk = sock.recv(size - len(buffer))
        if not chunk:
            raise ProtocolError(
                f"connection closed after {len(buffer)} of {size} bytes"
            )
        buffer.extend(chunk)
    return bytes(buffer)


def read_int32(sock) -> int:
    """Read one little-endian unsigned int32."""
    return _INT32.unpack(read_exact(sock, _INT32.size))[0]


def read_string(sock) -> str:
    """Read a length-prefixed UTF-8 string, stripping trailing NULs."""
    size = read_int32(sock)
    raw = read_exact(sock, size)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProtocolError(f"invalid utf-8 string payload: {exc}") from exc
    return text.rstrip(_NUL)


def read_frame(sock) -> tuple[str, bytes]:
    """Read one full frame and return ``(destination, payload)``."""
    destination = read_string(sock)
    size = read_int32(sock)
    return destination, read_exact(sock, size)


def encode_int32(value: int) -> bytes:
    """Encode an unsigned int32 in little-endian order."""
    try:
        return _INT32.pack(value)
    except struct.error as exc:
        raise ProtocolError(f"value {value!r} is not an unsigned int32") from exc


def encode_string(text: str) -> bytes:
    """Encode a UTF-8 string prefixed by its byte length."""
    raw = text.encode("utf-8")
    return encode_int32(len(raw)) + raw


def encode_frame(destination: str, payload: bytes) -> bytes:
    """Encode one frame from a destination and a raw payload."""
    raw = bytes(payload)
    return encode_string(destination) + encode_int32(len(raw)) + raw


def serialize_command(command: str, params: dict) -> bytes:
    """Encode a system command as its length-prefixed JSON body.

    The result is the payload of a single command frame, not a full frame.
    """
    cmd_bytes = command.encode("utf-8")
    json_bytes = json.dumps(params).encode("utf-8")
    return (
        encode_int32(len(cmd_bytes))
        + cmd_bytes
        + encode_int32(len(json_bytes))
        + json_bytes
    )


def parse_handshake(payload: bytes) -> dict[str, Any]:
    """Parse a ``__handshake`` payload into ``{"version", "metadata"}``.

    ``metadata`` is returned as a dict. Unity sends it as a JSON string, but an
    already-decoded object is accepted too. Raises ProtocolError on bad JSON.
    """
    try:
        raw = payload.decode("utf-8") if isinstance(payload, (bytes, bytearray)) else payload
    except UnicodeDecodeError as exc:
        raise ProtocolError(f"handshake payload is not valid utf-8: {exc}") from exc
    try:
        data = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"handshake payload is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ProtocolError("handshake payload must be a JSON object")

    metadata = data.get("metadata", {})
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata) if metadata else {}
        except ValueError as exc:
            raise ProtocolError(f"handshake metadata is not valid JSON: {exc}") from exc
    if not isinstance(metadata, dict):
        raise ProtocolError("handshake metadata must be a JSON object")

    return {"version": str(data.get("version", "")), "metadata": metadata}
