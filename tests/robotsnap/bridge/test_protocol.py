"""Tests for the Unity <-> Python framing protocol."""

import json
import socket
import struct

import pytest

from robotsnap.bridge import protocol


@pytest.fixture
def sock_pair():
    left, right = socket.socketpair()
    try:
        yield left, right
    finally:
        left.close()
        right.close()


def test_encode_frame_roundtrip(sock_pair):
    left, right = sock_pair
    payload = b"\x01\x02\x03\xff"
    left.sendall(protocol.encode_frame("/cmd_vel", payload))
    destination, data = protocol.read_frame(right)
    assert destination == "/cmd_vel"
    assert data == payload


def test_serialize_command_byte_layout():
    command = "__handshake"
    params = {"version": "1.0", "metadata": {}}
    raw = protocol.serialize_command(command, params)

    (cmd_len,) = struct.unpack_from("<I", raw, 0)
    assert cmd_len == len(command.encode("utf-8"))
    assert raw[4 : 4 + cmd_len] == command.encode("utf-8")

    json_offset = 4 + cmd_len
    (json_len,) = struct.unpack_from("<I", raw, json_offset)
    body = raw[json_offset + 4 : json_offset + 4 + json_len]
    assert json_len == len(body)
    assert json.loads(body.decode("utf-8")) == params
    assert len(raw) == json_offset + 4 + json_len


def test_read_string_strips_trailing_nuls(sock_pair):
    left, right = sock_pair
    left.sendall(protocol.encode_string("agent\x00\x00"))
    assert protocol.read_string(right) == "agent"


def test_read_string_roundtrip_unicode(sock_pair):
    left, right = sock_pair
    left.sendall(protocol.encode_string("/agents/global"))
    assert protocol.read_string(right) == "/agents/global"


def test_read_int32_is_little_endian(sock_pair):
    left, right = sock_pair
    left.sendall(struct.pack("<I", 258))
    assert protocol.read_int32(right) == 258


def test_parse_handshake_happy_path():
    payload = json.dumps(
        {"version": "2.0.1", "metadata": {"scene": "corridor", "agents": 3}}
    ).encode("utf-8")
    handshake = protocol.parse_handshake(payload)
    assert handshake["version"] == "2.0.1"
    assert handshake["metadata"] == {"scene": "corridor", "agents": 3}


def test_parse_handshake_accepts_string_metadata():
    payload = json.dumps(
        {"version": "1.0", "metadata": json.dumps({"scene": "plaza"})}
    ).encode("utf-8")
    handshake = protocol.parse_handshake(payload)
    assert handshake["metadata"] == {"scene": "plaza"}


def test_parse_handshake_bad_json():
    with pytest.raises(protocol.ProtocolError):
        protocol.parse_handshake(b"{not json")


def test_is_syscommand():
    assert protocol.is_syscommand("__handshake")
    assert protocol.is_syscommand("__publish")
    assert not protocol.is_syscommand("/odom")
    assert not protocol.is_syscommand("odom")
    assert not protocol.is_syscommand("")


def test_read_exact_raises_on_truncated_stream(sock_pair):
    left, right = sock_pair
    left.sendall(b"\x01\x02")
    left.shutdown(socket.SHUT_WR)
    with pytest.raises(protocol.ProtocolError):
        protocol.read_exact(right, 4)
