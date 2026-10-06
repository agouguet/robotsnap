"""Public surface of the RobotSNAP Unity bridge.

Re-exports the topics (:mod:`~robotsnap.topics`), the framing helpers
(:mod:`~robotsnap.bridge.protocol`), the CDR codecs
(:mod:`~robotsnap.bridge.codec`) and the state model
(:mod:`~robotsnap.bridge.state`).
"""

from robotsnap import topics
from robotsnap.bridge import codec, protocol, state
from robotsnap.bridge.codec import (
    TYPESTORE,
    CodecError,
    decode,
    decode_agents_json,
    decode_bool,
    decode_laserscan,
    decode_occupancy_grid,
    decode_odometry,
    decode_string,
    encode,
    encode_twist,
)
from robotsnap.bridge.protocol import (
    SYSCOMMAND_PREFIX,
    ProtocolError,
    encode_frame,
    encode_int32,
    encode_string,
    is_syscommand,
    parse_handshake,
    read_exact,
    read_frame,
    read_int32,
    read_string,
    serialize_command,
)
from robotsnap.bridge.state import (
    AgentState,
    HumanState,
    LaserScanState,
    PointState,
    PoseState,
    RobotSNAPState,
    RobotState,
    SimulationState,
    build_state,
    to_world_frame,
)

__all__ = [
    "codec",
    "protocol",
    "state",
    "topics",
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
    "AgentState",
    "HumanState",
    "PointState",
    "PoseState",
    "RobotState",
    "LaserScanState",
    "SimulationState",
    "RobotSNAPState",
    "build_state",
    "to_world_frame",
]
