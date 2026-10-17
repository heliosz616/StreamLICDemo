"""Compact single-frame StreamLIC container.

The file contains an 18-byte header followed by the z payload and exactly four
independent y arithmetic payloads. All offsets are absolute little-endian file
offsets. The final y payload ends at EOF.
"""

from dataclasses import dataclass
import struct
from typing import Tuple


TILE_SIZE = 128
ENTROPY_STREAMS = 4
HEADER_SIZE = 18
_OFFSETS = struct.Struct("<4I")


def _validate_dimensions(width: int, height: int) -> None:
    if (width <= 0 or height <= 0 or width % TILE_SIZE or height % TILE_SIZE or
            width // TILE_SIZE > 0xFF or height // TILE_SIZE > 0xFF):
        raise ValueError(
            "StreamLIC dimensions must be nonzero multiples of 128 representable as tile bytes")


@dataclass(frozen=True)
class FrameBitstream:
    width: int
    height: int
    z_payload: bytes
    y_payloads: Tuple[bytes, bytes, bytes, bytes]

    def to_bytes(self) -> bytes:
        _validate_dimensions(self.width, self.height)
        if not self.z_payload:
            raise ValueError("StreamLIC z payload cannot be empty")
        if len(self.y_payloads) != ENTROPY_STREAMS or any(
                not payload for payload in self.y_payloads):
            raise ValueError("StreamLIC requires exactly four nonempty y payloads")

        offsets = []
        position = HEADER_SIZE + len(self.z_payload)
        for payload in self.y_payloads:
            if position > 0xFFFFFFFF:
                raise ValueError("StreamLIC frame exceeds the 32-bit offset range")
            offsets.append(position)
            position += len(payload)
        if position > 0xFFFFFFFF:
            raise ValueError("StreamLIC frame exceeds the 32-bit offset range")
        header = bytes((self.width // TILE_SIZE, self.height // TILE_SIZE))
        header += _OFFSETS.pack(*offsets)
        return header + self.z_payload + b"".join(self.y_payloads)

    @classmethod
    def from_bytes(cls, data: bytes) -> "FrameBitstream":
        if len(data) < HEADER_SIZE:
            raise ValueError("truncated StreamLIC frame header")
        if len(data) > 0xFFFFFFFF:
            raise ValueError("StreamLIC frame exceeds the 32-bit offset range")
        width = data[0] * TILE_SIZE
        height = data[1] * TILE_SIZE
        _validate_dimensions(width, height)
        offsets = _OFFSETS.unpack_from(data, 2)
        if not (HEADER_SIZE < offsets[0] < offsets[1] < offsets[2] <
                offsets[3] < len(data)):
            raise ValueError("invalid StreamLIC payload offsets")
        z_payload = data[HEADER_SIZE:offsets[0]]
        ends = offsets[1:] + (len(data),)
        y_payloads = tuple(data[start:end] for start, end in zip(offsets, ends))
        return cls(width, height, z_payload, y_payloads)
