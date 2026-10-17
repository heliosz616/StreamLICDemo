"""Forward (FIFO) integer arithmetic coder.

This is deliberately independent from CompressAI's reverse-order rANS coder.
It uses the conventional E1/E2/E3 renormalization algorithm, a 32-bit coding
interval, and 64-bit-equivalent Python integer intermediates.
"""

from typing import Iterable, List, Sequence, Tuple

from .cdf import CDFTable


_STATE_BITS = 32
_FULL = 1 << _STATE_BITS
_MASK = _FULL - 1
_HALF = _FULL >> 1
_QUARTER = _HALF >> 1
_THREE_QUARTERS = _QUARTER * 3


class _BitWriter:
    def __init__(self) -> None:
        self.data = bytearray()
        self.current = 0
        self.used = 0

    def write(self, bit: int) -> None:
        self.current = (self.current << 1) | (bit & 1)
        self.used += 1
        if self.used == 8:
            self.data.append(self.current)
            self.current = 0
            self.used = 0

    def finish(self) -> bytes:
        if self.used:
            self.data.append(self.current << (8 - self.used))
        return bytes(self.data)


class _BitReader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.position = 0

    def read(self) -> int:
        if self.position >= len(self.data) * 8:
            # Arithmetic decoders conventionally extend a finite, terminated
            # payload with zero bits. The image dimensions determine how many
            # symbols the caller decodes.
            self.position += 1
            return 0
        byte = self.data[self.position >> 3]
        bit = (byte >> (7 - (self.position & 7))) & 1
        self.position += 1
        return bit


class ArithmeticEncoder:
    def __init__(self) -> None:
        self.low = 0
        self.high = _MASK
        self.pending = 0
        self.writer = _BitWriter()
        self.finished = False

    def _bit_plus_pending(self, bit: int) -> None:
        self.writer.write(bit)
        opposite = bit ^ 1
        for _ in range(self.pending):
            self.writer.write(opposite)
        self.pending = 0

    def encode_interval(self, cdf_low: int, cdf_high: int, total: int) -> None:
        if self.finished:
            raise RuntimeError("encoder is already finished")
        if not (0 <= cdf_low < cdf_high <= total):
            raise ValueError("invalid arithmetic interval")
        span = self.high - self.low + 1
        old_low = self.low
        self.high = old_low + (span * cdf_high // total) - 1
        self.low = old_low + (span * cdf_low // total)
        while True:
            if self.high < _HALF:
                self._bit_plus_pending(0)
            elif self.low >= _HALF:
                self._bit_plus_pending(1)
                self.low -= _HALF
                self.high -= _HALF
            elif self.low >= _QUARTER and self.high < _THREE_QUARTERS:
                self.pending += 1
                self.low -= _QUARTER
                self.high -= _QUARTER
            else:
                break
            self.low = (self.low << 1) & _MASK
            self.high = ((self.high << 1) & _MASK) | 1

    def encode(self, symbol: int, table: CDFTable) -> None:
        low, high, total = table.interval(symbol)
        self.encode_interval(low, high, total)

    def finish(self) -> bytes:
        if not self.finished:
            self.pending += 1
            self._bit_plus_pending(0 if self.low < _QUARTER else 1)
            self.finished = True
        return self.writer.finish()


class ArithmeticDecoder:
    def __init__(self, payload: bytes) -> None:
        self.low = 0
        self.high = _MASK
        self.reader = _BitReader(payload)
        self.code = 0
        self.last_renormalizations = 0
        self.total_renormalizations = 0
        for _ in range(_STATE_BITS):
            self.code = ((self.code << 1) & _MASK) | self.reader.read()

    def decode(self, table: CDFTable) -> int:
        self.last_renormalizations = 0
        total = table.total
        span = self.high - self.low + 1
        scaled = ((self.code - self.low + 1) * total - 1) // span
        # CDFs are short enough that binary search is preferable to a large
        # decoder lookup table and preserves cache locality on the host CPU.
        lo, hi = 0, table.symbol_count
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if table.cdf[mid] <= scaled:
                lo = mid
            else:
                hi = mid
        cdf_low, cdf_high = table.cdf[lo], table.cdf[lo + 1]
        old_low = self.low
        self.high = old_low + (span * cdf_high // total) - 1
        self.low = old_low + (span * cdf_low // total)
        while True:
            if self.high < _HALF:
                pass
            elif self.low >= _HALF:
                self.low -= _HALF
                self.high -= _HALF
                self.code -= _HALF
            elif self.low >= _QUARTER and self.high < _THREE_QUARTERS:
                self.low -= _QUARTER
                self.high -= _QUARTER
                self.code -= _QUARTER
            else:
                break
            self.low = (self.low << 1) & _MASK
            self.high = ((self.high << 1) & _MASK) | 1
            self.code = ((self.code << 1) & _MASK) | self.reader.read()
            self.last_renormalizations += 1
            self.total_renormalizations += 1
        return table.offset + lo


class InterleavedArithmeticEncoder:
    """Round-robin FIFO arithmetic streams preserving global token order."""

    def __init__(self, stream_count: int) -> None:
        if stream_count < 1:
            raise ValueError("arithmetic stream count must be positive")
        self.encoders = [ArithmeticEncoder() for _ in range(stream_count)]
        self.position = 0

    def encode(self, symbol: int, table: CDFTable) -> None:
        self.encoders[self.position % len(self.encoders)].encode(symbol, table)
        self.position += 1

    def finish(self) -> Tuple[bytes, ...]:
        return tuple(encoder.finish() for encoder in self.encoders)


class InterleavedArithmeticDecoder:
    """Inverse of :class:`InterleavedArithmeticEncoder`."""

    def __init__(self, payloads: Sequence[bytes]) -> None:
        if not payloads:
            raise ValueError("at least one arithmetic payload is required")
        self.decoders = [ArithmeticDecoder(payload) for payload in payloads]
        self.position = 0

    def decode(self, table: CDFTable) -> int:
        symbol = self.decoders[self.position % len(self.decoders)].decode(table)
        self.position += 1
        return symbol

    @property
    def last_renormalizations(self) -> int:
        if not self.position:
            return 0
        return self.decoders[(self.position - 1) % len(self.decoders)].last_renormalizations


def encode_symbols(symbols: Iterable[int], tables: Sequence[CDFTable],
                   indexes: Iterable[int]) -> bytes:
    encoder = ArithmeticEncoder()
    count = 0
    for symbol, index in zip(symbols, indexes):
        encoder.encode(int(symbol), tables[int(index)])
        count += 1
    return encoder.finish()


def decode_symbols(payload: bytes, tables: Sequence[CDFTable],
                   indexes: Iterable[int]) -> List[int]:
    decoder = ArithmeticDecoder(payload)
    return [decoder.decode(tables[int(index)]) for index in indexes]
