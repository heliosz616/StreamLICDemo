"""Deterministic learned-logit to finite integer-CDF conversion."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import struct
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch

from streamlic.model.constants import SYMBOL_DOMAIN_MAX, SYMBOL_DOMAIN_MIN


_MAGIC = b"SLCDF001"
_HEADER = struct.Struct("<8sHBBHH32s")
_TABLE_HEADER = struct.Struct("<hH")


@dataclass(frozen=True)
class CDFTable:
    offset: int
    cdf: Tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.cdf) < 2:
            raise ValueError("a CDF needs at least one symbol")
        if self.cdf[0] != 0:
            raise ValueError("CDF must start at zero")
        if any(b <= a for a, b in zip(self.cdf, self.cdf[1:])):
            raise ValueError("every retained symbol must have nonzero frequency")

    @property
    def symbol_count(self) -> int:
        return len(self.cdf) - 1

    @property
    def minimum(self) -> int:
        return self.offset

    @property
    def maximum(self) -> int:
        return self.offset + self.symbol_count - 1

    @property
    def total(self) -> int:
        return self.cdf[-1]

    def interval(self, symbol: int) -> Tuple[int, int, int]:
        index = int(symbol) - self.offset
        if index < 0 or index >= self.symbol_count:
            raise ValueError("symbol {} outside [{}, {}]".format(
                symbol, self.minimum, self.maximum))
        return self.cdf[index], self.cdf[index + 1], self.total


@dataclass(frozen=True)
class CDFBundle:
    precision: int
    model_sha256: str
    y: Tuple[CDFTable, ...]
    z: Tuple[CDFTable, ...]

    def __post_init__(self) -> None:
        total = 1 << self.precision
        if not (1 <= self.precision <= 24):
            raise ValueError("CDF precision must be in [1,24]")
        for table in self.y + self.z:
            if table.total != total:
                raise ValueError("CDF total does not match precision")

    @property
    def y_bounds(self) -> Tuple[Tuple[int, int], ...]:
        return tuple((table.minimum, table.maximum) for table in self.y)

    @property
    def z_bounds(self) -> Tuple[Tuple[int, int], ...]:
        return tuple((table.minimum, table.maximum) for table in self.z)

    def save(self, path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = bytes.fromhex(self.model_sha256)
        if len(digest) != 32:
            raise ValueError("model_sha256 must contain 32 bytes")
        with target.open("wb") as stream:
            stream.write(_HEADER.pack(
                _MAGIC, 1, self.precision, 0, len(self.y), len(self.z), digest))
            for table in self.y + self.z:
                stream.write(_TABLE_HEADER.pack(table.offset, table.symbol_count))
                stream.write(struct.pack("<{}I".format(len(table.cdf)), *table.cdf))

    @classmethod
    def load(cls, path) -> "CDFBundle":
        with Path(path).open("rb") as stream:
            header = stream.read(_HEADER.size)
            if len(header) != _HEADER.size:
                raise ValueError("truncated CDF header")
            magic, version, precision, reserved, y_count, z_count, digest = _HEADER.unpack(header)
            if magic != _MAGIC or version != 1 or reserved != 0:
                raise ValueError("unsupported CDF bundle")
            tables: List[CDFTable] = []
            for _ in range(y_count + z_count):
                raw = stream.read(_TABLE_HEADER.size)
                if len(raw) != _TABLE_HEADER.size:
                    raise ValueError("truncated CDF table header")
                offset, count = _TABLE_HEADER.unpack(raw)
                raw_cdf = stream.read(4 * (count + 1))
                if len(raw_cdf) != 4 * (count + 1):
                    raise ValueError("truncated CDF table")
                values = struct.unpack("<{}I".format(count + 1), raw_cdf)
                tables.append(CDFTable(offset, tuple(values)))
            if stream.read(1):
                raise ValueError("trailing bytes in CDF bundle")
        return cls(precision, digest.hex(), tuple(tables[:y_count]), tuple(tables[y_count:]))

    def write_summary(self, path) -> None:
        def describe(tables: Sequence[CDFTable]):
            return [{"offset": t.offset, "length": t.symbol_count,
                     "minimum": t.minimum, "maximum": t.maximum}
                    for t in tables]
        Path(path).write_text(json.dumps({
            "format": "SLCDF001", "precision": self.precision,
            "total": 1 << self.precision, "model_sha256": self.model_sha256,
            "y": describe(self.y), "z": describe(self.z),
        }, indent=2) + "\n")


def _softmax64(logits: np.ndarray) -> np.ndarray:
    shifted = logits.astype(np.float64) - np.max(logits)
    values = np.exp(shifted)
    return values / values.sum()


def quantize_logits(logits: np.ndarray, precision: int) -> CDFTable:
    """Quantize one 256-bin learned PDF into a cropped positive-frequency CDF.

    Model bin 255 corresponds to residual +128, which is unreachable under the
    deployed signed-int8 contract. It is removed and the reachable 255 bins are
    renormalized. The first/last nonzero intervals of a raw rounded cumulative
    CDF define the truncation support. Retained probabilities are then
    renormalized and assigned at least one count with deterministic
    largest-remainder allocation.
    """
    if logits.shape != (256,):
        raise ValueError("expected exactly 256 logits")
    if precision < 8 or precision > 24:
        raise ValueError("supported CDF precision is 8..24")
    total = 1 << precision
    probabilities = _softmax64(logits)[:255]
    probabilities /= probabilities.sum()

    raw_cumulative = np.rint(np.cumsum(probabilities) * total).astype(np.int64)
    raw_cumulative[-1] = total
    raw_frequencies = np.diff(np.concatenate(([0], raw_cumulative)))
    nonzero = np.flatnonzero(raw_frequencies > 0)
    if not len(nonzero):
        raise ValueError("quantized PDF has no support")
    first, last = int(nonzero[0]), int(nonzero[-1])
    retained = probabilities[first:last + 1]
    retained /= retained.sum()

    count = len(retained)
    remaining = total - count
    if remaining < 0:
        raise ValueError("CDF precision cannot represent retained support")
    desired = retained * remaining
    extra = np.floor(desired).astype(np.int64)
    frequencies = extra + 1
    remainder = total - int(frequencies.sum())
    if remainder:
        fractions = desired - extra
        # lexsort uses the final key as primary: descending fractional part,
        # then ascending symbol index gives stable, cross-platform tie breaking.
        order = np.lexsort((np.arange(count), -fractions))
        frequencies[order[:remainder]] += 1
    cdf = np.concatenate(([0], np.cumsum(frequencies))).astype(np.int64)
    if int(cdf[-1]) != total or np.any(np.diff(cdf) <= 0):
        raise AssertionError("invalid positive-frequency CDF")
    return CDFTable(SYMBOL_DOMAIN_MIN + first, tuple(int(v) for v in cdf))


def build_bundle(y_logits: np.ndarray, z_logits: np.ndarray, precision: int,
                 model_sha256: str) -> CDFBundle:
    if y_logits.shape != (128, 256) or z_logits.shape != (96, 256):
        raise ValueError("unexpected StreamLIC logits shape")
    y = tuple(quantize_logits(row, precision) for row in y_logits)
    z = tuple(quantize_logits(row, precision) for row in z_logits)
    return CDFBundle(precision, model_sha256, y, z)


def build_bundle_from_archive(archive_path, precision: int = 24) -> CDFBundle:
    archive_path = Path(archive_path)
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    parameters = torch.load(str(archive_path), map_location="cpu", weights_only=False)
    pdf = parameters["pdf_lists"]
    return build_bundle(
        pdf["y_pdf_logits"].detach().cpu().numpy(),
        pdf["z_pdf_logits"].detach().cpu().numpy(), precision, digest)
