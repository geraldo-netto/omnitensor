"""Read a GGUF file's declared shape without loading it.

Deciding whether a model fits a card needs four numbers — how many blocks it
has, how many key/value heads, and how wide a key and a value are — and all
four are written in the file's own metadata header. Loading the model to find
out costs the very memory the question is about, and fails for exactly the
candidates most worth asking about.

So this reads the header and stops. It never touches tensor data, refuses a
file whose header is not plausibly a header, and is bounded everywhere a
length is read from the file itself: this parses bytes that arrived from
somewhere else, and a length field is the first thing to lie.

The format is the published GGUF v2/v3 layout: a magic, a version, two counts,
then that many key/value pairs, each a length-prefixed key, a type tag, and a
value of that type.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

MAGIC = b"GGUF"
SUPPORTED_VERSIONS = (2, 3)

# Nothing here needs more than the front of the file, and a header this large
# is a file that is not what it claims.
# Token vocabularies and merge tables live in this header, and for a modern
# tokenizer they are megabytes on their own.
MAX_HEADER_BYTES = 48 * 1024 * 1024
MAX_KEY_BYTES = 1024
MAX_STRING_BYTES = 64 * 1024
MAX_ENTRIES = 4096
MAX_ARRAY_ITEMS = 65_536

_UINT8, _INT8, _UINT16, _INT16, _UINT32, _INT32 = 0, 1, 2, 3, 4, 5
_FLOAT32, _BOOL, _STRING, _ARRAY, _UINT64, _INT64, _FLOAT64 = 6, 7, 8, 9, 10, 11, 12

_FIXED = {
    _UINT8: ("<B", 1),
    _INT8: ("<b", 1),
    _UINT16: ("<H", 2),
    _INT16: ("<h", 2),
    _UINT32: ("<I", 4),
    _INT32: ("<i", 4),
    _FLOAT32: ("<f", 4),
    _BOOL: ("<?", 1),
    _UINT64: ("<Q", 8),
    _INT64: ("<q", 8),
    _FLOAT64: ("<d", 8),
}


@dataclass(frozen=True, slots=True)
class _Skipped:
    """An array too long to be worth holding: its shape, not its contents."""

    kind: int
    count: int


class GgufError(ValueError):
    """This file is not a GGUF file, or not one this can read."""


@dataclass(frozen=True, slots=True)
class ModelShape:
    """What a model's own header says about its size.

    ``file_bytes`` is what the weights occupy on a card once fully offloaded —
    the whole file, since nothing here runs a model partly on the CPU.
    """

    name: str
    architecture: str
    file_bytes: int
    blocks: int
    kv_heads: int
    key_length: int
    value_length: int

    @property
    def kv_bytes_per_token(self) -> float:
        """Cache bytes one token costs at one byte per element.

        Multiplied by the cache precision and the context length by
        :mod:`omnitensor.fit`; kept separate because this half is the model's
        own fact and the other half is a choice somebody made.
        """
        return float(self.blocks * self.kv_heads * (self.key_length + self.value_length))


def read_shape(path: Path | str) -> ModelShape:
    """The shape declared by the GGUF file at ``path``."""
    path = Path(path)
    metadata = read_metadata(path)
    architecture = str(metadata.get("general.architecture", "")).strip()
    if not architecture:
        raise GgufError("GGUF file declares no architecture")

    def declared(suffix: str) -> int:
        value = metadata.get(f"{architecture}.{suffix}")
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise GgufError(f"GGUF file declares no usable {suffix}")
        return value

    heads = declared("attention.head_count")
    embedding = declared("embedding_length")
    kv_heads = metadata.get(f"{architecture}.attention.head_count_kv")
    if isinstance(kv_heads, bool) or not isinstance(kv_heads, int) or kv_heads < 1:
        # Older exports omit it, and the value it omits is "the same as the
        # attention head count" — multi-head attention before grouping existed.
        kv_heads = heads
    head_dimension = embedding // heads
    return ModelShape(
        name=str(metadata.get("general.name", path.stem))[:120],
        architecture=architecture,
        file_bytes=path.stat().st_size,
        blocks=declared("block_count"),
        kv_heads=kv_heads,
        key_length=_length(metadata, architecture, "key_length", head_dimension),
        value_length=_length(metadata, architecture, "value_length", head_dimension),
    )


def _length(metadata: dict, architecture: str, suffix: str, fallback: int) -> int:
    value = metadata.get(f"{architecture}.attention.{suffix}")
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        # The published default: a key or value is as wide as one head.
        return fallback
    return value


def read_metadata(path: Path | str) -> dict:
    """Every metadata pair in the file's header, by key."""
    with Path(path).open("rb") as handle:
        header = handle.read(MAX_HEADER_BYTES)
    return _parse(header)


def _parse(header: bytes) -> dict:
    reader = _Reader(header)
    if reader.take(4) != MAGIC:
        raise GgufError("file does not begin with the GGUF magic")
    version = reader.unpack("<I", 4)
    if version not in SUPPORTED_VERSIONS:
        raise GgufError(f"unsupported GGUF version: {version}")
    reader.unpack("<Q", 8)  # tensor count, not needed here
    entries = reader.unpack("<Q", 8)
    if entries > MAX_ENTRIES:
        raise GgufError("GGUF header declares implausibly many entries")
    metadata: dict = {}
    for _index in range(entries):
        key = reader.string(MAX_KEY_BYTES)
        metadata[key] = reader.value()
    return metadata


class _Reader:
    """A cursor over the header that refuses to read past what it was given."""

    __slots__ = ("_at", "_data")

    def __init__(self, data: bytes):
        self._data = data
        self._at = 0

    def take(self, count: int) -> bytes:
        if count < 0 or self._at + count > len(self._data):
            raise GgufError("GGUF header ends inside a value")
        chunk = self._data[self._at : self._at + count]
        self._at += count
        return chunk

    def unpack(self, layout: str, size: int):
        return struct.unpack(layout, self.take(size))[0]

    def string(self, limit: int) -> str:
        length = self.unpack("<Q", 8)
        if length > limit:
            raise GgufError("GGUF header holds an implausibly long string")
        return self.take(length).decode("utf-8", "replace")

    def _skip(self, kind: int, count: int) -> None:
        if kind == _STRING:
            for _index in range(count):
                self.take(self.unpack("<Q", 8))
            return
        if kind == _ARRAY:
            for _index in range(count):
                self._array()
            return
        fixed = _FIXED.get(kind)
        if fixed is None:
            raise GgufError(f"unsupported GGUF value type: {kind}")
        self.take(fixed[1] * count)

    def value(self):
        kind = self.unpack("<I", 4)
        if kind == _STRING:
            return self.string(MAX_STRING_BYTES)
        if kind == _ARRAY:
            return self._array()
        fixed = _FIXED.get(kind)
        if fixed is None:
            raise GgufError(f"unsupported GGUF value type: {kind}")
        return self.unpack(*fixed)

    def _array(self):
        kind = self.unpack("<I", 4)
        count = self.unpack("<Q", 8)
        if count > MAX_ARRAY_ITEMS:
            # A token vocabulary, in every real file: hundreds of thousands of
            # strings that no question here is about. Stepped over rather than
            # refused — the entries after it are the ones being read for — and
            # reported as its shape so a caller sees that something was there.
            self._skip(kind, count)
            return _Skipped(kind, count)
        if kind == _STRING:
            return [self.string(MAX_STRING_BYTES) for _index in range(count)]
        if kind == _ARRAY:
            return [self._array() for _index in range(count)]
        fixed = _FIXED.get(kind)
        if fixed is None:
            raise GgufError(f"unsupported GGUF value type: {kind}")
        return [self.unpack(*fixed) for _index in range(count)]


__all__ = ["GgufError", "ModelShape", "read_metadata", "read_shape"]
