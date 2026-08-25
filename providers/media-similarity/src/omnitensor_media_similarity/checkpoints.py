"""Durable completed-file checkpoints for interrupted media scans."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from .fingerprints import MediaFingerprint, MediaQuality

CHECKPOINT_FILENAME = "media-similarity-checkpoints.sqlite3"
_MODALITIES = frozenset({"audio", "video", "audio-video"})
_PAYLOAD_FIELDS = frozenset(
    {"version", "sha256", "modality", "durationMs", "visual", "audio", "quality"}
)
_QUALITY_FIELDS = frozenset(asdict(MediaQuality()))


def selection_key(relative_paths: Sequence[str]) -> str:
    """Stable private identity for one ordered selection, never its absolute paths."""
    encoded = json.dumps(
        list(relative_paths), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class FingerprintCheckpointStore:
    """One transactional fingerprint row per completed file in an interrupted scan."""

    def __init__(self, state_path: Path) -> None:
        root = Path(state_path)
        if not root.is_absolute() or not root.is_dir():
            raise ValueError("checkpoint state path must be an absolute directory")
        self.path = root / CHECKPOINT_FILENAME

    def load(
        self, selection: str, relative_path: str, digest: str
    ) -> MediaFingerprint | None:
        if not self.path.is_file():
            return None
        try:
            with self._connect() as database:
                row = database.execute(
                    "SELECT payload FROM fingerprints "
                    "WHERE selection_key = ? AND relative_path = ? AND sha256 = ?",
                    (selection, relative_path, digest),
                ).fetchone()
            return None if row is None else _decode(row[0], relative_path, digest)
        except (OSError, sqlite3.Error, TypeError, ValueError, json.JSONDecodeError):
            return None

    def save(self, selection: str, fingerprint: MediaFingerprint) -> None:
        try:
            payload = _encode(fingerprint)
            with self._connect() as database:
                database.execute(
                    "INSERT OR REPLACE INTO fingerprints "
                    "(selection_key, relative_path, sha256, payload) VALUES (?, ?, ?, ?)",
                    (selection, fingerprint.relative_path, fingerprint.sha256, payload),
                )
        except (OSError, sqlite3.Error, TypeError, ValueError):
            # Checkpoints improve recovery; an unwritable cache must not change
            # the deterministic answer the media scan can still compute.
            return

    def clear(self, selection: str) -> None:
        if not self.path.is_file():
            return
        try:
            with self._connect() as database:
                database.execute(
                    "DELETE FROM fingerprints WHERE selection_key = ?", (selection,)
                )
        except (OSError, sqlite3.Error):
            return

    def _connect(self) -> sqlite3.Connection:
        database = sqlite3.connect(self.path)
        try:
            database.execute(
                "CREATE TABLE IF NOT EXISTS fingerprints ("
                "selection_key TEXT NOT NULL, relative_path TEXT NOT NULL, "
                "sha256 TEXT NOT NULL, payload TEXT NOT NULL, "
                "PRIMARY KEY (selection_key, relative_path))"
            )
            database.commit()
            os.chmod(self.path, 0o600)
            return database
        except BaseException:
            database.close()
            raise


def _encode(fingerprint: MediaFingerprint) -> str:
    document = {
        "version": 1,
        "sha256": fingerprint.sha256,
        "modality": fingerprint.modality,
        "durationMs": fingerprint.duration_ms,
        "visual": list(fingerprint.visual),
        "audio": list(fingerprint.audio),
        "quality": asdict(fingerprint.quality),
    }
    return json.dumps(document, separators=(",", ":"))


def _decode(payload: object, relative_path: str, digest: str) -> MediaFingerprint:
    document = json.loads(payload)
    if not isinstance(document, dict) or set(document) != _PAYLOAD_FIELDS:
        raise ValueError("checkpoint fields are invalid")
    if document["version"] != 1 or document["sha256"] != digest:
        raise ValueError("checkpoint identity is invalid")
    modality = document["modality"]
    duration = document["durationMs"]
    if modality not in _MODALITIES or not _optional_positive_integer(duration):
        raise ValueError("checkpoint media metadata is invalid")
    visual = _hashes(document["visual"])
    audio = _hashes(document["audio"])
    if (modality == "video" and (not visual or audio)) or (
        modality == "audio" and (visual or not audio)
    ):
        raise ValueError("checkpoint modality is invalid")
    if modality == "audio-video" and (not visual or not audio):
        raise ValueError("checkpoint modality is invalid")
    quality = _quality(document["quality"])
    return MediaFingerprint(
        relative_path,
        Path(relative_path).name,
        digest,
        modality,
        duration,
        visual,
        audio,
        quality,
    )


def _hashes(value: object) -> tuple[int, ...]:
    if not isinstance(value, list) or not all(
        type(item) is int and 0 <= item < 1 << 64 for item in value
    ):
        raise ValueError("checkpoint hashes are invalid")
    return tuple(value)


def _quality(value: object) -> MediaQuality:
    if not isinstance(value, dict) or set(value) != _QUALITY_FIELDS:
        raise ValueError("checkpoint quality is invalid")
    integers = (
        "width",
        "height",
        "video_bitrate",
        "audio_bitrate",
        "audio_sample_rate",
        "audio_channels",
    )
    if not all(_optional_positive_integer(value[field]) for field in integers):
        raise ValueError("checkpoint quality is invalid")
    lossless = value["lossless_audio"]
    if lossless is not None and type(lossless) is not bool:
        raise ValueError("checkpoint quality is invalid")
    return MediaQuality(**value)


def _optional_positive_integer(value: object) -> bool:
    return value is None or (type(value) is int and value > 0)


__all__ = ["FingerprintCheckpointStore", "selection_key"]
