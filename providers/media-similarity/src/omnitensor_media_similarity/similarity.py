"""Candidate indexing, temporal comparison, and near-duplicate grouping.

Media is fingerprinted once.  LSH-style hash bands select plausible neighbors;
only those compact sequences are compared.  A directory of N files therefore
does not decode N*(N-1)/2 pairs, and a large collision bucket compares every
member with one representative instead of expanding every pair.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .fingerprints import MediaFingerprint

VISUAL_BITS = 64
AUDIO_BITS = 32
VISUAL_MATCH_FLOOR = 0.72
AUDIO_MATCH_FLOOR = 0.68


@dataclass(frozen=True, slots=True)
class SequenceScore:
    score: float
    left_coverage: float
    right_coverage: float
    offset: int


@dataclass(frozen=True, slots=True)
class PairScore:
    score: float
    visual_score: float | None
    audio_score: float | None
    left_coverage: float
    right_coverage: float
    offset_ms: int
    exact: bool = False


def similarity_groups(
    fingerprints: tuple[MediaFingerprint, ...], minimum_similarity: float
) -> tuple[dict, ...]:
    """Return deterministic connected groups without exhaustive comparisons."""
    ordered = tuple(sorted(fingerprints, key=lambda item: item.relative_path.casefold()))
    by_id = {item.relative_path: item for item in ordered}
    union = _Union(tuple(by_id))
    pair_scores: dict[tuple[str, str], PairScore] = {}
    for left_id, right_id in _candidate_pairs(ordered):
        score = compare_media(by_id[left_id], by_id[right_id])
        pair_scores[(left_id, right_id)] = score
        if score.score >= minimum_similarity:
            union.join(left_id, right_id)
    grouped: dict[str, list[str]] = defaultdict(list)
    for identifier in by_id:
        grouped[union.find(identifier)].append(identifier)
    results = []
    for members in grouped.values():
        if len(members) < 2:
            continue
        results.append(_render_group(tuple(sorted(members)), by_id, pair_scores))
    return tuple(
        sorted(results, key=lambda item: (-item["score"], item["files"][0]["relativePath"]))
    )


def compare_media(left: MediaFingerprint, right: MediaFingerprint) -> PairScore:
    if left.sha256 == right.sha256:
        return PairScore(
            1.0, 1.0 if left.visual else None, 1.0 if left.audio else None, 1.0, 1.0, 0, True
        )
    visual = _sequence_similarity(left.visual, right.visual, VISUAL_BITS, VISUAL_MATCH_FLOOR)
    audio = _sequence_similarity(left.audio, right.audio, AUDIO_BITS, AUDIO_MATCH_FLOOR)
    available = [item for item in (visual, audio) if item is not None]
    if not available:
        return PairScore(0.0, None, None, 0.0, 0.0, 0)
    if visual is not None and audio is not None:
        score = (visual.score * 0.7) + (audio.score * 0.3)
        offset_ms = round((visual.offset * 500 * 0.7) + (audio.offset * 1000 * 0.3))
    else:
        only = available[0]
        score = only.score
        offset_ms = only.offset * (500 if visual is not None else 1000)
    left_coverage = sum(item.left_coverage for item in available) / len(available)
    right_coverage = sum(item.right_coverage for item in available) / len(available)
    return PairScore(
        _bounded(score),
        visual.score if visual is not None else None,
        audio.score if audio is not None else None,
        _bounded(left_coverage),
        _bounded(right_coverage),
        offset_ms,
    )


def _sequence_similarity(
    left: tuple[int, ...], right: tuple[int, ...], width: int, floor: float
) -> SequenceScore | None:
    if not left or not right:
        return None
    offset = _best_offset(left, right, width)
    comparisons = _aligned_similarities(left, right, offset, width)
    if not comparisons:
        return SequenceScore(0.0, 0.0, 0.0, offset)
    content = sum(comparisons) / len(comparisons)
    matched = sum(value >= floor for value in comparisons)
    left_coverage = matched / len(left)
    right_coverage = matched / len(right)
    coverage = (left_coverage * right_coverage) ** 0.5
    return SequenceScore(_bounded(content * coverage), left_coverage, right_coverage, offset)


def _candidate_pairs(fingerprints: tuple[MediaFingerprint, ...]) -> tuple[tuple[str, str], ...]:
    buckets: dict[tuple, set[str]] = defaultdict(set)
    digests: dict[str, set[str]] = defaultdict(set)
    for item in fingerprints:
        digests[item.sha256].add(item.relative_path)
        for key in _file_bands(item):
            buckets[key].add(item.relative_path)
    pairs: set[tuple[str, str]] = set()
    for members in (*digests.values(), *buckets.values()):
        ordered = sorted(members)
        if len(ordered) < 2:
            continue
        anchor = ordered[0]
        pairs.update((anchor, member) for member in ordered[1:])
    return tuple(sorted(pairs))


def _file_bands(item: MediaFingerprint) -> set[tuple]:
    keys: set[tuple] = set()
    for modality, sequence, width in (
        ("visual", item.visual, VISUAL_BITS),
        ("audio", item.audio, AUDIO_BITS),
    ):
        if not sequence:
            continue
        signature = _majority(sequence, width)
        keys.update((modality, *band) for band in _bands(signature, width))
    duration = item.duration_ms
    if duration is not None:
        # A broad logarithmic bucket catches a transcode even when every hash
        # band sits beside its old value.  It does not drive scoring.
        duration_band = max(0, duration.bit_length() - 1)
        if item.visual:
            keys.add(("duration", "visual", duration_band))
        if item.audio:
            keys.add(("duration", "audio", duration_band))
    return keys


def _bands(value: int, width: int) -> tuple[tuple[int, int], ...]:
    band_width = 8
    count = (width + band_width - 1) // band_width
    mask = (1 << band_width) - 1
    return tuple((index, (value >> (index * band_width)) & mask) for index in range(count))


def _best_offset(left: tuple[int, ...], right: tuple[int, ...], width: int) -> int:
    """Align two hash streams in O(bits * timeline * log(timeline)).

    Expanding every matching hash position makes two long, static videos a
    quadratic comparison even after file-level candidate indexing.  Binary
    cross-correlation scores every temporal offset together.  Agreement is
    centred around random chance, so a tiny coincidental overlap does not beat
    a long aligned region merely by being perfect.
    """
    left_values = np.asarray(left, dtype=np.uint64)
    right_values = np.asarray(right, dtype=np.uint64)
    result_size = left_values.size + right_values.size - 1
    fft_size = 1 << (result_size - 1).bit_length()
    correlations = np.zeros((fft_size // 2) + 1, dtype=np.complex128)
    for first_bit in range(0, width, 4):
        shifts = np.arange(first_bit, min(first_bit + 4, width), dtype=np.uint64)
        left_bits = ((left_values[:, None] >> shifts) & 1).T.astype(np.float32)
        right_bits = ((right_values[:, None] >> shifts) & 1).T.astype(np.float32)
        left_frequency = np.fft.rfft(left_bits[:, ::-1], n=fft_size, axis=1)
        right_frequency = np.fft.rfft(right_bits, n=fft_size, axis=1)
        correlations += np.sum(right_frequency * left_frequency, axis=0)
        correlations += np.sum(
            np.fft.rfft(1.0 - right_bits, n=fft_size, axis=1)
            * np.fft.rfft((1.0 - left_bits)[:, ::-1], n=fft_size, axis=1),
            axis=0,
        )
    matches = np.rint(np.fft.irfft(correlations, n=fft_size)[:result_size])
    offsets = np.arange(-(left_values.size - 1), right_values.size, dtype=np.int64)
    starts = np.maximum(0, -offsets)
    stops = np.minimum(left_values.size, right_values.size - offsets)
    overlaps = np.maximum(0, stops - starts)
    evidence = (2.0 * matches) - (width * overlaps)
    best = np.flatnonzero(evidence == evidence.max())
    direction = -1 if left < right else 1
    chosen = min(
        best,
        key=lambda index: (
            abs(offsets[index]),
            offsets[index] != direction * abs(offsets[index]),
        ),
    )
    return int(offsets[chosen])


def _aligned_similarities(
    left: tuple[int, ...], right: tuple[int, ...], offset: int, width: int
) -> tuple[float, ...]:
    start = max(0, -offset)
    stop = min(len(left), len(right) - offset)
    return tuple(
        _hash_similarity(left[index], right[index + offset], width) for index in range(start, stop)
    )


def _hash_similarity(left: int, right: int, width: int) -> float:
    return 1.0 - ((left ^ right).bit_count() / width)


def _majority(values: tuple[int, ...], width: int) -> int:
    threshold = len(values) / 2
    result = 0
    for bit in range(width):
        if sum((value >> bit) & 1 for value in values) > threshold:
            result |= 1 << bit
    return result


def _render_group(
    members: tuple[str, ...],
    by_id: dict[str, MediaFingerprint],
    known_scores: dict[tuple[str, str], PairScore],
) -> dict:
    representative = members[0]
    reference = by_id[representative]
    rendered = [
        _render_file(
            reference,
            PairScore(
                1.0,
                1.0 if reference.visual else None,
                1.0 if reference.audio else None,
                1.0,
                1.0,
                0,
                True,
            ),
        )
    ]
    scores = []
    bases: set[str] = set()
    for identifier in members[1:]:
        key = tuple(sorted((representative, identifier)))
        score = known_scores.get(key) or compare_media(reference, by_id[identifier])
        scores.append(score.score)
        if score.exact:
            bases.add("exact")
        if score.visual_score is not None:
            bases.add("visual")
        if score.audio_score is not None:
            bases.add("audio")
        rendered.append(_render_file(by_id[identifier], score))
    return {
        "id": f"group-{reference.sha256[:12]}",
        "score": _bounded(min(scores, default=1.0)),
        "basis": sorted(bases),
        "files": rendered,
    }


def _render_file(item: MediaFingerprint, score: PairScore) -> dict:
    return {
        "relativePath": item.relative_path,
        "fileName": Path(item.relative_path).name,
        "sha256": item.sha256,
        "modality": item.modality,
        "durationMs": item.duration_ms,
        "score": _bounded(score.score),
        "visualScore": None if score.visual_score is None else _bounded(score.visual_score),
        "audioScore": None if score.audio_score is None else _bounded(score.audio_score),
        "referenceCoverage": _bounded(score.left_coverage),
        "fileCoverage": _bounded(score.right_coverage),
        "offsetMs": score.offset_ms,
        "exact": score.exact,
    }


def _bounded(value: float) -> float:
    return round(max(0.0, min(1.0, float(value))), 6)


class _Union:
    def __init__(self, identifiers: tuple[str, ...]) -> None:
        self._parent = {identifier: identifier for identifier in identifiers}

    def find(self, identifier: str) -> str:
        parent = self._parent[identifier]
        if parent != identifier:
            self._parent[identifier] = self.find(parent)
        return self._parent[identifier]

    def join(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self._parent[max(left_root, right_root)] = min(left_root, right_root)


__all__ = ["PairScore", "SequenceScore", "compare_media", "similarity_groups"]
