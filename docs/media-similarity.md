# Media similarity

`media-similarity` scans either every supported audio and video file below one
folder or a set of media files selected directly by the client. It groups exact
duplicates and near-duplicate encodes; it does not claim that unrelated
recordings about the same subject are semantically equivalent.

For a folder, the client walks it without following symbolic links. For direct
selection, it keeps every chosen file as a separate source. Both routes submit
one ordered `sources` list plus matching, unique relative display paths. The
selected-file broker copies those files into the isolated worker in one job.
No absolute source directory enters the result. The runtime's published
`selectedFiles.maxBytes` remains the per-file staging boundary. The runtime
publishes and enforces a 1 GiB ceiling so ordinary large video files can be
scanned; the workload adds no second file-count ceiling.

Supported suffixes are WAV, FLAC, MP3, OGG/Vorbis, Opus, M4A/AAC, MP4, WebM,
MKV, MOV, AVI, and M4V. A malformed supported file becomes one result failure;
it does not discard groups found among the remaining files.

## Comparison pipeline

Each file is decoded once per present stream, never once per candidate pair.

1. SHA-256 identifies byte-exact copies.
2. Every decoded video frame is rotated into display orientation, converted to
   grayscale, resized into one aspect-preserving fixed representation, and
   perceptually hashed. Every frame contributes to a 500 ms majority
   descriptor, so encodes with different frame rates align without keeping
   every decoded frame in memory.
3. Audio is resampled to mono 16 kHz float PCM. Each one-second window becomes
   a gain-normalized spectral fingerprint.
4. Compact hash bands form an inverted candidate index. Each collision bucket
   is compared with one representative, avoiding exhaustive directory-wide
   pair expansion. Temporal comparison estimates an offset and reports content
   similarity plus coverage in both directions.
5. Connected matches become groups. Every group and every member is returned;
   no top-N display ceiling hides work. The first member is the preferred
   preservation source. Video groups favor higher pixel resolution before
   available stream bitrate; audio groups favor lossless streams before sample
   rate, channels, and bitrate. Duration and path provide deterministic later
   tie-breakers. These properties rank a completed group and never affect
   similarity or membership.

The fixed score threshold is published in each result. Component visual and
audio scores, coverage, offset, digest, and exact-copy status remain visible so
the combined score is reviewable rather than a hidden verdict.

## Long-running scans and recovery

A scan has no total-duration ceiling. The runtime instead keeps its normal
30-second silence watchdog and treats observable work as proof of life:

- the selected-file broker reports digest, copy, source-verification, and
  staged-copy-verification bytes while preparing every file;
- the provider reports source-digest bytes, decoded video frames, and decoded
  audio samples while fingerprinting; and
- repeated work in one phase is rate-limited to one report every five seconds,
  while phase changes and known completions are reported immediately.

The watchdog therefore continues to detect a worker or broker stage that
stops making progress, while a directory job that keeps working may run for as
long as its inputs require.

After each file is fingerprinted, the provider transactionally checkpoints
that completed file in its sandbox-private plugin state. A checkpoint contains
the relative display path, source SHA-256, compact visual/audio hashes,
duration, modality, and quality metadata. It contains no media bytes, decoded
frames, PCM, or absolute source path. An interrupted retry for the same ordered
selection hashes each staged file again and reuses a checkpoint only when its
relative path and source digest still match. Successfully completing the scan
removes that selection's checkpoint rows.

## Safety and privacy

The workload is read-only. It never deletes, renames, moves, or rewrites media.
It declares only `files:read-selected`, which remains revocable while the scan
runs. Decoding and deterministic fingerprinting are host analysis, not a CPU
inference backend or a fallback from an accelerator model. No source
content, full frame, PCM window, or absolute path is persisted by this
provider; only the private, compact recovery data described above may survive
an interrupted scan.

Build and install `providers/media-similarity` beside the service, grant its
selected-file permission, restart OmniTensor, then open the Media similarity
workload in `xpuwlm`.
