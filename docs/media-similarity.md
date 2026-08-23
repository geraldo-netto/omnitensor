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
`selectedFiles.maxBytes` remains the per-file staging boundary; the workload
adds no second file-count ceiling.

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
   no top-N display ceiling hides work.

The fixed score threshold is published in each result. Component visual and
audio scores, coverage, offset, digest, and exact-copy status remain visible so
the combined score is reviewable rather than a hidden verdict.

## Safety and privacy

The workload is read-only. It never deletes, renames, moves, or rewrites media.
It declares only `files:read-selected`, which remains revocable while the scan
runs. Decoding and deterministic fingerprinting are host analysis, not a CPU
inference backend or a fallback from an accelerator model. No source
content, full frame, PCM window, or absolute path is persisted by this
provider.

Build and install `providers/media-similarity` beside the service, grant its
selected-file permission, restart OmniTensor, then open the Media similarity
workload in `xpuwlm`.
