"""The applet vendors the shared schemas; nothing compared the two copies.

`schemas/` is the contract with the Cinnamon applet, but a Cinnamon applet
cannot fetch a schema at load time, so it ships its own copy of every schema it
shares with this service. One had already drifted: the applet's copy did not
list the `ggml-whisper` artifact format this repository publishes in
`plugin-manifests/media-transcription.json`, so the applet refused the only one
of five manifests that used it.

The applet keeps the mirror image of this gate. Both run wherever the sibling
checkout is present and both report themselves as unavailable, never as
passing, where only one repository exists.
"""

import json
import os
from pathlib import Path

import pytest

SHARED_SCHEMAS = (
    "control-reply",
    "control-request",
    "runtime-acknowledgement",
    "runtime-command",
    "runtime-contract",
    "runtime-refusal",
    "runtime-snapshot",
    "workload-manifest",
)

# The two schemas are one contract in two dialects. The applet compiles every
# shipped schema with ajv under `strict: true`, which refuses two constructs
# this repository's Python validator accepts. Each entry is a JSON pointer the
# applet may legitimately spell differently, with the reason it must; anything
# else that differs is drift.
DIALECT_EXCEPTIONS = {
    ("oneOf",): (
        "ajv strictRequired refuses `not: {required: [plugin]}` unless the "
        "subschema also declares the property, so the applet spells the "
        "version 1 discriminator as `plugin: false`"
    ),
    ("properties", "requirements", "allOf"): (
        "this schema constrains `acceleratorPreference[0]` with a one-entry "
        "`prefixItems`; ajv strictTuples rejects an open-ended tuple, so the "
        "applet enforces the rule in `declaresDesignedForFirst` instead"
    ),
}

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
APPLET_UUID = "cinnamon-xpuwlm@geraldo-netto"


def _applet_schema_root() -> Path | None:
    configured = os.environ.get("OMNITENSOR_APPLET_ROOT")
    roots = [Path(configured)] if configured else [REPOSITORY_ROOT.parent / "cinnamon-xpuwlm"]
    for root in roots:
        candidate = root / "files" / APPLET_UUID
        if (candidate / "workload-manifest.schema.json").is_file():
            return candidate
    return None


APPLET_SCHEMAS = _applet_schema_root()
requires_applet = pytest.mark.skipif(
    APPLET_SCHEMAS is None,
    reason="set OMNITENSOR_APPLET_ROOT to run the cross-repository schema parity gate",
)


def _read(root: Path, name: str) -> dict:
    return json.loads((root / f"{name}.schema.json").read_text())


def _excepted(pointer: tuple[str, ...]) -> bool:
    return any(pointer[: len(prefix)] == prefix for prefix in DIALECT_EXCEPTIONS)


def _describe(value: object) -> str:
    rendered = json.dumps(value)
    return rendered if len(rendered) <= 80 else f"{rendered[:77]}..."


def _differences(published, vendored, pointer: tuple[str, ...] = ()) -> list[str]:
    """Every difference, so one run names the whole drift."""
    if _excepted(pointer):
        return []
    location = "/" + "/".join(pointer)
    if isinstance(published, dict) and isinstance(vendored, dict):
        return _mapping_differences(published, vendored, pointer)
    if isinstance(published, list) and isinstance(vendored, list):
        if len(published) != len(vendored):
            return [f"{location}: {len(published)} entries published, {len(vendored)} vendored"]
        return [
            difference
            for index, (left, right) in enumerate(zip(published, vendored, strict=True))
            for difference in _differences(left, right, (*pointer, str(index)))
        ]
    if published == vendored and isinstance(published, type(vendored)):
        return []
    return [f"{location}: {_describe(published)} published, {_describe(vendored)} vendored"]


def _mapping_differences(published: dict, vendored: dict, pointer: tuple[str, ...]) -> list[str]:
    differences: list[str] = []
    for key in sorted({*published, *vendored}):
        child = (*pointer, key)
        if _excepted(child):
            continue
        location = "/" + "/".join(child)
        if key not in published:
            differences.append(f"{location}: vendored by the applet, absent here")
        elif key not in vendored:
            differences.append(f"{location}: published here, missing from the applet")
        else:
            differences.extend(_differences(published[key], vendored[key], child))
    return differences


@requires_applet
@pytest.mark.parametrize("name", SHARED_SCHEMAS)
def test_the_applet_vendors_every_shared_schema_without_drift(name: str) -> None:
    drift = _differences(_read(REPOSITORY_ROOT / "schemas", name), _read(APPLET_SCHEMAS, name))
    assert drift == [], f"{name}.schema.json has drifted:\n" + "\n".join(drift)


@requires_applet
def test_every_dialect_exception_still_covers_a_real_difference() -> None:
    """An exception that stops being needed starts hiding drift."""
    published = _read(REPOSITORY_ROOT / "schemas", "workload-manifest")
    vendored = _read(APPLET_SCHEMAS, "workload-manifest")
    for pointer, why in DIALECT_EXCEPTIONS.items():
        left, right = published, vendored
        for key in pointer:
            left, right = left[key], right[key]
        assert left != right, f"/{'/'.join(pointer)} no longer differs; drop the exception ({why})"


@requires_applet
def test_every_published_plugin_manifest_is_installable_by_the_applet() -> None:
    """Parity is only worth gating because these documents cross the boundary."""
    import jsonschema

    from omnitensor.registry import validate_workload_document

    vendored = jsonschema.Draft202012Validator(_read(APPLET_SCHEMAS, "workload-manifest"))
    manifests = sorted((REPOSITORY_ROOT / "plugin-manifests").glob("*.json"))
    assert manifests, "no plug-in manifests to check"
    for path in manifests:
        document = json.loads(path.read_text())
        assert validate_workload_document(document) == [], f"{path.name} fails its own schema"
        rejected = [error.message for error in vendored.iter_errors(document)]
        assert rejected == [], f"{path.name} is published here, refused by the applet: {rejected}"
