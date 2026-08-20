"""Every script in `scripts/` is runtime source, and no gate loaded most of it.

OMNI-0535. With the root suite, the six provider suites and the template suite
combined, 248 of 256 runtime source files were imported or executed by some
gate. `provider-requirements.py` was run as a subprocess by
`test_ci_supply_chain.py`; `collect-selected-text-acceptance.py` was only read
as text; and `benchmark-transport-codecs.py` and
`generate-media-acceptance-fixtures.py` were touched by nothing at all — a
rename or a broken import in any of them was found by a person running it, not
by a gate.

The discovery is derived, so a fifth script is covered the day it lands rather
than the day somebody remembers to add it here.
"""

from __future__ import annotations

import importlib.util
import json
import struct
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def script_paths() -> list[Path]:
    return sorted(path for path in SCRIPTS.glob("*.py") if not path.name.startswith("_"))


def load(name: str):
    """Import one hyphenated script by path, the way nothing else can."""
    path = SCRIPTS / name
    module_name = f"omnitensor_script_{path.stem.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_there_are_scripts_to_cover():
    assert [path.name for path in script_paths()] == [
        "benchmark-transport-codecs.py",
        "collect-selected-text-acceptance.py",
        "generate-media-acceptance-fixtures.py",
        "provider-requirements.py",
    ]


@pytest.mark.parametrize("path", script_paths(), ids=lambda path: path.name)
def test_every_script_imports(path: Path):
    """A broken import in a script is a script nobody can run.

    Importing is the whole assertion: every one of them guards its entry point
    with `if __name__ == "__main__"`, so nothing here runs a benchmark, opens a
    control socket, or shells out to ffmpeg.
    """
    module = load(path.name)

    assert module.__doc__, f"{path.name} says nothing about what it is for"
    entry = getattr(module, "main", None) or getattr(module, "requirements", None)
    assert callable(entry), f"{path.name} exposes no entry point"


def test_the_selected_text_collector_parses_the_arguments_the_guide_documents():
    module = load("collect-selected-text-acceptance.py")

    parsed = module._arguments(
        [
            "--output",
            "/tmp/out.json",
            "--qwen-model",
            "/models/qwen.gguf",
            "--hebrew-model",
            "/models/dictalm.gguf",
            "--worker-load-receipt",
            "/state/selected-text-worker-load.json",
        ]
    )

    assert parsed.output == Path("/tmp/out.json")
    assert parsed.corpus is None
    assert parsed.timeout_seconds == 90.0

    # Each required option is required: dropping one must refuse rather than
    # collect evidence against a model nobody named.
    for dropped in ("--output", "--qwen-model", "--hebrew-model", "--worker-load-receipt"):
        argv = [
            "--output",
            "/tmp/out.json",
            "--qwen-model",
            "/models/qwen.gguf",
            "--hebrew-model",
            "/models/dictalm.gguf",
            "--worker-load-receipt",
            "/state/receipt.json",
        ]
        index = argv.index(dropped)
        del argv[index : index + 2]
        with pytest.raises(SystemExit):
            module._arguments(argv)


def test_the_selected_text_collector_refuses_a_worker_that_is_not_the_qualified_one():
    module = load("collect-selected-text-acceptance.py")
    ready = {
        "plugins": [
            {
                "id": "selected-text-tools",
                "version": "1.1.0",
                "workerState": "ready",
                "artifacts": [
                    {"id": "qwen3-8b-q4-k-m", "ready": True},
                    {"id": "dictalm2-hebrew-q4-k-m", "ready": True},
                    {"id": "something-else-the-manifest-declares", "ready": False},
                ],
            }
        ]
    }

    module._require_ready(ready)

    for mutate, expected in (
        (lambda plugin: plugin.update(workerState="starting"), "is not ready"),
        (lambda plugin: plugin.update(version="1.0.0"), "is not ready"),
        (
            lambda plugin: plugin.update(artifacts=[{"id": "qwen3-8b-q4-k-m", "ready": True}]),
            "not installed",
        ),
        (
            lambda plugin: plugin["artifacts"][1].update(ready=False),
            "are not ready",
        ),
        (lambda plugin: plugin.update(id="something-else"), "identity is missing"),
    ):
        document = json.loads(json.dumps(ready))
        mutate(document["plugins"][0])
        with pytest.raises(RuntimeError, match=expected):
            module._require_ready(document)


def test_the_codec_benchmark_frames_and_measures_what_it_claims_to():
    module = load("benchmark-transport-codecs.py")

    payload = b'{"a":1}'
    framed = module.frame(payload)
    assert struct.unpack(">I", framed[:4])[0] == len(payload)
    assert framed[4:] == payload

    values = module.tensor_values(512)
    assert len(values) == 512
    assert values[0] == 0.0
    assert values[255] == 1.0

    acknowledgement = module.command_acknowledgement()
    assert acknowledgement["status"] == "rejected"
    assert set(acknowledgement["portfolio"]) == {"paused", "profiles", "deviceChoices"}

    document = {"version": 1, "values": values[:8]}
    wire, encode_us, decode_us = module.measure(
        lambda: module.frame(json.dumps(document).encode("utf-8")), 5
    )
    assert json.loads(wire[4:]) == document
    assert encode_us > 0 and decode_us > 0

    # The live probes are probes: no service here, and they must answer None
    # rather than take the benchmark down with them.
    assert module.live_snapshot() is None or isinstance(module.live_snapshot(), dict)
    assert module.live_inventory() is None or isinstance(module.live_inventory(), dict)


def test_the_media_fixture_generator_refuses_a_target_it_would_overwrite(tmp_path):
    module = load("generate-media-acceptance-fixtures.py")

    with pytest.raises(SystemExit, match="new absolute directory"):
        module.generate(Path("relative-target"))
    with pytest.raises(SystemExit, match="new absolute directory"):
        module.generate(tmp_path)

    with pytest.raises(SystemExit, match="tool is unavailable: definitely-not-a-tool"):
        module._command("definitely-not-a-tool")


def test_the_media_fixture_generator_writes_the_one_fixture_that_needs_no_tool(tmp_path):
    """The KOI8-R fixture and the manifest are pure Python; the rest shell out."""
    module = load("generate-media-acceptance-fixtures.py")

    cyrillic = module._generate_text_fixture(tmp_path)
    assert cyrillic == "Привет мир"
    assert (tmp_path / "legacy-koi8-r.txt").read_bytes() == cyrillic.encode("koi8-r")

    assert cyrillic in module._svg(cyrillic)

    module._write_fixture_manifest(tmp_path)
    manifest = json.loads((tmp_path / "fixtures.json").read_text(encoding="utf-8"))
    assert set(manifest) == {"legacy-koi8-r.txt"}
    assert manifest["legacy-koi8-r.txt"]["bytes"] == len(cyrillic.encode("koi8-r"))
    assert len(manifest["legacy-koi8-r.txt"]["sha256"]) == 64


def test_the_provider_requirements_script_drops_only_the_local_distributions():
    module = load("provider-requirements.py")

    media = module.requirements(ROOT / "providers/media-transcription/pyproject.toml")

    assert all(not item.startswith("omnitensor") for item in media)
    assert "pywhispercpp==1.5.0" in media
    # Extras are third-party requirements too, and were exactly what went
    # missing when a provider suite failed 35 of 107.
    runtime = module.requirements(ROOT / "providers/vulkan-runtime/pyproject.toml")
    assert "ncnn>=1.0.20260526" in runtime
    assert "pymupdf>=1.24" in runtime
