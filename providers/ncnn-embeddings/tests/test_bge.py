from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace

import pytest
from omnitensor_ncnn_embeddings import bge

from omnitensor.plugins.document_spans import EmbeddingProvider
from omnitensor.sdk import CancellationController


def _ncnn(names, kinds=None):
    """A fake ncnn whose devices state their type, as the real one does.

    ncnn reports 0 discrete, 1 integrated, 2 virtual and 3 software (CPU);
    a device with no type given here is discrete hardware.
    """
    types = tuple(kinds) if kinds is not None else (0,) * len(names)
    return SimpleNamespace(
        get_gpu_count=lambda: len(names),
        get_gpu_info=lambda index: SimpleNamespace(
            device_name=lambda: names[index], type=lambda: types[index]
        ),
    )


def test_the_named_device_is_used_when_it_is_present(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "ncnn",
        _ncnn(("integrated", bge.QUALIFIED_DEVICE, "llvmpipe"), kinds=(1, 0, 3)),
    )

    assert bge.vulkan_device_index(bge.QUALIFIED_DEVICE) == 1


def test_a_sole_gpu_is_used_whatever_it_is_called(monkeypatch):
    """The measured card is a preference, and this machine does not have it.

    This is the case the old gate refused outright: `ask-selected-files` was
    unavailable on every machine but one, because the only device present was
    not the device the receipt named.
    """
    monkeypatch.setitem(sys.modules, "ncnn", _ncnn(("AMD Radeon 610M (RADV GFX1103_R1)",)))

    assert bge.vulkan_device_index(bge.QUALIFIED_DEVICE) == 0


def test_device_selection_refuses_missing_ncnn(monkeypatch):
    monkeypatch.setitem(sys.modules, "ncnn", None)

    with pytest.raises(ValueError) as excinfo:
        bge.vulkan_device_index(bge.QUALIFIED_DEVICE)

    assert str(excinfo.value) == "ncnn is unavailable"


def test_device_selection_refuses_only_when_the_choice_is_genuinely_ambiguous(monkeypatch):
    """Two unnamed cards is a question for the person, not a default to guess.

    The refusal names them, because "qualified BGE Vulkan device is
    unavailable" told somebody with two working GPUs nothing they could act on.
    """
    monkeypatch.setitem(sys.modules, "ncnn", _ncnn(("integrated", "discrete"), kinds=(1, 0)))

    with pytest.raises(ValueError, match="one of: discrete, integrated"):
        bge.vulkan_device_index(bge.QUALIFIED_DEVICE)

    monkeypatch.setitem(sys.modules, "ncnn", _ncnn(()))
    with pytest.raises(ValueError, match="no Vulkan device is available"):
        bge.vulkan_device_index(bge.QUALIFIED_DEVICE)


def test_a_software_device_is_never_embedded_on(monkeypatch):
    """llvmpipe alone is a refusal, not "the only one there is".

    The sole-device shortcut used to be taken without looking at the type, so
    a machine whose only Vulkan device was software embedded on the host CPU —
    the one thing the accelerator rule forbids.
    """
    monkeypatch.setitem(sys.modules, "ncnn", _ncnn(("llvmpipe",), kinds=(3,)))

    with pytest.raises(ValueError, match="CPU execution is not used"):
        bge.vulkan_device_index(bge.QUALIFIED_DEVICE)


def test_a_hardware_card_beside_a_software_one_is_unambiguous(monkeypatch):
    monkeypatch.setitem(sys.modules, "ncnn", _ncnn(("llvmpipe", "AMD Radeon 610M"), kinds=(3, 1)))

    assert bge.vulkan_device_index(bge.QUALIFIED_DEVICE) == 1


def test_two_cards_with_the_measured_name_is_still_a_choice(monkeypatch):
    """Duplicates cannot disambiguate themselves, so the person is asked."""
    monkeypatch.setitem(sys.modules, "ncnn", _ncnn((bge.QUALIFIED_DEVICE,) * 2))

    with pytest.raises(ValueError, match="name the Vulkan device"):
        bge.vulkan_device_index(bge.QUALIFIED_DEVICE)


def test_bge_embedder_uses_the_selected_device_and_query_prefix(tmp_path, monkeypatch):
    lease = tmp_path / "generation.lock"
    lease.touch()
    model = tmp_path / "model.param"
    tokenizer = tmp_path / "tokenizer.json"
    calls = []

    class FakeRunner:
        device_name = "whatever this machine has"

        def __init__(self, selected_model, selected_tokenizer, *, device_index):
            calls.append((selected_model, selected_tokenizer, device_index))

        def embed(self, text):
            calls.append(text)
            return [0.5] * 384

    monkeypatch.setattr(bge, "BgeTokenizer", lambda path: ("tokenizer", path))
    monkeypatch.setattr(bge, "VulkanBgeRunner", FakeRunner)
    monkeypatch.setattr(bge, "vulkan_device_index", lambda _preferred: 2)
    embedder = bge.BgeVulkanEmbedder(model, tokenizer, "a" * 64, lease)

    vectors = embedder._embed_sync(("question",), True, None)

    # The runner's device is recorded, never compared: what ran is what ran.
    assert calls[0] == (model, ("tokenizer", tokenizer), 2)
    assert calls[1] == f"{bge.QUERY_PREFIX}question"
    assert vectors == ((0.5,) * 384,)
    assert embedder.descriptor == EmbeddingProvider(
        "bge-small-documents-gpu",
        "gpu",
        "a" * 64,
        True,
    )


def test_the_model_is_loaded_once_per_embedder_not_once_per_embed(tmp_path, monkeypatch):
    lease = tmp_path / "generation.lock"
    lease.touch()
    builds = []
    enumerations = []

    class FakeRunner:
        device_name = "whatever this machine has"

        def __init__(self, *_args, **_kwargs):
            builds.append(1)
            self.closed = False

        def embed(self, _text):
            return [0.5] * 384

        def close(self):
            self.closed = True

    monkeypatch.setattr(bge, "BgeTokenizer", lambda path: path)
    monkeypatch.setattr(bge, "VulkanBgeRunner", FakeRunner)
    monkeypatch.setattr(bge, "vulkan_device_index", lambda _preferred: enumerations.append(1) or 0)
    embedder = bge.BgeVulkanEmbedder(tmp_path / "model", tmp_path / "tokenizer", "a" * 64, lease)

    embedder._embed_sync(("one",), False, None)
    embedder._embed_sync(("two",), False, None)

    assert builds == [1], "the ncnn model was rebuilt for a second question"
    assert enumerations == [1], "Vulkan was re-enumerated for a second question"

    runner = embedder._runner
    asyncio.run(embedder.aclose())
    asyncio.run(embedder.aclose())
    assert runner.closed
    embedder._embed_sync(("three",), False, None)
    assert builds == [1, 1], "a closed embedder did not reload"


def test_a_caller_may_name_the_device_to_embed_on(tmp_path, monkeypatch):
    """The 610M case: the 6600 XT is busy, so the work goes to the other card."""
    lease = tmp_path / "generation.lock"
    lease.touch()
    asked = []

    class FakeRunner:
        device_name = "AMD Radeon 610M (RADV GFX1103_R1)"

        def __init__(self, *_args, **_kwargs):
            pass

        def embed(self, _text):
            return [0.5] * 384

    monkeypatch.setattr(bge, "BgeTokenizer", lambda path: path)
    monkeypatch.setattr(bge, "VulkanBgeRunner", FakeRunner)
    monkeypatch.setattr(bge, "vulkan_device_index", lambda preferred: asked.append(preferred) or 0)
    embedder = bge.BgeVulkanEmbedder(
        tmp_path / "model",
        tmp_path / "tokenizer",
        "a" * 64,
        lease,
        device="AMD Radeon 610M (RADV GFX1103_R1)",
    )

    embedder._embed_sync(("text",), False, None)

    assert asked == ["AMD Radeon 610M (RADV GFX1103_R1)"]


def test_bge_public_async_surface_preflights_and_observes_cancellation(tmp_path, monkeypatch):
    lease = tmp_path / "generation.lock"
    lease.touch()
    embedder = bge.BgeVulkanEmbedder(tmp_path / "model", tmp_path / "tokenizer", "a" * 64, lease)
    calls = []

    def fake_embed(texts, query, cancellation):
        calls.append((texts, query, cancellation))
        return ((1.0,) * 384,)

    monkeypatch.setattr(embedder, "_embed_sync", fake_embed)
    asyncio.run(embedder.preflight())
    cancellation = CancellationController()
    vectors = asyncio.run(embedder.embed(("question",), query=True, cancellation=cancellation))

    assert calls == [
        (("BGE startup probe",), False, None),
        (("question",), True, cancellation),
    ]
    assert vectors == ((1.0,) * 384,)


def test_bge_refuses_a_missing_lease_and_an_invalid_vector(tmp_path, monkeypatch):
    with pytest.raises(ValueError) as missing_lease:
        bge.BgeVulkanEmbedder(tmp_path / "model", tmp_path / "tokenizer", "a" * 64, tmp_path / "x")
    assert str(missing_lease.value) == "accelerator lease is unavailable"

    lease = tmp_path / "generation.lock"
    lease.touch()
    embedder = bge.BgeVulkanEmbedder(tmp_path / "model", tmp_path / "tokenizer", "a" * 64, lease)
    monkeypatch.setattr(bge, "BgeTokenizer", lambda _path: object())
    monkeypatch.setattr(bge, "vulkan_device_index", lambda _preferred: 0)

    class InvalidVector:
        device_name = "some other card"

        def __init__(self, *_args, **_kwargs):
            pass

        def embed(self, _text):
            return [float("nan")] * 384

    # An unmeasured device is no longer a refusal; a vector that is not a
    # vector still is.
    monkeypatch.setattr(bge, "VulkanBgeRunner", InvalidVector)
    with pytest.raises(ValueError, match="invalid embedding"):
        embedder._embed_sync(("text",), False, CancellationController())
