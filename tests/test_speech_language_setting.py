"""Telling the transcriber what language a recording is in.

Reviewed from `../video-doc-transcriber`, whose most person-facing feature is
exactly this: it takes `--language pt|en|fr|it|de|es|auto` rather than always
detecting. Detection reads the first window that carries audio, so a recording
that opens with noise, with music, or with one borrowed English word is
transcribed as the wrong language from beginning to end, and nothing in the
answer says why.

Everything else that repository does — audio, video, images, PDFs — is already
this workload, on the GPU with pinned weights; what it does that this service
will not is CPU OCR engines that download their own models. So this is the
part worth having, and it is a setting rather than a new workload.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from omnitensor.plugins.media_transcription import (
    AUTOMATIC_LANGUAGE,
    SPEECH_LANGUAGE_SETTING,
    LanguageDirectedTranscriber,
    MediaProviderIdentity,
    MediaTranscriptionPlugin,
    SpeechTranscript,
)
from omnitensor.plugins.protocol import PluginContext

ROOT = Path(__file__).parents[1]
MANIFEST = ROOT / "plugin-manifests" / "media-transcription.json"


class Recording:
    """A speech transcriber that records what it was told, and detects when not."""

    def __init__(self) -> None:
        self.preferred: str | None | object = "never set"

    def prefer_language(self, language: str | None) -> None:
        self.preferred = language

    async def transcribe(self, source, cancellation):  # pragma: no cover - never called here
        return SpeechTranscript(None, ())


class Detecting:
    """One that only detects, which is every transcriber before this setting."""

    async def transcribe(self, source, cancellation):  # pragma: no cover - never called here
        return SpeechTranscript(None, ())


class _Probe:
    async def inspect(self, source):  # pragma: no cover - port stub
        raise NotImplementedError


class _Frames:
    async def sample(self, source, media, cancellation):  # pragma: no cover - port stub
        raise NotImplementedError

    async def discard(self, frames):  # pragma: no cover - port stub
        raise NotImplementedError


class _Vision:
    async def transcribe(self, frame, cancellation):  # pragma: no cover - port stub
        raise NotImplementedError

    async def release(self):  # pragma: no cover - port stub
        raise NotImplementedError


class _Pages:
    async def transcribe(self, source, cancellation):  # pragma: no cover - port stub
        raise NotImplementedError


def _plugin(speech):
    """The plugin with every port but speech stubbed: none of them is reached.

    `on_start` is where the setting lands, and it runs before any source is
    inspected — so this exercises exactly the wiring under test.
    """

    async def nothing():
        return None

    return MediaTranscriptionPlugin(
        identity=MediaProviderIdentity("test-provider", "gpu"),
        probe=_Probe(),
        speech=speech,
        frames=_Frames(),
        vision=_Vision(),
        presentations=_Pages(),
        documents=_Pages(),
        preflight=nothing,
    )


def _start(plugin, configuration):
    context = PluginContext(
        "media-transcription", 1, configuration, frozenset({"files:read-selected"})
    )
    asyncio.run(plugin.start(context))


@pytest.mark.parametrize("language", ["pt", "en", "de", "zh", "pt-BR"])
def test_a_chosen_language_is_what_the_transcriber_is_told(language):
    speech = Recording()

    _start(_plugin(speech), {SPEECH_LANGUAGE_SETTING: language})

    assert speech.preferred == language


@pytest.mark.parametrize("configuration", [{}, {SPEECH_LANGUAGE_SETTING: AUTOMATIC_LANGUAGE}])
def test_auto_and_absent_both_leave_the_detector_to_answer(configuration):
    """The default is the behaviour this workload always had."""
    speech = Recording()

    _start(_plugin(speech), configuration)

    assert speech.preferred is None


def test_a_transcriber_that_only_detects_is_driven_exactly_as_before():
    """`prefer_language` is a port, not a requirement."""
    plugin = _plugin(Detecting())

    _start(plugin, {SPEECH_LANGUAGE_SETTING: "pt"})

    assert not isinstance(plugin._speech, LanguageDirectedTranscriber)


def test_the_manifest_declares_the_setting_and_defaults_to_detecting():
    from omnitensor.plugins import manifest_configuration_spec

    document = json.loads(MANIFEST.read_text(encoding="utf-8"))
    spec = manifest_configuration_spec(document["id"], document)

    assert spec.defaults == {SPEECH_LANGUAGE_SETTING: AUTOMATIC_LANGUAGE}
    assert spec.schema["additionalProperties"] is False


@pytest.mark.parametrize("language", ["auto", "pt", "en", "zh", "yue", "pt-BR", "sr-Latn"])
def test_the_manifest_accepts_a_language_code_the_model_understands(language):
    """Not an allowlist of six: whisper's own set is large and not ours to narrow."""
    import jsonschema

    document = json.loads(MANIFEST.read_text(encoding="utf-8"))
    schema = document["plugin"]["schemas"]["configuration"]["properties"][SPEECH_LANGUAGE_SETTING]

    jsonschema.Draft202012Validator(schema).validate(language)


@pytest.mark.parametrize("value", ["", "Portuguese", "p", "PT", "en_US", "x" * 13, "auto "])
def test_the_manifest_refuses_what_is_not_a_language_code(value):
    import jsonschema

    document = json.loads(MANIFEST.read_text(encoding="utf-8"))
    schema = document["plugin"]["schemas"]["configuration"]["properties"][SPEECH_LANGUAGE_SETTING]

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(value)
