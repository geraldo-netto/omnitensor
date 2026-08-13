"""Shared bounded text normalization for extracted visual content."""

from omnitensor.plugins.media_transcription import MediaTranscriptionError


def joined_visible_text(parts) -> str:
    text = "\n".join(str(part).strip() for part in parts if str(part).strip())
    if len(text) > 16_384 or "\x00" in text:
        raise MediaTranscriptionError(
            "presentation-invalid", "presentation slide text exceeds its limit"
        )
    return text


__all__ = ["joined_visible_text"]
