"""Owner for explicit deletion of user-controlled desktop training history."""

from __future__ import annotations

import sys
from pathlib import Path

from omnitensor.storelock import store_lock
from omnitensor.training.contracts import TrainingError

DESKTOP_REVOCATION_CONFIRMATION = "I-confirm-delete-desktop-training-history"
_ORIGINAL_STORE_LOCK = store_lock


def revoke_desktop_history(path: Path | str, confirmation: str) -> bool:
    """Delete one history under its lock after explicit exact confirmation."""
    if confirmation != DESKTOP_REVOCATION_CONFIRMATION:
        raise TrainingError(
            "revocation-not-confirmed",
            f"pass exactly {DESKTOP_REVOCATION_CONFIRMATION}",
        )
    source = Path(path).expanduser()
    if source.is_symlink() or (source.exists() and not source.is_file()):
        raise TrainingError("history-invalid", "desktop history is not a safe regular file")
    lock = store_lock
    legacy = sys.modules.get("omnitensor.training.desktop")
    if legacy is not None:
        legacy_lock = getattr(legacy, "store_lock", _ORIGINAL_STORE_LOCK)
        if legacy_lock is not _ORIGINAL_STORE_LOCK:
            lock = legacy_lock
    with lock(source.parent, ".desktop-training.lock"):
        if not source.exists():
            return False
        if source.is_symlink() or not source.is_file():
            raise TrainingError("history-invalid", "desktop history changed during revocation")
        source.unlink()
    return True


revoke_desktop_history.__module__ = "omnitensor.training.desktop"
