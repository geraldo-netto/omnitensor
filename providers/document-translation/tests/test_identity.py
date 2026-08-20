"""The shim's identity, checked where the wheel that ships it is built.

Every structural half of this — the entry-point name, the packaged manifest,
the plugin id inside it, the wheel's package list — is derived from the tree
and checked in both directions by `tests/test_generation_installation.py` in
the service repository. What is left is the half that needs the wheel to be
installed, and it is stated once in `providers/shim_identity.py` rather than
copied into five files that then have to agree.
"""

from __future__ import annotations

from pathlib import Path

from shim_identity import assert_installed_shim_binds_the_runtime_factory

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "omnitensor_document_translation"


def test_the_installed_shim_binds_the_runtime_factory():
    assert_installed_shim_binds_the_runtime_factory(ROOT, PACKAGE)
