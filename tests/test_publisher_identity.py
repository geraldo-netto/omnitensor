from __future__ import annotations

import base64
import json
import stat

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import omnitensor.publisher_identity as publisher_identity
from omnitensor.plugins.artifact_trust import ArtifactTrustVerifier
from omnitensor.plugins.artifacts import ArtifactReference
from omnitensor.publisher_identity import (
    PUBLISHER_PURPOSE,
    PublisherIdentityError,
    load_or_create_publisher_identity,
    load_publisher_trust_root,
    setup_main,
)


def _roots(tmp_path):
    return tmp_path / "data", tmp_path / "config"


def _reference():
    return ArtifactReference("retinexformer-lol-v1", "1.0.0", "ncnn", "a" * 64)


def test_default_identity_is_created_once_reused_and_signs_offline(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg-data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg-config"))

    created = load_or_create_publisher_identity()
    private_before = created.paths.private_key.read_bytes()
    reused = load_or_create_publisher_identity()

    assert created.created is True
    assert reused.created is False
    assert reused.label == "xpuwlm"
    assert reused.publisher == "xpuwlm.local"
    assert reused.key_id == "xpuwlm-ed25519-v1"
    assert reused.paths.private_key.read_bytes() == private_before
    assert stat.S_IMODE(reused.paths.private_key.stat().st_mode) == 0o600
    assert stat.S_IMODE(reused.paths.public_key.stat().st_mode) == 0o644
    assert stat.S_IMODE(reused.paths.private_key.parent.stat().st_mode) == 0o700
    provenance = reused.sign(
        _reference(),
        "recipe:retinexformer-lol-v1@1.0.0",
        published_at_ms=1_700_000_000_000,
    )
    verifier = ArtifactTrustVerifier([reused.trust_root()])
    assert verifier.verify(_reference(), provenance).trusted is True


def test_setup_cli_reports_paths_without_private_material(tmp_path, capsys):
    keys, trusts = _roots(tmp_path)

    assert setup_main(["--key-root", str(keys), "--trust-root", str(trusts)]) == 0

    document = json.loads(capsys.readouterr().out)
    assert document["created"] is True
    assert document["purpose"] == PUBLISHER_PURPOSE
    assert document["privateKeyPrinted"] is False
    assert "BEGIN PRIVATE KEY" not in json.dumps(document)


@pytest.mark.parametrize(
    ("label", "detail"),
    [
        ("", "publisher label is invalid"),
        ("XPUWLM", "publisher label is invalid"),
        ("bad label", "publisher label is invalid"),
        ("x" * 49, "publisher label is invalid"),
    ],
)
def test_invalid_labels_fail_closed(tmp_path, label, detail):
    keys, trusts = _roots(tmp_path)
    with pytest.raises(PublisherIdentityError, match=detail):
        load_or_create_publisher_identity(label=label, key_root=keys, trust_root=trusts)


@given(st.text(min_size=0, max_size=20).filter(lambda value: not value or not value.isascii()))
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_non_ascii_labels_are_never_filesystem_identities(tmp_path, value):
    keys, trusts = _roots(tmp_path)
    with pytest.raises(PublisherIdentityError, match="publisher label is invalid"):
        load_or_create_publisher_identity(label=value, key_root=keys, trust_root=trusts)


def test_relative_key_root_and_non_regular_key_are_refused(tmp_path):
    with pytest.raises(PublisherIdentityError, match="publisher key root must be absolute"):
        load_or_create_publisher_identity(key_root="relative", trust_root=tmp_path)

    keys, trusts = _roots(tmp_path)
    private = keys / "xpuwlm/xpuwlm-ed25519-v1.private.pem"
    private.mkdir(parents=True)
    with pytest.raises(PublisherIdentityError, match="publisher key is not regular"):
        load_or_create_publisher_identity(key_root=keys, trust_root=trusts)


def test_incomplete_and_mismatched_keypairs_are_refused(tmp_path):
    keys, trusts = _roots(tmp_path)
    key_directory = keys / "xpuwlm"
    key_directory.mkdir(parents=True)
    private = key_directory / "xpuwlm-ed25519-v1.private.pem"
    private.write_bytes(b"only-one-half")
    with pytest.raises(PublisherIdentityError, match="keypair-incomplete"):
        load_or_create_publisher_identity(key_root=keys, trust_root=trusts)

    private.unlink()
    identity = load_or_create_publisher_identity(key_root=keys, trust_root=trusts)
    other_public = Ed25519PrivateKey.generate().public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    identity.paths.public_key.write_bytes(other_public)
    with pytest.raises(PublisherIdentityError, match="publisher keypair does not match"):
        load_or_create_publisher_identity(key_root=keys, trust_root=trusts)


def test_invalid_private_public_and_oversized_keys_are_refused(tmp_path):
    keys, trusts = _roots(tmp_path)
    identity = load_or_create_publisher_identity(key_root=keys, trust_root=trusts)
    identity.paths.private_key.write_bytes(b"not a private key")
    with pytest.raises(PublisherIdentityError, match="publisher private key is invalid"):
        load_or_create_publisher_identity(key_root=keys, trust_root=trusts)

    identity.paths.private_key.write_bytes(b"x" * (16 * 1024 + 1))
    with pytest.raises(PublisherIdentityError, match="publisher key is too large"):
        load_or_create_publisher_identity(key_root=keys, trust_root=trusts)


def test_existing_conflicting_trust_document_is_not_overwritten(tmp_path):
    keys, trusts = _roots(tmp_path)
    identity = load_or_create_publisher_identity(key_root=keys, trust_root=trusts)
    document = json.loads(identity.paths.trust_document.read_text())
    document["purpose"] = "something else"
    identity.paths.trust_document.write_text(json.dumps(document))

    with pytest.raises(PublisherIdentityError, match="trust-conflict"):
        load_or_create_publisher_identity(key_root=keys, trust_root=trusts)
    assert json.loads(identity.paths.trust_document.read_text())["purpose"] == "something else"


@pytest.mark.parametrize(
    "change",
    [
        lambda document: document.pop("purpose"),
        lambda document: document.__setitem__("version", 2),
        lambda document: document.__setitem__("algorithm", "RSA"),
        lambda document: document.__setitem__("publicKey", "%%%"),
        lambda document: document.__setitem__("fingerprintSha256", "0" * 64),
    ],
)
def test_trust_loader_rejects_contract_drift(tmp_path, change):
    keys, trusts = _roots(tmp_path)
    identity = load_or_create_publisher_identity(key_root=keys, trust_root=trusts)
    document = json.loads(identity.paths.trust_document.read_text())
    change(document)
    identity.paths.trust_document.write_text(json.dumps(document))

    with pytest.raises(PublisherIdentityError, match="trust-invalid"):
        load_publisher_trust_root(identity.paths.trust_document)


def test_trust_loader_rejects_non_ed25519_public_key_length(tmp_path):
    path = tmp_path / "trust.json"
    public_key = b"short"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "label": "xpuwlm",
                "publisher": "xpuwlm.local",
                "keyId": "xpuwlm-ed25519-v1",
                "algorithm": "Ed25519",
                "publicKey": base64.b64encode(public_key).decode(),
                "fingerprintSha256": __import__("hashlib").sha256(public_key).hexdigest(),
                "purpose": PUBLISHER_PURPOSE,
            }
        )
    )
    root = load_publisher_trust_root(path)
    with pytest.raises(ValueError, match="32-byte Ed25519"):
        ArtifactTrustVerifier([root])


def test_cli_reports_setup_failure(tmp_path, capsys):
    assert setup_main(["--label", "Bad", "--key-root", str(tmp_path)]) == 1
    assert "publisher key setup failed: label-invalid" in capsys.readouterr().err


def test_atomic_key_write_removes_partial_file_on_sync_failure(tmp_path, monkeypatch):
    destination = tmp_path / "key.pem"

    def fail_sync(_descriptor):
        raise OSError("sync")

    monkeypatch.setattr("omnitensor.atomicio.os.fsync", fail_sync)

    with pytest.raises(OSError, match="sync"):
        publisher_identity._write_bytes_atomic(destination, b"secret", 0o600)

    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


def test_keypair_creation_removes_private_half_when_public_write_fails(
    tmp_path, monkeypatch
):
    paths = publisher_identity.PublisherIdentityPaths(
        tmp_path / "private.pem",
        tmp_path / "public.pem",
        tmp_path / "trust.json",
    )
    original = publisher_identity._write_bytes_atomic
    original_remove = publisher_identity._remove_durable
    calls = 0
    removed = []

    def fail_second(path, payload, mode):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("public write")
        original(path, payload, mode)

    def remove_durable(path):
        removed.append(path)
        original_remove(path)

    monkeypatch.setattr(publisher_identity, "_write_bytes_atomic", fail_second)
    monkeypatch.setattr(publisher_identity, "_remove_durable", remove_durable)
    with pytest.raises(OSError, match="public write"):
        publisher_identity._create_keypair(paths)

    assert removed == [paths.private_key]
    assert not paths.private_key.exists()
    assert not paths.public_key.exists()
