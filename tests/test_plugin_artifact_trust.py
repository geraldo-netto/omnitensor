from __future__ import annotations

import base64
import dataclasses
import hashlib
import json

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from hypothesis import given
from hypothesis import strategies as st

from omnitensor.plugins import (
    ArtifactActivation,
    ArtifactInstallationError,
    ArtifactInstaller,
    ArtifactProvenance,
    ArtifactReference,
    ArtifactTrustDecision,
    ArtifactTrustPolicy,
    ArtifactTrustRoot,
    ArtifactTrustVerifier,
    artifact_provenance_document,
    artifact_signature_payload,
)


def reference(content: bytes = b"model", *, version: str = "1.2.3") -> ArtifactReference:
    return ArtifactReference(
        "sample-model",
        version,
        "onnx",
        hashlib.sha256(content).hexdigest(),
    )


def signer(
    artifact: ArtifactReference,
    *,
    publisher: str = "acme.models",
    key_id: str = "release-1",
    source: str = "https://models.example/sample-model",
    published_at_ms: int = 1_700_000_000_000,
):
    private_key = Ed25519PrivateKey.generate()
    provenance = ArtifactProvenance(publisher, key_id, source, published_at_ms, "")
    signature = private_key.sign(artifact_signature_payload(artifact, provenance))
    provenance = dataclasses.replace(
        provenance,
        signature=base64.b64encode(signature).decode("ascii"),
    )
    public_key = private_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    root = ArtifactTrustRoot(publisher, key_id, public_key)
    return ArtifactTrustVerifier([root]), provenance, root


def test_signed_install_persists_provenance_and_verifies_offline(tmp_path):
    artifact = reference()
    verifier, provenance, _root = signer(artifact)
    source = tmp_path / "model.onnx"
    source.write_bytes(b"model")
    installer = ArtifactInstaller(tmp_path / "store", trust_verifier=verifier)

    installed = installer.install(artifact, source, provenance=provenance)
    source.unlink()

    assert installer.resolve_active(artifact.id).path == installed.path
    assert json.loads((installed.path.parent / "provenance.json").read_text()) == (
        artifact_provenance_document(artifact, provenance)
    )
    assert verifier.verify_installed(artifact, installed.path.parent) == ArtifactTrustDecision(
        True, ""
    )


def test_signature_payload_and_provenance_document_are_frozen(artifact_policy_case):
    artifact, provenance, _root = artifact_policy_case

    assert artifact_signature_payload(artifact, provenance) == (
        b'omnitensor-artifact-signature-v1\n{"artifact":{"format":"onnx",'
        b'"id":"sample-model","sha256":"9372c470eeadd5ecd9c3c74c2b3cb633f8e2f2fad'
        b'799250a0f70d652b6b825e4","version":"1.2.3"},"provenance":{"keyId":'
        b'"release-1","publishedAtMs":1700000000000,"publisher":"acme.models",'
        b'"source":"https://models.example/sample-model"}}'
    )
    assert artifact_provenance_document(artifact, provenance) == {
        "version": 1,
        "artifact": {
            "id": "sample-model",
            "version": "1.2.3",
            "format": "onnx",
            "sha256": "9372c470eeadd5ecd9c3c74c2b3cb633f8e2f2fad799250a0f70d652b6b825e4",
        },
        "provenance": {
            "publisher": "acme.models",
            "keyId": "release-1",
            "source": "https://models.example/sample-model",
            "publishedAtMs": 1_700_000_000_000,
            "signature": provenance.signature,
        },
    }


def test_strict_policy_rejects_unsigned_install_before_activation(tmp_path):
    artifact = reference()
    verifier = ArtifactTrustVerifier([])
    source = tmp_path / "model.onnx"
    source.write_bytes(b"model")
    store = tmp_path / "store"
    installer = ArtifactInstaller(store, trust_verifier=verifier)

    with pytest.raises(ArtifactInstallationError) as excinfo:
        installer.install(artifact, source)

    assert (excinfo.value.code, excinfo.value.detail) == (
        "artifact-untrusted",
        "artifact is unsigned",
    )
    assert installer.activation(artifact.id) == ArtifactActivation(None, None)
    assert not (store / artifact.id / artifact.version).exists()


def test_policy_can_explicitly_allow_and_persist_unsigned_artifacts(tmp_path):
    artifact = reference()
    verifier = ArtifactTrustVerifier([], policy=ArtifactTrustPolicy(require_signature=False))
    source = tmp_path / "model.onnx"
    source.write_bytes(b"model")
    installer = ArtifactInstaller(tmp_path / "store", trust_verifier=verifier)

    installed = installer.install(artifact, source)

    assert installer.resolve_active(artifact.id).ready is True
    assert json.loads((installed.path.parent / "provenance.json").read_text())["provenance"] is None


@pytest.mark.parametrize(
    ("policy", "reason"),
    [
        (
            ArtifactTrustPolicy(revoked_publishers=frozenset({"acme.models"})),
            "publisher is revoked: acme.models",
        ),
        (
            ArtifactTrustPolicy(revoked_keys=frozenset({("acme.models", "release-1")})),
            "publisher key is revoked: acme.models/release-1",
        ),
    ],
)
def test_revocation_policy_rejects_valid_signatures(artifact_policy_case, policy, reason):
    artifact, provenance, root = artifact_policy_case
    verifier = ArtifactTrustVerifier([root], policy=policy)
    assert verifier.verify(artifact, provenance) == ArtifactTrustDecision(False, reason)


@pytest.fixture
def artifact_policy_case():
    artifact = reference()
    _verifier, provenance, root = signer(artifact)
    return artifact, provenance, root


def test_updated_revocation_policy_makes_installed_active_artifact_unready(tmp_path):
    artifact = reference()
    verifier, provenance, root = signer(artifact)
    source = tmp_path / "model.onnx"
    source.write_bytes(b"model")
    store = tmp_path / "store"
    ArtifactInstaller(store, trust_verifier=verifier).install(
        artifact, source, provenance=provenance
    )
    revoked = ArtifactTrustVerifier(
        [root],
        policy=ArtifactTrustPolicy(revoked_publishers=frozenset({root.publisher})),
    )

    resolution = ArtifactInstaller(store, trust_verifier=revoked).resolve_active(artifact.id)

    assert resolution.ready is False
    assert resolution.path is None
    assert resolution.reason == "publisher is revoked: acme.models"


def test_signature_is_bound_to_every_artifact_and_provenance_field(artifact_policy_case):
    artifact, provenance, root = artifact_policy_case
    verifier = ArtifactTrustVerifier([root])
    mutations = [
        dataclasses.replace(artifact, version="2.0.0"),
        dataclasses.replace(artifact, sha256="0" * 64),
    ]
    provenance_mutations = [
        dataclasses.replace(provenance, source="https://models.example/other"),
        dataclasses.replace(provenance, published_at_ms=provenance.published_at_ms + 1),
    ]

    for changed_artifact in mutations:
        assert verifier.verify(changed_artifact, provenance).reason == (
            "artifact signature verification failed"
        )
    for changed_provenance in provenance_mutations:
        assert verifier.verify(artifact, changed_provenance).reason == (
            "artifact signature verification failed"
        )


def test_untrusted_key_and_malformed_signatures_have_stable_reasons(artifact_policy_case):
    artifact, provenance, _root = artifact_policy_case
    assert ArtifactTrustVerifier([]).verify(artifact, provenance).reason == (
        "publisher key is not trusted: acme.models/release-1"
    )
    verifier, _other_provenance, root = signer(artifact)
    assert verifier.verify(artifact, dataclasses.replace(provenance, signature="!")) == (
        ArtifactTrustDecision(False, "artifact signature is not valid base64")
    )
    short = dataclasses.replace(provenance, signature=base64.b64encode(b"x").decode())
    assert ArtifactTrustVerifier([root]).verify(artifact, short) == ArtifactTrustDecision(
        False, "artifact signature has invalid length"
    )
    random_signature = dataclasses.replace(
        provenance,
        signature=base64.b64encode(b"x" * 64).decode(),
    )
    assert ArtifactTrustVerifier([root]).verify(
        artifact, random_signature
    ) == ArtifactTrustDecision(False, "artifact signature verification failed")


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"publisher": "Bad Publisher"}, "invalid artifact publisher: 'Bad Publisher'"),
        ({"key_id": "bad key"}, "invalid artifact publisher key id: 'bad key'"),
        ({"source": ""}, "invalid artifact provenance source"),
        ({"source": "x" * 2049}, "invalid artifact provenance source"),
        ({"published_at_ms": True}, "invalid artifact provenance timestamp"),
        ({"published_at_ms": -1}, "invalid artifact provenance timestamp"),
        ({"published_at_ms": 253402300800000}, "invalid artifact provenance timestamp"),
        ({"signature": 42}, "invalid artifact signature"),
    ],
)
def test_invalid_provenance_fields_have_stable_reasons(artifact_policy_case, changes, reason):
    artifact, provenance, root = artifact_policy_case
    decision = ArtifactTrustVerifier([root]).verify(
        artifact, dataclasses.replace(provenance, **changes)
    )
    assert decision == ArtifactTrustDecision(False, reason)


def test_missing_provenance_obeys_current_policy(tmp_path):
    artifact = reference()
    strict = ArtifactTrustVerifier([]).verify_installed(artifact, tmp_path)
    relaxed = ArtifactTrustVerifier(
        [], policy=ArtifactTrustPolicy(require_signature=False)
    ).verify_installed(artifact, tmp_path)
    assert strict == ArtifactTrustDecision(False, "artifact is unsigned")
    assert relaxed == ArtifactTrustDecision(True, "")


@pytest.mark.parametrize(
    ("root", "reason"),
    [
        (
            ArtifactTrustRoot("Bad Publisher", "release-1", b"0" * 32),
            "invalid artifact publisher: 'Bad Publisher'",
        ),
        (
            ArtifactTrustRoot("acme.models", "bad key", b"0" * 32),
            "invalid artifact publisher key id: 'bad key'",
        ),
        (
            ArtifactTrustRoot("acme.models", "release-1", b"short"),
            "artifact trust root must contain a 32-byte Ed25519 public key",
        ),
    ],
)
def test_trust_root_configuration_is_strict(root, reason):
    with pytest.raises(ValueError) as excinfo:
        ArtifactTrustVerifier([root])
    assert str(excinfo.value) == reason


def test_duplicate_trust_root_is_rejected(artifact_policy_case):
    _artifact, _provenance, root = artifact_policy_case
    with pytest.raises(ValueError) as excinfo:
        ArtifactTrustVerifier([root, root])
    assert str(excinfo.value) == "duplicate artifact trust root: acme.models/release-1"


@pytest.mark.parametrize(
    ("document_change", "reason"),
    [
        (lambda document: document.update(extra=True), "artifact provenance has invalid fields"),
        (
            lambda document: document.update(version=2),
            "artifact provenance has unsupported version",
        ),
        (
            lambda document: document["artifact"].update(version="9.0.0"),
            "artifact provenance identity does not match",
        ),
        (
            lambda document: document["provenance"].update(extra=True),
            "artifact provenance signature has invalid fields",
        ),
    ],
)
def test_stored_provenance_rejects_corruption(
    tmp_path, artifact_policy_case, document_change, reason
):
    artifact, provenance, root = artifact_policy_case
    version_root = tmp_path / artifact.id / artifact.version
    version_root.mkdir(parents=True)
    document = artifact_provenance_document(artifact, provenance)
    document_change(document)
    (version_root / "provenance.json").write_text(json.dumps(document))

    decision = ArtifactTrustVerifier([root]).verify_installed(artifact, version_root)

    assert decision == ArtifactTrustDecision(False, reason)


def test_stored_provenance_read_failure_is_stable(tmp_path, artifact_policy_case):
    artifact, _provenance, root = artifact_policy_case
    version_root = tmp_path / artifact.id / artifact.version
    version_root.mkdir(parents=True)
    (version_root / "provenance.json").write_text("not-json")
    decision = ArtifactTrustVerifier([root]).verify_installed(artifact, version_root)
    assert decision.trusted is False
    assert decision.reason.startswith("artifact provenance cannot be read:")


def test_rollback_rechecks_current_revocation_policy(tmp_path):
    store = tmp_path / "store"
    source_v1 = tmp_path / "v1.onnx"
    source_v2 = tmp_path / "v2.onnx"
    source_v1.write_bytes(b"v1")
    source_v2.write_bytes(b"v2")
    v1 = reference(b"v1", version="1.0.0")
    v2 = reference(b"v2", version="2.0.0")
    verifier_v1, provenance_v1, root_v1 = signer(v1, key_id="release-1")
    _verifier_v2, provenance_v2, root_v2 = signer(v2, key_id="release-2")
    verifier = ArtifactTrustVerifier([root_v1, root_v2])
    installer = ArtifactInstaller(store, trust_verifier=verifier)
    installer.install(v1, source_v1, provenance=provenance_v1)
    installer.install(v2, source_v2, provenance=provenance_v2)
    revoked = ArtifactTrustVerifier(
        [root_v1, root_v2],
        policy=ArtifactTrustPolicy(revoked_keys=frozenset({("acme.models", "release-1")})),
    )
    installer = ArtifactInstaller(store, trust_verifier=revoked)

    with pytest.raises(ArtifactInstallationError) as excinfo:
        installer.rollback(v1.id)

    assert excinfo.value.code == "rollback-invalid"
    assert excinfo.value.detail == "publisher key is revoked: acme.models/release-1"
    assert installer.activation(v1.id) == ArtifactActivation(v2, v1)


@given(
    publisher=st.one_of(st.none(), st.integers(), st.text(max_size=80)),
    key_id=st.one_of(st.none(), st.integers(), st.text(max_size=80)),
    source=st.one_of(st.none(), st.integers(), st.text(max_size=2100)),
    published_at_ms=st.one_of(st.none(), st.booleans(), st.integers()),
    signature=st.one_of(st.none(), st.integers(), st.text(max_size=120)),
)
def test_arbitrary_provenance_metadata_never_raises(
    publisher, key_id, source, published_at_ms, signature
):
    provenance = ArtifactProvenance(publisher, key_id, source, published_at_ms, signature)
    decision = ArtifactTrustVerifier([]).verify(reference(), provenance)
    assert isinstance(decision.trusted, bool)
    assert isinstance(decision.reason, str)
