from __future__ import annotations

import json

import pytest

from omnitensor.contract import (
    CONTRACT_SCHEMA,
    RUNTIME_CONTRACT_VERSION,
    build_contract_document,
    contract_document_text,
    schema_versions,
    wire_schema_names,
)
from omnitensor.registry import load_schema, schema_names, validate_document
from omnitensor.service import RUNTIME_METHODS


def dispatched_control_methods() -> set[str]:
    """The names the control socket actually dispatches, read from its table."""
    from omnitensor.ports import RuntimeHandler
    from omnitensor.socket_transport import _dispatch_table

    class Handler:
        async def apply_command_text(self, text):
            raise NotImplementedError

        submit_job_text = cancel_job_text = job_result_text = apply_command_text

        def describe_plugins_text(self):
            raise NotImplementedError

        describe_contract_text = describe_plugins_text

    handler: RuntimeHandler = Handler()
    return set(_dispatch_table(handler))


def test_the_handshake_announces_every_method_the_socket_dispatches():
    """An unannounced method is one a client can only find by calling it."""
    assert set(RUNTIME_METHODS) == dispatched_control_methods()


def test_the_handshake_announces_itself():
    """A client that must already know one method to ask has no handshake."""
    assert "describe-contract" in RUNTIME_METHODS


def test_the_document_validates_against_its_own_schema():
    document = json.loads(contract_document_text(RUNTIME_METHODS))

    assert validate_document(CONTRACT_SCHEMA, document) == []
    assert document["version"] == RUNTIME_CONTRACT_VERSION
    assert document["methods"] == sorted(RUNTIME_METHODS)


def test_every_wire_contract_is_announced_with_the_version_it_pins():
    """Read from each schema, so the announced number cannot drift from it."""
    announced = schema_versions()

    for name in wire_schema_names(schema_names()):
        declared = load_schema(name)["properties"]["version"]["const"]
        assert announced[name.removesuffix(".schema.json")] == declared


def test_contracts_the_service_never_puts_on_the_bus_are_not_announced():
    """A workload manifest is read from disk; announcing it describes an
    agreement this handshake is not part of."""
    announced = schema_versions()

    assert "workload-manifest" not in announced
    assert "runtime-job-submit" in announced


def test_the_runtime_schema_prefix_surface_is_exact_and_sorted():
    expected = (
        "control-reply.schema.json",
        "control-request.schema.json",
        "plugin-inventory.schema.json",
        "runtime-acknowledgement.schema.json",
        "runtime-command.schema.json",
        "runtime-contract.schema.json",
        "runtime-job-acknowledgement.schema.json",
        "runtime-job-cancel.schema.json",
        "runtime-job-result-request.schema.json",
        "runtime-job-result.schema.json",
        "runtime-job-submit.schema.json",
        "runtime-refusal.schema.json",
        "runtime-snapshot.schema.json",
    )

    assert wire_schema_names(schema_names()) == expected
    assert wire_schema_names(
        (
            "workload-manifest.schema.json",
            "xruntime-command.schema.json",
            "runtime-z.schema.json",
            "plugin-inventory.schema.json",
            "runtime-a.schema.json",
        )
    ) == (
        "plugin-inventory.schema.json",
        "runtime-a.schema.json",
        "runtime-z.schema.json",
    )


def test_a_schema_pinning_no_version_is_omitted_rather_than_guessed():
    """An announced version nobody enforces is worse than no announcement."""
    unpinned = {"runtime-command.schema.json": {"properties": {"version": {"type": "integer"}}}}

    versions = schema_versions(lambda name: unpinned.get(name, load_schema(name)))

    assert "runtime-command" not in versions
    assert "runtime-snapshot" in versions


def test_a_boolean_is_not_mistaken_for_a_version():
    """``True`` is an ``int`` in Python and would announce version 1."""
    patched = {"runtime-command.schema.json": {"properties": {"version": {"const": True}}}}

    versions = schema_versions(lambda name: patched.get(name, load_schema(name)))

    assert "runtime-command" not in versions


def test_a_document_that_does_not_validate_is_refused_rather_than_sent():
    with pytest.raises(ValueError, match="contract description is invalid"):
        contract_document_text(["not a control method name"])


def test_the_methods_are_sorted_so_two_hosts_answer_identically():
    document = build_contract_document(["submit-job", "apply-command"], load_schema)

    assert document["methods"] == ["apply-command", "submit-job"]
