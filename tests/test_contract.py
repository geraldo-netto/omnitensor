from __future__ import annotations

import ast
import inspect
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
from omnitensor.service import BUS_METHODS


def decorated_bus_methods() -> set[str]:
    """The names actually exported on the bus, read from the interface."""
    from omnitensor import service

    tree = ast.parse(inspect.getsource(service))
    interface = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "OmniTensorInterface"
    )
    return {
        node.name
        for node in interface.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and any(
            isinstance(decorator, ast.Call) and getattr(decorator.func, "id", "") == "method"
            for decorator in node.decorator_list
        )
    }


def test_the_handshake_announces_every_method_the_bus_exports():
    """An unannounced method is one a client can only find by calling it."""
    assert set(BUS_METHODS) == decorated_bus_methods()


def test_the_handshake_announces_itself():
    """A client that must already know one method to ask has no handshake."""
    assert "DescribeContract" in BUS_METHODS


def test_the_document_validates_against_its_own_schema():
    document = json.loads(contract_document_text(BUS_METHODS))

    assert validate_document(CONTRACT_SCHEMA, document) == []
    assert document["version"] == RUNTIME_CONTRACT_VERSION
    assert document["methods"] == sorted(BUS_METHODS)


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
        contract_document_text(["not a D-Bus method name"])


def test_the_methods_are_sorted_so_two_hosts_answer_identically():
    document = build_contract_document(["SubmitJob", "ApplyCommand"], load_schema)

    assert document["methods"] == ["ApplyCommand", "SubmitJob"]
