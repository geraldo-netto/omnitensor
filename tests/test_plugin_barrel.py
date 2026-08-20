from __future__ import annotations

import ast
import hashlib
import importlib
import json
import subprocess
import sys
import threading
from pathlib import Path

import pytest

import omnitensor.plugins as plugins

ROOT = Path(__file__).parents[1]
KERNEL_EXPORTS = (
    "KernelAggregate",
    "KernelTelemetryError",
    "KernelTelemetryState",
    "LatencyHistogram",
    "KernelCounter",
    "UnixSocketAggregateSource",
    "AbsentAggregateSource",
    "KernelAggregateSource",
    "parse_aggregate",
    "scheduler_features",
)
# Covers the targets as well as the names, so retargeting an export at its owning
# module moves it. The surface itself — which names, in which order — is asserted
# above and did not move when the supervisor facade was retired.
EXPORT_MAP_SHA256 = "46bd2c4aff49de68e4ada6b8e38992f948a5cca40f6245f634f88f24c08b5549"


def test_export_map_preserves_the_locked_public_surface_and_order():
    assert isinstance(plugins.__all__, list)
    assert len(plugins.__all__) == len(plugins._EXPORTS) == 364
    assert plugins.__all__ == list(plugins._EXPORTS)
    assert set(plugins.__all__) <= set(dir(plugins))
    assert tuple(plugins.__all__[-len(KERNEL_EXPORTS) :]) == KERNEL_EXPORTS

    serialized = "\n".join(
        f"{name}={module}:{attribute}" for name, (module, attribute) in plugins._EXPORTS.items()
    )
    assert hashlib.sha256(serialized.encode()).hexdigest() == EXPORT_MAP_SHA256


def test_importing_the_package_does_not_import_any_export_owner():
    statement = """
import json
import sys
import omnitensor.plugins as plugins
print(json.dumps({
    "count": len(plugins.__all__),
    "owners": sorted(
        name for name in sys.modules
        if name.startswith("omnitensor.plugins.")
    ),
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", statement],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {"count": 364, "owners": []}


def test_dir_lists_lazy_exports_without_resolving_them():
    statement = """
import json
import sys
import omnitensor.plugins as plugins
listed = dir(plugins)
print(json.dumps({
    "complete": set(plugins.__all__) <= set(listed),
    "owners": sorted(
        name for name in sys.modules
        if name.startswith("omnitensor.plugins.")
    ),
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", statement],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {"complete": True, "owners": []}


def test_every_export_is_the_exact_canonical_owner_object_and_is_cached():
    for name, (module_name, attribute_name) in plugins._EXPORTS.items():
        owner = importlib.import_module(module_name, plugins.__name__)
        value = getattr(plugins, name)
        assert value is getattr(owner, attribute_name), name
        assert plugins.__dict__[name] is value


def test_successful_resolution_imports_once_and_then_uses_globals(monkeypatch):
    name = "ArtifactCache"
    module_name, _attribute_name = plugins._EXPORTS[name]
    plugins.__dict__.pop(name, None)
    imported = []
    real_import = plugins._import_module

    def tracking_import(requested, package):
        imported.append((requested, package))
        return real_import(requested, package)

    monkeypatch.setattr(plugins, "_import_module", tracking_import)

    first = getattr(plugins, name)
    second = getattr(plugins, name)

    assert first is second
    assert imported == [(module_name, plugins.__name__)]


def test_unknown_name_raises_attribute_error_without_importing_an_owner():
    before = set(sys.modules)

    with pytest.raises(
        AttributeError,
        match="module 'omnitensor.plugins' has no attribute 'NotAPluginContract'",
    ):
        assert plugins.NotAPluginContract is None

    assert set(sys.modules) == before


def test_failed_resolution_is_not_cached_and_can_be_retried(monkeypatch):
    name = "ArtifactCacheError"
    plugins.__dict__.pop(name, None)
    real_import = plugins._import_module
    attempts = 0

    def fail_once(module_name, package):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ImportError("owner unavailable")
        return real_import(module_name, package)

    monkeypatch.setattr(plugins, "_import_module", fail_once)

    with pytest.raises(ImportError, match="owner unavailable"):
        getattr(plugins, name)
    assert name not in plugins.__dict__
    assert getattr(plugins, name).__name__ == name
    assert attempts == 2


def test_recursive_resolution_fails_locally_then_outer_resolution_completes(monkeypatch):
    name = "ArtifactCacheAccounting"
    plugins.__dict__.pop(name, None)
    real_import = plugins._import_module
    recursive_errors = []

    def recursive_import(module_name, package):
        try:
            getattr(plugins, name)
        except AttributeError as error:
            recursive_errors.append(str(error))
        return real_import(module_name, package)

    monkeypatch.setattr(plugins, "_import_module", recursive_import)

    resolved = getattr(plugins, name)

    assert resolved.__name__ == name
    assert recursive_errors == [
        "module 'omnitensor.plugins' cannot resolve 'ArtifactCacheAccounting' recursively"
    ]
    assert plugins.__dict__[name] is resolved


def test_concurrent_resolution_returns_one_canonical_identity(monkeypatch):
    name = "ArtifactCacheCollection"
    plugins.__dict__.pop(name, None)
    real_import = plugins._import_module
    barrier = threading.Barrier(2)
    resolved = []
    failures = []

    def concurrent_import(module_name, package):
        barrier.wait(timeout=5)
        return real_import(module_name, package)

    def resolve():
        try:
            resolved.append(getattr(plugins, name))
        except BaseException as error:  # noqa: BLE001 - containment: this must not escape into the caller
            failures.append(error)

    monkeypatch.setattr(plugins, "_import_module", concurrent_import)
    threads = [threading.Thread(target=resolve) for _index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert failures == []
    assert len(resolved) == 2
    assert resolved[0] is resolved[1] is plugins.__dict__[name]


def test_import_star_and_submodule_from_import_keep_standard_semantics():
    statement = """
namespace = {}
exec("from omnitensor.plugins import *", namespace)
import omnitensor.plugins as plugins
from omnitensor.plugins import generation_catalog
import omnitensor.plugins.generation_catalog as direct
assert [name for name in namespace if name != "__builtins__"] == plugins.__all__
assert namespace["ArtifactReference"] is plugins.ArtifactReference
assert namespace["KernelAggregate"] is plugins.KernelAggregate
assert generation_catalog is direct
"""
    completed = subprocess.run(
        [sys.executable, "-c", statement],
        check=False,
        capture_output=True,
        text=True,
    )

    assert (completed.returncode, completed.stderr) == (0, "")


def test_no_module_in_the_package_imports_public_names_back_from_the_barrel():
    """Every module in the package, not only the 42 that own an export.

    Gate audit, 2026-08-20: this read `_EXPORTS` for its file list, so it
    checked 42 of the 85 modules in `omnitensor/plugins` and `import
    omnitensor.plugins` inside any of the other 43 was green. It also had no
    case for `from ..plugins import X`, which is how a module one package down
    would reach the barrel. Both were found by feeding it the mistake it
    exists to catch, which nothing had done.
    """
    package = ROOT / "src/omnitensor/plugins"
    modules = [
        path
        for path in sorted(package.rglob("*.py"))
        if "__pycache__" not in path.parts and path.name != "__init__.py"
    ]
    assert len(modules) >= 80
    # The owners are a subset, and were the whole of what this used to read.
    owners = {
        package / f"{module_name.removeprefix('.')}.py"
        for module_name, _attribute_name in plugins._EXPORTS.values()
    }
    assert owners < set(modules)

    reaching = {
        str(path.relative_to(ROOT)): sorted(_barrel_reads(path))
        for path in modules
        if _barrel_reads(path)
    }

    assert reaching == {}


def _barrel_reads(path: Path) -> set[str]:
    """Every way this module reaches its own package barrel, in any spelling."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names if alias.name == "omnitensor.plugins")
        elif isinstance(node, ast.ImportFrom):
            found.update(_barrel_names(node))
    return found


def _barrel_names(node: ast.ImportFrom) -> set[str]:
    names = {alias.name for alias in node.names}
    if node.module == "omnitensor.plugins" or (node.level == 1 and node.module is None):
        return names & set(plugins._EXPORTS)
    if node.module == "omnitensor" or (node.level >= 2 and node.module is None):
        return names & {"plugins"}
    # `from ..plugins import X` reaches the barrel from a subpackage, and this
    # rule had no case for it at all.
    if node.level >= 1 and node.module == "plugins":
        return names
    return set()


def test_kernel_exports_keep_canonical_identity_defaults_and_consumer_features():
    kernel = importlib.import_module(".kernel_telemetry", plugins.__name__)
    for name in KERNEL_EXPORTS:
        assert getattr(plugins, name) is getattr(kernel, name)

    source = plugins.UnixSocketAggregateSource()
    assert source.socket_path == Path("/run/omnitensor/bpf-aggregate.sock")
    assert source._max_aggregate_bytes == 256 * 1024
    assert source._timeout_seconds == 2.0

    aggregate = plugins.parse_aggregate(
        {
            "version": 1,
            "histograms": [{"name": "runq_latency_us", "unit": "us", "buckets": [1, 3]}],
            "counters": [{"name": "wakeups", "value": 4}],
        }
    )
    assert isinstance(aggregate, plugins.KernelAggregate)
    assert isinstance(aggregate.histograms[0], plugins.LatencyHistogram)
    assert isinstance(aggregate.counters[0], plugins.KernelCounter)
    assert plugins.scheduler_features(aggregate) == {
        "kernelRunQueueSamples": 4.0,
        "kernelRunQueueP50UpperUs": 4.0,
        "kernelRunQueueP95UpperUs": 4.0,
    }
    assert (
        plugins.AbsentAggregateSource().read().state is plugins.KernelTelemetryState.HELPER_ABSENT
    )


def test_kernel_reader_defaults_are_documented():
    documentation = (ROOT / "docs/installation.md").read_text(encoding="utf-8")
    normalized = " ".join(documentation.split())
    assert (
        "The unprivileged reader defaults to `/run/omnitensor/bpf-aggregate.sock`, "
        "a 256 KiB aggregate limit, and a 2.0-second timeout." in normalized
    )
