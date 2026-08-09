"""Reusable protocol, fuzz, failure, and changed-function conformance fixtures."""

from __future__ import annotations

import ast
import asyncio
import math
import random
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum

from ..plugins.protocol import (
    JsonObject,
    JsonValue,
    PluginContext,
    PluginHealth,
    PluginProgress,
    PluginRequest,
    PluginResult,
    PluginResultStatus,
    WorkloadPlugin,
)
from .helpers import CancellationController

DEFAULT_CONTRACT_TIMEOUT_SECONDS = 1.0
DEFAULT_MAX_CONFORMANCE_PROGRESS = 128
DEFAULT_FUZZ_CASES = 64
DEFAULT_FUZZ_DEPTH = 4
DEFAULT_FUZZ_WIDTH = 8
DEFAULT_FUZZ_STRING_CHARS = 128


class ConformanceError(AssertionError):
    """A plugin violated a stable public contract."""


@dataclass(frozen=True, slots=True)
class ContractReport:
    health: PluginHealth
    result: PluginResult
    progress: tuple[PluginProgress, ...]


class ProgressProbe:
    """Bounded `ProgressReporter` fixture validating every observation."""

    def __init__(
        self,
        job_id: str,
        *,
        limit: int = DEFAULT_MAX_CONFORMANCE_PROGRESS,
    ) -> None:
        if not isinstance(job_id, str) or not job_id:
            raise ValueError("job_id must be a non-empty string")
        _positive_integer(limit, "limit")
        self._job_id = job_id
        self._limit = limit
        self._fraction = 0.0
        self.items: list[PluginProgress] = []

    async def report(self, progress: PluginProgress) -> None:
        if not isinstance(progress, PluginProgress):
            raise ConformanceError("progress reporter received a non-PluginProgress value")
        if progress.job_id != self._job_id:
            raise ConformanceError("progress job identity differs from the request")
        if (
            isinstance(progress.fraction, bool)
            or not isinstance(progress.fraction, (int, float))
            or not math.isfinite(progress.fraction)
            or progress.fraction < self._fraction
            or progress.fraction > 1
        ):
            raise ConformanceError("progress fraction is invalid or moved backward")
        if len(self.items) >= self._limit:
            raise ConformanceError(f"plugin emitted more than {self._limit} progress updates")
        self._fraction = float(progress.fraction)
        self.items.append(progress)


async def run_plugin_contract(
    plugin: WorkloadPlugin,
    context: PluginContext,
    request: PluginRequest,
    *,
    timeout: float = DEFAULT_CONTRACT_TIMEOUT_SECONDS,
    max_progress: int = DEFAULT_MAX_CONFORMANCE_PROGRESS,
) -> ContractReport:
    """Exercise one normal request and always release plugin lifecycle state."""
    _positive_number(timeout, "timeout")
    if not isinstance(plugin, WorkloadPlugin):
        raise ConformanceError("plugin does not implement WorkloadPlugin")
    if request.plugin_id != context.plugin_id:
        raise ConformanceError("request and context plugin identities differ")
    progress = ProgressProbe(request.job_id, limit=max_progress)
    cancellation = CancellationController()
    started = False
    try:
        await asyncio.wait_for(plugin.start(context), timeout=timeout)
        started = True
        health = await asyncio.wait_for(plugin.health(), timeout=timeout)
        _validate_health(health)
        result = await asyncio.wait_for(
            plugin.execute(request, cancellation, progress),
            timeout=timeout,
        )
        _validate_result(request, result)
        return ContractReport(health, result, tuple(progress.items))
    finally:
        if started:
            await asyncio.wait_for(plugin.stop(), timeout=timeout)


async def run_cancellation_contract(
    plugin: WorkloadPlugin,
    context: PluginContext,
    request: PluginRequest,
    *,
    timeout: float = DEFAULT_CONTRACT_TIMEOUT_SECONDS,
) -> ContractReport:
    """Require a pre-cancelled request to terminate as cancelled within a bound."""
    _positive_number(timeout, "timeout")
    if not isinstance(plugin, WorkloadPlugin):
        raise ConformanceError("plugin does not implement WorkloadPlugin")
    progress = ProgressProbe(request.job_id)
    cancellation = CancellationController()
    cancellation.cancel("conformance cancellation")
    started = False
    try:
        await asyncio.wait_for(plugin.start(context), timeout=timeout)
        started = True
        health = await asyncio.wait_for(plugin.health(), timeout=timeout)
        _validate_health(health)
        result = await asyncio.wait_for(
            plugin.execute(request, cancellation, progress),
            timeout=timeout,
        )
        _validate_result(request, result)
        if result.status is not PluginResultStatus.CANCELLED:
            raise ConformanceError("pre-cancelled request did not return cancelled status")
        return ContractReport(health, result, tuple(progress.items))
    finally:
        if started:
            await asyncio.wait_for(plugin.stop(), timeout=timeout)


class FailurePoint(StrEnum):
    START = "start"
    HEALTH = "health"
    COLLECT = "collect"
    PREPROCESS = "preprocess"
    RESOLVE = "resolve"
    INFER = "infer"
    POSTPROCESS = "postprocess"
    DELIVER = "deliver"
    RESULT = "result"
    STOP = "stop"


class InjectedFailureError(RuntimeError):
    def __init__(self, point: FailurePoint) -> None:
        self.point = point
        super().__init__(f"injected failure at {point}")


class FailureInjector:
    """Deterministically raise once or repeatedly at named pipeline boundaries."""

    def __init__(self, failures: Mapping[FailurePoint, int]) -> None:
        remaining: dict[FailurePoint, int] = {}
        for point, count in failures.items():
            if not isinstance(point, FailurePoint):
                raise TypeError("failure point must be a FailurePoint")
            _positive_integer(count, "failure count")
            remaining[point] = count
        self._remaining = remaining
        self.hits: list[FailurePoint] = []

    def hit(self, point: FailurePoint) -> None:
        if not isinstance(point, FailurePoint):
            raise TypeError("failure point must be a FailurePoint")
        self.hits.append(point)
        count = self._remaining.get(point, 0)
        if count == 0:
            return
        self._remaining[point] = count - 1
        raise InjectedFailureError(point)

    def remaining(self, point: FailurePoint) -> int:
        if not isinstance(point, FailurePoint):
            raise TypeError("failure point must be a FailurePoint")
        return self._remaining.get(point, 0)


def bounded_json_payloads(
    *,
    seed: int = 0,
    cases: int = DEFAULT_FUZZ_CASES,
    max_depth: int = DEFAULT_FUZZ_DEPTH,
    max_width: int = DEFAULT_FUZZ_WIDTH,
    max_string_chars: int = DEFAULT_FUZZ_STRING_CHARS,
) -> tuple[JsonObject, ...]:
    """Build a reproducible JSON fuzz corpus with strict structural bounds."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    for value, name in (
        (cases, "cases"),
        (max_depth, "max_depth"),
        (max_width, "max_width"),
        (max_string_chars, "max_string_chars"),
    ):
        _positive_integer(value, name)
    generator = random.Random(seed)
    payloads = []
    for index in range(cases):
        width = generator.randint(0, max_width - 1)
        payload = {
            f"field-{item}": _random_json(
                generator,
                depth=max_depth - 1,
                max_width=max_width,
                max_string_chars=max_string_chars,
            )
            for item in range(width)
        }
        payload["case"] = index
        payloads.append(payload)
    return tuple(payloads)


def assert_json_bounds(
    value: JsonValue,
    *,
    max_depth: int,
    max_width: int,
    max_string_chars: int,
) -> None:
    """Assert that arbitrary JSON-compatible data stays within declared bounds."""
    _validate_json_limits(max_depth, max_width, max_string_chars)
    _visit_json(value, 0, max_depth, max_width, max_string_chars)


@dataclass(frozen=True, slots=True)
class ChangedFunction:
    qualified_name: str
    first_line: int
    last_line: int


def changed_function_targets(
    source: str,
    changed_lines: Iterable[int],
) -> tuple[ChangedFunction, ...]:
    """Map changed source lines to the narrowest containing functions."""
    if not isinstance(source, str):
        raise TypeError("source must be a string")
    lines = frozenset(changed_lines)
    if any(isinstance(line, bool) or not isinstance(line, int) or line < 1 for line in lines):
        raise ValueError("changed lines must be positive integers")
    tree = ast.parse(source)
    functions = _function_ranges(tree)
    selected = []
    for line in sorted(lines):
        containing = [
            function
            for function in functions
            if function.first_line <= line <= function.last_line
        ]
        if containing:
            selected.append(min(containing, key=lambda item: item.last_line - item.first_line))
    return tuple(sorted(set(selected), key=lambda item: (item.first_line, item.qualified_name)))


def mutmut_function_patterns(
    module: str,
    functions: Iterable[ChangedFunction],
) -> tuple[str, ...]:
    """Return stable changed-function selectors for mutation runners."""
    if not isinstance(module, str) or not module:
        raise ValueError("module must be a non-empty string")
    return tuple(f"{module}.{function.qualified_name}*" for function in functions)


def _validate_health(health: PluginHealth) -> None:
    if not isinstance(health, PluginHealth):
        raise ConformanceError("health() did not return PluginHealth")
    if not isinstance(health.detail, str):
        raise ConformanceError("health detail is not a string")
    _non_negative_timestamp(health.checked_at_ms, "health timestamp")


def _validate_result(request: PluginRequest, result: PluginResult) -> None:
    if not isinstance(result, PluginResult):
        raise ConformanceError("execute() did not return PluginResult")
    if result.job_id != request.job_id:
        raise ConformanceError("result job identity differs from the request")
    if not isinstance(result.status, PluginResultStatus):
        raise ConformanceError("result status is invalid")
    if not isinstance(result.output, Mapping):
        raise ConformanceError("result output is not an object")
    if not isinstance(result.detail, str):
        raise ConformanceError("result detail is not a string")
    _non_negative_timestamp(result.completed_at_ms, "result timestamp")


def _validate_json_limits(max_depth: int, max_width: int, max_string_chars: int) -> None:
    for limit, name in (
        (max_depth, "max_depth"),
        (max_width, "max_width"),
        (max_string_chars, "max_string_chars"),
    ):
        _positive_integer(limit, name)


def _visit_json(
    candidate: JsonValue,
    depth: int,
    max_depth: int,
    max_width: int,
    max_string_chars: int,
) -> None:
    if depth > max_depth:
        raise ConformanceError("JSON value exceeds maximum depth")
    if isinstance(candidate, str) and len(candidate) > max_string_chars:
        raise ConformanceError("JSON string exceeds maximum length")
    if isinstance(candidate, list):
        _visit_json_list(candidate, depth, max_depth, max_width, max_string_chars)
    elif isinstance(candidate, dict):
        _visit_json_object(candidate, depth, max_depth, max_width, max_string_chars)
    elif candidate is not None and not isinstance(candidate, (bool, int, float, str)):
        raise ConformanceError("value is not JSON compatible")
    elif isinstance(candidate, float) and not math.isfinite(candidate):
        raise ConformanceError("JSON number is not finite")


def _visit_json_list(
    candidate: list,
    depth: int,
    max_depth: int,
    max_width: int,
    max_string_chars: int,
) -> None:
    if len(candidate) > max_width:
        raise ConformanceError("JSON array exceeds maximum width")
    for child in candidate:
        _visit_json(child, depth + 1, max_depth, max_width, max_string_chars)


def _visit_json_object(
    candidate: dict,
    depth: int,
    max_depth: int,
    max_width: int,
    max_string_chars: int,
) -> None:
    if len(candidate) > max_width:
        raise ConformanceError("JSON object exceeds maximum width")
    for key, child in candidate.items():
        if not isinstance(key, str):
            raise ConformanceError("JSON object key is not a string")
        if len(key) > max_string_chars:
            raise ConformanceError("JSON object key exceeds maximum length")
        _visit_json(child, depth + 1, max_depth, max_width, max_string_chars)


def _non_negative_timestamp(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConformanceError(f"{name} is not a non-negative integer")


def _random_json(
    generator: random.Random,
    *,
    depth: int,
    max_width: int,
    max_string_chars: int,
) -> JsonValue:
    scalar_factories: tuple[Callable[[], JsonValue], ...] = (
        lambda: None,
        lambda: bool(generator.getrandbits(1)),
        lambda: generator.randint(-1_000_000, 1_000_000),
        lambda: generator.uniform(-1_000_000, 1_000_000),
        lambda: _random_text(generator, max_string_chars),
    )
    if depth <= 0:
        return generator.choice(scalar_factories)()
    kind = generator.randrange(7)
    if kind < len(scalar_factories):
        return scalar_factories[kind]()
    width = generator.randint(0, max_width)
    if kind == 5:
        return [
            _random_json(
                generator,
                depth=depth - 1,
                max_width=max_width,
                max_string_chars=max_string_chars,
            )
            for _index in range(width)
        ]
    return {
        f"key-{index}": _random_json(
            generator,
            depth=depth - 1,
            max_width=max_width,
            max_string_chars=max_string_chars,
        )
        for index in range(width)
    }


def _random_text(generator: random.Random, maximum: int) -> str:
    alphabet = "abcXYZ09-_ ✓"
    return "".join(generator.choice(alphabet) for _index in range(generator.randint(0, maximum)))


def _function_ranges(tree: ast.AST) -> tuple[ChangedFunction, ...]:
    found: list[ChangedFunction] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.parents: list[str] = []

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self.parents.append(node.name)
            self.generic_visit(node)
            self.parents.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._function(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._function(node)

        def _function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
            first_line = min(
                [node.lineno, *(decorator.lineno for decorator in node.decorator_list)]
            )
            found.append(
                ChangedFunction(
                    ".".join([*self.parents, node.name]),
                    first_line,
                    node.end_lineno or node.lineno,
                )
            )
            self.parents.append(node.name)
            self.generic_visit(node)
            self.parents.pop()

    Visitor().visit(tree)
    return tuple(found)


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _positive_number(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{name} must be positive")
