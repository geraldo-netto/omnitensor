"""How long one plugin's calls may take, decided from what it declares.

Every worker was given the same thirty seconds. That is generous for a
classifier and impossible for a language model: an 8B GGUF has to be read from
disk, uploaded to the GPU, and only then can it produce a first token, so
``ask-selected-files`` failed every request with ``deadline-exceeded: call
exceeded 30 seconds`` — including one against a two-line file, where the whole
budget went on loading the model.

This is the same shape as the memory ceiling before it, which killed a Qwen
worker legitimately holding 831 MB against a 512 MiB limit. A single number
cannot describe every workload, so the number comes from what the plugin
publishes about itself: the format of its artifacts says what class of work it
does, and a plugin may state a budget outright when it knows better.

Nothing here removes a bound. A plugin that hangs is still cut off; it is cut
off at a deadline that matches the work it was asked to do.
"""

from __future__ import annotations

from collections.abc import Mapping

from .budgets import DEFAULT_CALL_TIMEOUT_SECONDS, MAX_CALL_TIMEOUT_SECONDS, WorkerBudgetLimits

# Artifact formats whose plugins load a model before they can answer. GGUF is a
# quantised language model — weights measured in gigabytes, a cold call paying
# for load and inference together.
GENERATIVE_FORMATS = frozenset({"gguf"})
GENERATIVE_CALL_TIMEOUT_SECONDS = 600.0
FLOW_MARGIN_SECONDS = 5.0
BUDGETS_KEY = "budgets"
CALL_TIMEOUT_KEY = "callTimeoutSeconds"


def call_timeout_for(manifest: Mapping[str, object]) -> float:
    """The call deadline this plugin's manifest earns.

    Declared budget first, because a plugin that states a number knows more
    than this file does. Otherwise the artifact formats decide, and a plugin
    that loads no model keeps the short default that has always applied.
    """
    plugin = manifest.get("plugin") if isinstance(manifest, Mapping) else None
    plugin = plugin if isinstance(plugin, Mapping) else {}
    declared = _declared_timeout(plugin.get(BUDGETS_KEY))
    if declared is not None:
        return declared
    if _loads_a_model(plugin.get("artifacts")):
        return GENERATIVE_CALL_TIMEOUT_SECONDS
    return DEFAULT_CALL_TIMEOUT_SECONDS


def limits_for(manifest: Mapping[str, object]) -> WorkerBudgetLimits:
    """This plugin's budgets: its own deadline, and the defaults for the rest."""
    return WorkerBudgetLimits(call_timeout_seconds=call_timeout_for(manifest))


def flow_deadline_for(manifest: Mapping[str, object]) -> float:
    """The admission deadline, kept just wider than the worker's own.

    Two clocks bound the same call: the worker budget and the flow controller
    that admitted it. Equal numbers make which one fires a race, and the
    worker's message is the useful one — it names the call that overran rather
    than the profile that was waiting. So the flow waits a little longer and
    reports only when the worker has stopped reporting at all.
    """
    return min(call_timeout_for(manifest) + FLOW_MARGIN_SECONDS, MAX_CALL_TIMEOUT_SECONDS)


def _declared_timeout(candidate: object) -> float | None:
    if not isinstance(candidate, Mapping):
        return None
    value = candidate.get(CALL_TIMEOUT_KEY)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not 0 < value <= MAX_CALL_TIMEOUT_SECONDS:
        # Out of range is a manifest that means nothing rather than a manifest
        # that means "no limit"; the derived answer applies instead.
        return None
    return float(value)


def _loads_a_model(candidate: object) -> bool:
    if not isinstance(candidate, (list, tuple)):
        return False
    return any(
        isinstance(artifact, Mapping)
        and str(artifact.get("format", "")).lower() in GENERATIVE_FORMATS
        for artifact in candidate
    )
