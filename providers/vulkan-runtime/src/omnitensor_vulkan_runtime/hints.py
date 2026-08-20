"""Which opaque references this backend may be told to cite.

Everything else that used to be here — what a selected-text operation is, what
a calendar date looks like, which refusal earns one re-prompt — was workload
policy living inside a GPU adapter. It states its own policy now, beside the
task it belongs to, and reaches the adapter through the `TaskPrompting` port.
"""

from __future__ import annotations

import json

from omnitensor.plugins.generation import GenerationRequest, GenerationTask

from .grounding import citable_references


def citable_hint(task: GenerationTask, request: GenerationRequest) -> str:
    """The trusted sourceRef values, and the instruction to cite only those.

    Never any private text: opaque references only.
    """
    citable = citable_references(task.task_id, request.content_references)
    if not citable:
        return ""
    return (
        "Trusted citable sourceRef values are exactly "
        f"{json.dumps(citable, separators=(',', ':'))}. "
        "Copy only one of these opaque values into each evidence sourceRef; "
        "never cite another fragment. Answer every explicit part of the user's request "
        "that these sources support; preserve names, roles, dates, times, and places.\n"
    )


__all__ = ["citable_hint"]
