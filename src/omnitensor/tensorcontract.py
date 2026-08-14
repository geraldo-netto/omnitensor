"""What a model expects of its input, as its publisher states it.

A shape mismatch used to surface as ``RuntimeError: ncnn extraction failed for
output prob`` — raised inside the executor, after the job had been admitted,
queued, and dispatched, with nothing in it a caller could act on.  Admission
could already refuse such a job; it had nothing to check against, because no
manifest said what the model wanted.

Now ``requirements.model.tensorContract`` says it, and the split in that
document is the whole design:

*Shape and dtype are enforced.*  They are facts about the graph, the runtime
can compare them to what arrived, and disagreeing means the job cannot work.
Refusing at admission turns an opaque native error into a stable refusal in
the caller's own call.

*Preprocessing is not enforced, and cannot be.*  Channel order, mean, and
scale describe how the numbers were produced, and the runtime deliberately
decodes nothing — it never sees a picture, only a buffer, and a buffer
normalised the wrong way is a perfectly valid tensor that infers successfully
and means nothing.  So it is published for whoever holds the pixels and is
never checked here.  Stating it is still the point: nothing in an artifact
records it, an ncnn ``.param`` declares its input size and is silent about the
rest, and without it every caller guesses.

*Absent means no check.*  Every manifest written before this field still loads
and still runs, and a publisher who knows the shape but not the normalisation
can say so by declaring the one and omitting the other.

Deliberately not :class:`omnitensor.plugins.artifact_compatibility.TensorSpec`:
that type names a tensor and belongs to the aggregate that compares publisher
requirements against host capability, which is a different question asked at a
different time.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

MAX_INPUTS = 8


@dataclass(frozen=True, slots=True)
class ResizeSpec:
    """How a picture becomes the declared shape.

    Published, never enforced, like the rest of ``preprocess``: this service
    decodes nothing and so cannot tell a bilinear resize from a bicubic one.
    It exists because the alternative is each consumer picking its own and the
    two disagreeing about the same picture — a disagreement that leaves a
    model's confident answers intact and silently reorders its uncertain ones,
    which is the worst shape for a defect to have.
    """

    resample: str
    fit: str

    def document(self) -> dict:
        return {"filter": self.resample, "fit": self.fit}


@dataclass(frozen=True, slots=True)
class PreprocessSpec:
    """How the publisher produced the values, for a caller holding pixels."""

    channel_order: str
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    resize: ResizeSpec | None = None

    def document(self) -> dict:
        described = {
            "channelOrder": self.channel_order,
            "mean": list(self.mean),
            "scale": list(self.scale),
        }
        if self.resize is not None:
            described["resize"] = self.resize.document()
        return described


@dataclass(frozen=True, slots=True)
class InputSpec:
    """One declared input: what is enforced, and what is only published."""

    shape: tuple[int, ...]
    dtype: str
    layout: str | None = None
    preprocess: PreprocessSpec | None = None

    @property
    def element_count(self) -> int:
        return math.prod(self.shape)

    def document(self) -> dict:
        described: dict = {"shape": list(self.shape), "dtype": self.dtype}
        if self.layout is not None:
            described["layout"] = self.layout
        if self.preprocess is not None:
            described["preprocess"] = self.preprocess.document()
        return described


def declared_inputs(model: Mapping | None) -> tuple[InputSpec, ...] | None:
    """The contract a manifest's model declares, or ``None`` when it declares none.

    Reads only what the schema already validated, so a malformed contract is a
    manifest that never loaded rather than a check that half ran.
    """
    if not isinstance(model, Mapping):
        return None
    contract = model.get("tensorContract")
    if not isinstance(contract, Mapping):
        return None
    declared = contract.get("inputs")
    if not isinstance(declared, Sequence) or not declared:
        return None
    if len(declared) > MAX_INPUTS:
        raise ValueError(f"tensor contract exceeds {MAX_INPUTS} inputs")
    return tuple(_input_spec(item) for item in declared)


def _resize_spec(preprocess: Mapping) -> ResizeSpec | None:
    resize = preprocess.get("resize")
    return ResizeSpec(resize["filter"], resize["fit"]) if isinstance(resize, Mapping) else None


def _preprocess_spec(preprocess: object) -> PreprocessSpec | None:
    if not isinstance(preprocess, Mapping):
        return None
    return PreprocessSpec(
        preprocess["channelOrder"],
        tuple(float(value) for value in preprocess["mean"]),
        tuple(float(value) for value in preprocess["scale"]),
        _resize_spec(preprocess),
    )


def _input_spec(item: Mapping) -> InputSpec:
    return InputSpec(
        tuple(item["shape"]),
        item["dtype"],
        item.get("layout"),
        _preprocess_spec(item.get("preprocess")),
    )


def measured_shape(tensor: object) -> tuple[int, ...] | None:
    """The shape of a nested-list tensor, or ``None`` when it is ragged.

    A ragged input is not reported as a shape mismatch: it is refused by the
    tensor validator with a better message than a shape comparison could give.
    """
    if not isinstance(tensor, list):
        return None
    return _rectangular_shape(tensor)


def _rectangular_shape(value: object) -> tuple[int, ...] | None:
    if not isinstance(value, list):
        return ()
    if not value:
        return None
    item_shape = _rectangular_shape(value[0])
    if item_shape is None:
        return None
    if any(_rectangular_shape(item) != item_shape for item in value[1:]):
        return None
    return (len(value), *item_shape)


def contract_error(
    specs: Sequence[InputSpec] | None,
    shapes: Sequence[tuple[tuple[int, ...] | None, str | None]],
) -> str | None:
    """Why these inputs cannot satisfy this contract, or ``None``.

    ``shapes`` pairs each supplied input's shape with its dtype; either may be
    ``None`` where the caller did not state it and it could not be measured,
    and an unknown value is never reported as a disagreement.
    """
    if not specs:
        return None
    if len(shapes) != len(specs):
        return (
            f"the model declares {len(specs)} input"
            f"{'' if len(specs) == 1 else 's'} and {len(shapes)} "
            f"{'was' if len(shapes) == 1 else 'were'} supplied"
        )
    for index, (spec, (shape, dtype)) in enumerate(zip(specs, shapes, strict=True)):
        if shape is not None and tuple(shape) != spec.shape:
            return (
                f"input {index} has shape {list(shape)} and the model declares "
                f"{list(spec.shape)}"
            )
        if dtype is not None and dtype != spec.dtype:
            return f"input {index} is {dtype} and the model declares {spec.dtype}"
    return None
