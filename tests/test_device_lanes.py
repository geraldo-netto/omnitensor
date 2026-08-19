"""The two collaborators the service stopped owning: lanes and resolution."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from omnitensor.artifact_readiness import ArtifactResolver
from omnitensor.device_lanes import DeviceLanes
from omnitensor.plugins.artifacts import ArtifactReference, ArtifactResolution

REFERENCE = ArtifactReference("sample-model", "1.0.0", "gguf", "a" * 64, ())


class DeviceAwareExecutors(dict):
    """The real executor collection's shape: it knows one lane per device."""

    def for_device(self, device_id):
        return {"gpu": f"executor-for-{device_id or 'any'}"}

    def lane_key(self, backend, device_id):
        return f"{backend}-{device_id}" if device_id else backend

    def device_id(self, backend, device_id):
        return device_id or None

    def executor_for_device(self, backend, device_id):
        return f"{backend}:{device_id}" if device_id == "gpu-renderD129" else None


class TestALaneFollowsTheChosenDevice:
    def lanes(self, executors, choices):
        return DeviceLanes(lambda: executors, choices.get)

    def test_the_chosen_card_names_the_queue_and_the_identity(self):
        lanes = self.lanes(DeviceAwareExecutors(), {"watcher": "gpu-renderD129"})

        assert lanes.lane("watcher", "gpu") == "gpu-gpu-renderD129"
        assert lanes.identity("watcher", "gpu") == "gpu-renderD129"
        assert lanes.executors_for("watcher") == {"gpu": "executor-for-gpu-renderD129"}
        assert lanes.executor_for("gpu", "gpu-renderD129") == "gpu:gpu-renderD129"

    def test_a_profile_that_chose_nothing_queues_on_the_backend(self):
        lanes = self.lanes(DeviceAwareExecutors(), {})

        assert lanes.lane("watcher", "gpu") == "gpu"
        assert lanes.identity("watcher", "gpu") is None

    def test_a_plain_mapping_of_executors_still_answers(self):
        # An older build, or a test, hands over a dict with no device knowledge.
        # Every question then has exactly one honest answer: the backend itself.
        executors = {"gpu": "the-only-gpu-executor"}
        lanes = self.lanes(executors, {"watcher": "gpu-renderD129"})

        assert lanes.lane("watcher", "gpu") == "gpu"
        assert lanes.identity("watcher", "gpu") == "gpu"
        assert lanes.executors_for("watcher") == executors
        assert lanes.executor_for("gpu", "gpu-renderD129") == "the-only-gpu-executor"
        assert lanes.executor_for("tpu", "tpu-pcie-0") is None

    def test_lanes_read_the_executors_that_exist_now(self):
        # Rediscovery replaces the whole executor set; a lane resolved against
        # the set from ten seconds ago would name a device that is gone.
        current = {"gpu": "first"}
        lanes = DeviceLanes(lambda: current, {}.get)
        assert lanes.executor_for("gpu", "any") == "first"

        current = {"gpu": "second"}
        assert lanes.executor_for("gpu", "any") == "second"


class CountingStore:
    def __init__(self, reason="ready"):
        self.calls = []
        self._reason = reason

    def resolve(self, reference):
        self.calls.append(reference)
        return ArtifactResolution(True, Path("/weights.gguf"), self._reason, 17)


class TestResolutionIsRememberedButNotStale:
    def resolver(self, store, root=None, workloads=None):
        return ArtifactResolver(
            root,
            store,
            workloads_of=lambda: workloads or {},
            plugins_of=tuple,
        )

    def test_without_a_store_the_refusal_says_why(self):
        answer = self.resolver(None).resolve("sample-model")

        assert answer.ready is False
        assert answer.reason == "no artifact store is configured for this service"

    def test_an_undeclared_artifact_is_refused_before_the_store_is_asked(self):
        store = CountingStore()

        answer = self.resolver(store).resolve("sample-model")

        assert answer.reason == "no manifest declares a sha256 for this artifact"
        assert store.calls == []

    def test_the_same_file_is_digested_once(self, tmp_path):
        store = CountingStore()
        weights = tmp_path / "sample-model" / "1.0.0"
        weights.mkdir(parents=True)
        (weights / "model.gguf").write_bytes(b"x" * 17)
        resolver = self.resolver(store, root=tmp_path)

        first = resolver.cached("sample-model", REFERENCE)
        second = resolver.cached("sample-model", REFERENCE)

        assert first is second
        assert len(store.calls) == 1

    def test_a_replaced_weights_file_is_digested_again(self, tmp_path):
        store = CountingStore()
        weights = tmp_path / "sample-model" / "1.0.0"
        weights.mkdir(parents=True)
        artifact = weights / "model.gguf"
        artifact.write_bytes(b"x" * 17)
        resolver = self.resolver(store, root=tmp_path)

        resolver.cached("sample-model", REFERENCE)
        artifact.write_bytes(b"y" * 21)
        resolver.cached("sample-model", REFERENCE)

        assert len(store.calls) == 2

    def test_a_replaced_companion_is_digested_again(self, tmp_path):
        """The `.param` was stamped and its weights were not, so swapping the
        weights kept answering "ready" about the model that was retired."""
        store = CountingStore()
        reference = ArtifactReference(
            "sample-model", "1.0.0", "ncnn", "a" * 64, (("model.bin", "b" * 64),)
        )
        weights = tmp_path / "sample-model" / "1.0.0"
        weights.mkdir(parents=True)
        (weights / "model.param").write_bytes(b"x" * 17)
        companion = weights / "model.bin"
        companion.write_bytes(b"x" * 17)
        resolver = self.resolver(store, root=tmp_path)

        resolver.cached("sample-model", reference)
        resolver.cached("sample-model", reference)
        assert len(store.calls) == 1

        companion.write_bytes(b"y" * 21)
        resolver.cached("sample-model", reference)

        assert len(store.calls) == 2

    def test_an_absent_companion_is_never_remembered_as_an_answer(self, tmp_path):
        store = CountingStore()
        reference = ArtifactReference("sample-model", "1.0.0", "ncnn", "a" * 64, ())
        weights = tmp_path / "sample-model" / "1.0.0"
        weights.mkdir(parents=True)
        (weights / "model.param").write_bytes(b"x" * 17)
        resolver = self.resolver(store, root=tmp_path)

        resolver.cached("sample-model", reference)
        resolver.cached("sample-model", reference)

        assert len(store.calls) == 2

    def test_a_different_store_answers_for_itself(self, tmp_path):
        first = CountingStore("from the first store")
        weights = tmp_path / "sample-model" / "1.0.0"
        weights.mkdir(parents=True)
        (weights / "model.gguf").write_bytes(b"x" * 17)
        resolver = self.resolver(first, root=tmp_path)
        assert resolver.cached("sample-model", REFERENCE).reason == "from the first store"

        resolver.store = CountingStore("from the second store")

        assert resolver.cached("sample-model", REFERENCE).reason == "from the second store"

    def test_a_plugin_reference_goes_straight_to_the_store(self):
        store = CountingStore()

        assert self.resolver(store).resolve_plugin(REFERENCE).ready is True
        assert store.calls == [REFERENCE]

    def test_a_plugin_manifest_declares_the_reference_a_workload_does_not(self):
        plugin = SimpleNamespace(
            manifest={
                "plugin": {
                    "artifacts": [
                        {
                            "id": "sample-model",
                            "version": "1.0.0",
                            "format": "gguf",
                            "sha256": "a" * 64,
                        }
                    ]
                }
            }
        )
        resolver = ArtifactResolver(
            None,
            CountingStore(),
            workloads_of=dict,
            plugins_of=lambda: (plugin,),
        )

        assert resolver.declared_reference("sample-model") == REFERENCE
        assert resolver.declared_reference("another-model") is None
