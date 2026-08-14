"""Compatibility facade for the neutral telemetry recorder."""

from .. import telemetry_recorder as _recorder

DEFAULT_MAX_SEGMENT_BYTES = _recorder.DEFAULT_MAX_SEGMENT_BYTES
DEFAULT_MAX_SEGMENTS = _recorder.DEFAULT_MAX_SEGMENTS
MAX_FEATURES = _recorder.MAX_FEATURES
MAX_NAME_LENGTH = _recorder.MAX_NAME_LENGTH
MAX_PROFILE_LENGTH = _recorder.MAX_PROFILE_LENGTH
RECORD_VERSION = _recorder.RECORD_VERSION
FeatureRow = _recorder.FeatureRow
RecorderError = _recorder.RecorderError
TelemetryRecorder = _recorder.TelemetryRecorder
feature_matrix = _recorder.feature_matrix
validated_features = _recorder.validated_features
