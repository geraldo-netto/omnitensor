"""Compatibility exports for the renamed job-results module.

Import from :mod:`omnitensor.plugins.job_results` in new code.  These explicit
aliases remain for one release and keep legacy imports and pickle globals valid.
"""

from .job_results import (
    DEFAULT_MAX_RETAINED_JOBS,
    DEFAULT_RESULT_TTL_SECONDS,
    MAX_RETAINED_JOBS_LIMIT,
    JobRecord,
    JobResultError,
    JobResultStore,
)

__all__ = [
    "DEFAULT_MAX_RETAINED_JOBS",
    "DEFAULT_RESULT_TTL_SECONDS",
    "MAX_RETAINED_JOBS_LIMIT",
    "JobRecord",
    "JobResultError",
    "JobResultStore",
]
