"""On-demand plugin workers and their idle exit (OMNI-0478).

Nothing here sleeps and nothing needs a real plugin process: the supervisor is
driven through a fake launcher, and the runtime's idle timer is an injected
coroutine the test decides when to resolve.
"""

from __future__ import annotations

from tests.conftest import *  # noqa: F401,F403  (only for pytest plugin discovery)
