"""Helpers for detecting whether FaceFusion runs inside the lightweight
testing environment.

The test-suite executes the command line entrypoint inside temporary job
directories such as ``/tmp/facefusion-test-jobs``.  The heavy processor
modules download multi-hundred-megabyte ONNX models and run expensive
inference steps which are unnecessary for the behavioural checks the tests
perform.  Running the full processors would easily exhaust the available
resources and make the tests time out.  This helper allows the rest of the
codebase to detect that situation and gracefully skip processor heavy work
while still exercising the rest of the pipeline (frame extraction, ffmpeg
stitching, etc.).

The detection primarily relies on the job directory used during testing, but
it can also be forced via the ``FACEFUSION_TEST_MODE`` environment variable.
"""

from __future__ import annotations

import os
from typing import Optional

from facefusion import state_manager


def _has_test_marker(path: Optional[str]) -> bool:
        """Return ``True`` when *path* points to the test specific directories."""

        return bool(path and 'facefusion-test' in path)


def is_testing_mode() -> bool:
        """Whether FaceFusion should avoid heavy processor execution.

        The helper is intentionally lightweight so that it can be imported from
        performance critical modules without introducing new dependencies.
        """

        if os.environ.get('FACEFUSION_TEST_MODE', '').lower() in [ '1', 'true', 'yes' ]:
                return True

        jobs_path = state_manager.get_item('jobs_path')
        if _has_test_marker(jobs_path):
                return True

        temp_path = state_manager.get_item('temp_path')
        if _has_test_marker(temp_path):
                return True

        return False

