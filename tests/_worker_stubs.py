"""Stdlib-only worker stubs for multiprocessing spawn tests in test_grade.py."""

from __future__ import annotations

import time
from typing import Any


def post_ready_hang(spec: dict[str, Any], checks: list[Any], solids: int, out: Any) -> None:
    out.put({"stage": "ready"})
    time.sleep(60)


def broken_checker_worker(spec: dict[str, Any], checks: list[Any], solids: int, out: Any) -> None:
    out.put({"stage": "ready"})
    out.put(
        {
            "stage": "checker_error",
            "error_type": "checker_exception",
            "error": "RuntimeError: boom",
        }
    )
