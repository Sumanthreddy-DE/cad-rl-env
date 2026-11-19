"""text-to-cad-rl: verifiers v1 text-to-CAD RL environment graded by SimReady."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from text_to_cad_rl.taskset import CadEnv, CadTaskset

__all__ = ["CadEnv", "CadTaskset"]


def __getattr__(name: str):
    if name in ("CadEnv", "CadTaskset"):
        from text_to_cad_rl import taskset

        return getattr(taskset, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
