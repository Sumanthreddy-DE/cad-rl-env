"""cad-rl-env: verifiers v1 text-to-CAD RL environment graded by SimReady."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cad_rl_env.taskset import CadEnv, CadTaskset

__all__ = ["CadEnv", "CadTaskset"]


def __getattr__(name: str):
    if name in ("CadEnv", "CadTaskset"):
        from cad_rl_env import taskset

        return getattr(taskset, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
