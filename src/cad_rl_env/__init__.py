"""cad-rl-env: verifiers v1 text-to-CAD RL environment graded by SimReady."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cad_rl_env.taskset import CadTaskset

__all__ = ["CadTaskset"]


def __getattr__(name: str):
    if name == "CadTaskset":
        from cad_rl_env.taskset import CadTaskset

        return CadTaskset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
