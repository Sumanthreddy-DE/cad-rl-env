"""cad-rl-env taskset: ``eval`` = SimReady concepts_v1 in-vocab, ``train`` = generator."""

from __future__ import annotations

from typing import Literal

import verifiers.v1 as vf
from pydantic import Field

from cad_rl_env.prompt import SYSTEM_PROMPT
from cad_rl_env.task import CadData, CadTask, CadTaskConfig

EVAL_PHRASINGS = ("explicit", "engineer", "intent")
TRAIN_PHRASINGS = ("explicit", "engineer")


class CadConfig(vf.TasksetConfig):
    split: Literal["train", "eval"] = "eval"
    seed: int = 0
    n_train: int = 1000
    families: list[str] = Field(default_factory=lambda: ["plate_holes", "bushing", "flange"])
    phrasings: list[str] | None = None
    """None = all phrasings the split supports."""
    task: CadTaskConfig = Field(default_factory=CadTaskConfig)


def eval_tasks(config: CadConfig) -> list[CadTask]:
    from simready.bench.concepts import load_concepts

    phrasings = config.phrasings or list(EVAL_PHRASINGS)
    bad = [p for p in phrasings if p not in EVAL_PHRASINGS]
    if bad:
        raise ValueError(f"eval split supports {EVAL_PHRASINGS}, got {bad}")
    tasks: list[CadTask] = []
    for concept in load_concepts():
        if not concept.in_vocab:
            continue
        for phrasing in phrasings:
            tasks.append(
                CadTask(
                    CadData(
                        id=f"{concept.concept_id}-{phrasing}",
                        prompt=getattr(concept.prompts, phrasing),
                        system_prompt=SYSTEM_PROMPT,
                        checks=[c.model_dump(mode="json") for c in getattr(concept.checks, phrasing)],
                        expected_solids=concept.solids,
                        reference_spec=concept.reference_spec,
                        split="eval",
                        source_id=concept.concept_id,
                        phrasing=phrasing,
                    ),
                    config.task,
                )
            )
    return tasks


class CadTaskset(vf.Taskset[CadTask, CadConfig]):
    def load(self) -> list[CadTask]:
        if self.config.split == "eval":
            return eval_tasks(self.config)
        raise NotImplementedError("train split lands in Task B5")
