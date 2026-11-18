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

    phrasings = list(EVAL_PHRASINGS) if config.phrasings is None else config.phrasings
    if not phrasings:
        raise ValueError("phrasings must not be empty")
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


def train_tasks(config: CadConfig) -> list[CadTask]:
    from cad_rl_env.generator import FAMILIES, make_task

    phrasings = list(TRAIN_PHRASINGS) if config.phrasings is None else config.phrasings
    if not phrasings:
        raise ValueError("phrasings must not be empty")
    bad = [p for p in phrasings if p not in TRAIN_PHRASINGS]
    if bad:
        raise ValueError(f"train split supports {TRAIN_PHRASINGS}, got {bad}")
    if not config.families:
        raise ValueError("families must not be empty")
    unknown = [f for f in config.families if f not in FAMILIES]
    if unknown:
        raise ValueError(f"unknown families {unknown}; known: {sorted(FAMILIES)}")
    tasks: list[CadTask] = []
    for i in range(config.n_train):
        family = config.families[i % len(config.families)]
        phrasing = phrasings[(i // len(config.families)) % len(phrasings)]
        g = make_task(config.seed, family, i, phrasing)
        tasks.append(
            CadTask(
                CadData(
                    id=f"{family}-{config.seed}-{i}-{phrasing}",
                    prompt=g.prompt,
                    system_prompt=SYSTEM_PROMPT,
                    checks=g.checks,
                    reference_spec=g.reference_spec,
                    split="train",
                    source_id=f"{family}-{config.seed}-{i}",
                    phrasing=phrasing,
                    params={"family": family, **g.params},
                ),
                config.task,
            )
        )
    return tasks


class CadTaskset(vf.Taskset[CadTask, CadConfig]):
    def load(self) -> list[CadTask]:
        return eval_tasks(self.config) if self.config.split == "eval" else train_tasks(self.config)
