from __future__ import annotations

import pytest

from text_to_cad_rl.taskset import CadConfig, CadTaskset


def load(**kw):
    return CadTaskset(CadConfig(split="train", **kw)).load()


def test_train_size_and_unique_ids() -> None:
    tasks = load(n_train=60)
    assert len(tasks) == 60 and len({t.data.id for t in tasks}) == 60


def test_train_round_robins_families_and_phrasings() -> None:
    tasks = load(n_train=6)
    assert [t.data.params["family"] for t in tasks[:3]] == ["plate_holes", "bushing", "flange"]
    assert {t.data.phrasing for t in tasks} == {"explicit", "engineer"}


def test_train_is_reproducible() -> None:
    assert [t.data.prompt for t in load(n_train=9)] == [t.data.prompt for t in load(n_train=9)]


def test_train_rejects_intent() -> None:
    with pytest.raises(ValueError):
        load(n_train=3, phrasings=["intent"])


def test_train_rejects_unknown_family() -> None:
    with pytest.raises(ValueError):
        load(n_train=3, families=["gear"])
