from __future__ import annotations

from cad_rl_env.leak import collides, eval_signatures, signature

PLATE = [
    {"type": "bbox_sorted", "ranges": [[4.9, 5.1], [41.9, 42.1], [41.9, 42.1]]},
    {"type": "hole_set", "diameter": [3.3, 3.5], "count": [4, 4], "through": True},
]


def test_signature_uses_range_midpoints() -> None:
    bbox, feats = signature(PLATE)
    assert bbox == (5.0, 42.0, 42.0)
    assert feats == (("hole", 3.4, 4),)


def test_same_part_collides_and_different_part_does_not() -> None:
    near = [dict(PLATE[0], ranges=[[4.9, 5.1], [42.4, 42.6], [41.9, 42.1]]), PLATE[1]]
    far = [dict(PLATE[0], ranges=[[4.9, 5.1], [59.9, 60.1], [41.9, 42.1]]), PLATE[1]]
    assert collides(signature(PLATE), signature(near))
    assert not collides(signature(PLATE), signature(far))


def test_feature_count_mismatch_does_not_collide() -> None:
    three = [PLATE[0], dict(PLATE[1], count=[3, 3])]
    assert not collides(signature(PLATE), signature(three))


def test_eval_signatures_cover_37_concepts() -> None:
    assert len(eval_signatures()) == 37
