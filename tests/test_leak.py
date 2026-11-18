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


def test_generator_shaped_checks_collide_with_eval_parts() -> None:
    from cad_rl_env.generator._checks import EXPLICIT_TOL, bbox_sorted, hole_set, tol

    sigs = eval_signatures()

    # c007 flange: generator emits boss(d1=60) + boss(d2=30), whereas c007 checks only boss(30)
    flange_c007 = [
        bbox_sorted([60, 60, 23]),
        {"type": "boss", "diameter": tol(60, EXPLICIT_TOL), "height_min": 7.9},
        {"type": "boss", "diameter": tol(30, EXPLICIT_TOL), "height_min": 14.9},
        {"type": "bore", "diameter": tol(12, EXPLICIT_TOL), "through": True},
        hole_set(tol(6.6, EXPLICIT_TOL), [(22.5, 0.0), (0.0, 22.5), (-22.5, 0.0), (0.0, -22.5)], EXPLICIT_TOL),
    ]
    assert any(collides(signature(flange_c007), e) for e in sigs)

    # c009 wall plate (100 x 60 x 4, four 5.5 mm corner holes)
    plate_c009 = [
        bbox_sorted([100, 60, 4]),
        hole_set(tol(5.5, EXPLICIT_TOL), [(8.0, 8.0), (92.0, 8.0), (8.0, 52.0), (92.0, 52.0)], EXPLICIT_TOL),
    ]
    assert any(collides(signature(plate_c009), e) for e in sigs)

    # c004 M8 spacer bushing (OD 16, ID 9, L 20)
    bushing_c004 = [
        bbox_sorted([16, 16, 20]),
        {"type": "bore", "diameter": tol(9.0, EXPLICIT_TOL), "through": True},
        {"type": "boss", "diameter": tol(16.0, EXPLICIT_TOL), "height_min": 19.9},
    ]
    assert any(collides(signature(bushing_c004), e) for e in sigs)


def test_boss_count_and_open_bounds_handled() -> None:
    one_boss = [
        {"type": "bbox_sorted", "ranges": [[25.9, 26.1], [49.9, 50.1], [49.9, 50.1]]},
        {"type": "boss", "diameter": [29.9, 30.1], "count": [1, 1]},
    ]
    two_bosses = [
        {"type": "bbox_sorted", "ranges": [[25.9, 26.1], [49.9, 50.1], [49.9, 50.1]]},
        {"type": "boss", "diameter": [29.9, 30.1], "count": [2, 2]},
    ]
    assert not collides(signature(one_boss), signature(two_bosses))

    wildcard_bbox = [{"type": "bbox_sorted", "ranges": [[19.8, 20.2], [None, None], [None, None]]}]
    concrete_bbox = [{"type": "bbox_sorted", "ranges": [[19.9, 20.1], [50.0, 50.2], [50.0, 50.2]]}]
    assert collides(signature(wildcard_bbox), signature(concrete_bbox))

