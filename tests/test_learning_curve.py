from __future__ import annotations

from d2draft.learning_curve import automatic_sizes, convergence_assessment


def _point(size: int, values: list[float]) -> dict:
    return {
        "train_matches": size,
        "runs": [
            {"seed": seed, "phase_3_hit_at_10": value}
            for seed, value in enumerate(values)
        ],
    }


def test_automatic_sizes_includes_full_training_pool() -> None:
    assert automatic_sizes(8054) == [1000, 2000, 4000, 6000, 8000, 8054]


def test_automatic_sizes_adds_dense_late_checkpoints() -> None:
    assert automatic_sizes(32750)[-4:] == [20000, 25000, 30000, 32750]


def test_tiny_remainder_does_not_decide_plateau() -> None:
    points = [
        _point(4000, [0.30, 0.30, 0.30]),
        _point(6000, [0.31, 0.31, 0.31]),
        _point(8000, [0.32, 0.32, 0.32]),
        _point(8054, [0.319, 0.320, 0.321]),
    ]
    assessment = convergence_assessment(points)
    assert assessment["status"] == "not_yet_demonstrated"
    assert assessment["minimum_increment_matches_for_assessment"] == 1000
