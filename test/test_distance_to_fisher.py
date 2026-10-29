import numpy as np

from source.distance_analysis import compute_misclassification_rates_from_curve


def test_curve_reconstruction_counts_boundary_and_recovers_fold_change():
    curve = [
        {
            "distance_threshold": 4.0,
            "removed_subjects": 2,
            "incorrectly_classified": 50.0,
        },
        {
            "distance_threshold": 3.0,
            "removed_subjects": 3,
            "incorrectly_classified": 25.0,
        },
        {
            "distance_threshold": 1.0,
            "removed_subjects": 10,
            "incorrectly_classified": 0.0,
        },
    ]

    result = compute_misclassification_rates_from_curve(
        curve, accuracy_before_rejection=0.6, threshold=3.0
    )

    assert result["misclassification_rate_above_threshold"] == 1.0
    assert result["misclassification_rate_at_or_below_threshold"] == 0.25
    assert result["misclassification_rate_fold_change"] == 4.0
