def compute_misclassification_rates_from_curve(
    curve, accuracy_before_rejection, threshold
):

    n_total = max(point["removed_subjects"] for point in curve)
    n_incorrect = round((1 - accuracy_before_rejection) * n_total)
    next_boundaries = [
        point
        for point in curve
        if point["distance_threshold"] > threshold
    ]

    boundary = min(
        next_boundaries, key=lambda point: point["distance_threshold"]
    )
    n_above = boundary["removed_subjects"]
    n_incorrect_at_or_below = round(
        n_incorrect * boundary["incorrectly_classified"] / 100
    )

    n_at_or_below = n_total - n_above
    above_rate = (n_incorrect - n_incorrect_at_or_below) / n_above
    at_or_below_rate = n_incorrect_at_or_below / n_at_or_below
    fold_change = above_rate / at_or_below_rate
    return {
        "misclassification_rate_above_threshold": above_rate,
        "misclassification_rate_at_or_below_threshold": at_or_below_rate,
        "misclassification_rate_fold_change": fold_change,
    }
