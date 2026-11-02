from collections import Counter, defaultdict
from math import isclose


def split_retention_curve_by_batch(curve):
    retention_fields = ("correctly_classified", "incorrectly_classified")
    previous_rates = {field: 100.0 for field in retention_fields}
    events = []
    totals_by_batch = defaultdict(Counter)

    for point in sorted(curve, key=lambda item: item["removed_subjects"]):
        changed_fields = [
            field
            for field in retention_fields
            if not isclose(point[field], previous_rates[field])
        ]
        if len(changed_fields) != 1:
            raise ValueError("Each curve point must remove one classified subject")

        removed_class = changed_fields[0]
        batch = point["removed_batch"]
        events.append((point, batch, removed_class))
        totals_by_batch[batch][removed_class] += 1
        previous_rates = {
            field: point[field] for field in retention_fields
        }

    removed_by_batch = defaultdict(Counter)
    curves_by_batch = defaultdict(list)
    for point, batch, removed_class in events:
        removed_by_batch[batch][removed_class] += 1
        rates = {}
        for field in retention_fields:
            total = totals_by_batch[batch][field]
            rates[field] = (
                100 * (total - removed_by_batch[batch][field]) / total
                if total
                else float("nan")
            )

        batch_point = dict(point)
        batch_point.update(rates)
        batch_point["removed_subjects"] = sum(removed_by_batch[batch].values())
        batch_point["retention_gap"] = (
            rates["correctly_classified"] - rates["incorrectly_classified"]
        )
        curves_by_batch[batch].append(batch_point)

    return dict(curves_by_batch)


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
