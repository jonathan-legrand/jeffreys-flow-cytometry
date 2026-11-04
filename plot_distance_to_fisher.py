# %%
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from run_config.simple_config import rc
from source.distance_analysis import (
    compute_misclassification_rates_from_curve,
    split_retention_curve_by_batch,
)

sns.set_theme(
    context="paper",
    style="whitegrid",
    rc=rc,
)
# For correctly vs incorrectly patients,
# we want a good vs bad kind of palette, so blue first
set1_colors = list(sns.color_palette("Set1", n_colors=2))[::-1]

with open("output/distance_to_fisher_results.json", "r") as input_file:
    saved_results = json.load(input_file)
classification_analysis = pd.DataFrame.from_records(
    saved_results["classification_analysis"]
)
results = pd.DataFrame.from_records(saved_results["results"])
curves = saved_results["curves"]
best_method = results.groupby("name")["criterion_on_validation"].mean().idxmax()

best_method_results = results[results["name"] == best_method].copy()


# %%

misclassification_rates = [
    compute_misclassification_rates_from_curve(
        curves[int(row.run)][best_method],
        row.accuracy_before_rejection,
        row.optimal_threshold,
    ) for row in best_method_results.itertuples()
]
best_method_results["misclassification_rate_fold_change"] = [
    rates["misclassification_rate_fold_change"]
    for rates in misclassification_rates
]
best_method_results["accuracy_rejected"] = [
    1 - rates["misclassification_rate_above_threshold"]
    for rates in misclassification_rates
]
best_method_results["misclassification_rate_retained"] = (
    1 - best_method_results["accuracy_after_rejection"]
)
best_method_results["misclassification_rate_rejected"] = (
    1 - best_method_results["accuracy_rejected"]
)


best_method_results["accuracy_delta"] = (
    best_method_results["accuracy_after_rejection"]
    - best_method_results["accuracy_before_rejection"]
)
accuracy_summary = best_method_results[[
    "accuracy_before_rejection",
    "accuracy_after_rejection",
    "accuracy_rejected",
    "accuracy_delta",
]].agg(["mean", "std"]).T
print(f"Evaluation accuracy for best method ({best_method}):")
print(accuracy_summary)
misclassification_summary = best_method_results[[
    "misclassification_rate_retained",
    "misclassification_rate_rejected",
]].agg(["mean", "std"]).T
print("Misclassification rates (1 - accuracy) at best threshold:")
print(misclassification_summary)
fold_change_summary = best_method_results[
    "misclassification_rate_fold_change"
].agg(["mean", "std"])
print(
    "Misclassification rate above threshold was "
    f"{fold_change_summary['mean']:.2f}-fold higher than at or below "
    f"the threshold (SD {fold_change_summary['std']:.2f}) across runs."
)

acc = classification_analysis.rename(
    columns={"Correctly classified": "correct"}
)
per_batch = acc.groupby(["run", "Batch"])["correct"].mean().unstack("Batch")
per_batch["Overall"] = acc.groupby("run")["correct"].mean()
summary = per_batch.agg(["mean", "std"]).T
print(summary)
# %%

plt.subplots(figsize=(6, 3))
sns.barplot(
    results[results.kind == "gmm_jeffreys"],
    x="n_components",
    y="criterion",
    hue="covariance_type",
    palette=sns.cubehelix_palette(
        n_colors=2,
        start=9,
        rot=1,
        dark=0.5,
        light=0.9,
    ),
    edgecolor="black",
)
plt.ylabel("Mean retention gap")
plt.xlabel("Number of subject GMM components")
plt.savefig("figures/methods-evaluation.eps", bbox_inches="tight")
plt.close()

# %%
# Gather rejection curves for best method
best_threshold = results[results.name == best_method].optimal_threshold.median()
curve_data = []
batch_curve_data = []
for run_index, curve_result in enumerate(curves):
    best_curve = pd.DataFrame(curve_result[best_method])
    best_curve["run"] = run_index
    curve_data.append(best_curve)
    for batch, batch_curve in split_retention_curve_by_batch(
        curve_result[best_method]
    ).items():
        batch_curve_frame = pd.DataFrame(batch_curve)
        batch_curve_frame["run"] = run_index
        batch_curve_data.append(batch_curve_frame)
curve_data = pd.concat(curve_data)
batch_curve_data = pd.concat(batch_curve_data, ignore_index=True)

# Align runs within each batch so each facet aggregates matching thresholds.
aligned_curves = []
for batch, batch_data in batch_curve_data.groupby("removed_batch"):
    run_threshold_ranges = batch_data.groupby("run")["distance_threshold"].agg(
        ["min", "max"]
    )
    shared_min = run_threshold_ranges["min"].max()
    shared_max = run_threshold_ranges["max"].min()
    shared_thresholds = np.sort(
        batch_data.loc[
            batch_data["distance_threshold"].between(shared_min, shared_max),
            "distance_threshold",
        ].unique()
    )
    for _, run_curve in batch_data.groupby("run"):
        run_curve = run_curve.sort_values("distance_threshold")
        indices = np.searchsorted(
            run_curve["distance_threshold"].to_numpy(),
            shared_thresholds,
            side="left",
        )
        aligned_curve = run_curve.iloc[indices].copy()
        aligned_curve["distance_threshold"] = shared_thresholds
        aligned_curves.append(aligned_curve)
plot_data = pd.concat(aligned_curves, ignore_index=True)

# %%
# Display curves and best threshold
plot_data["Rejected"] = plot_data["distance_threshold"] > best_threshold
melt = plot_data.melt(
    id_vars=["distance_threshold", "Rejected", "run", "removed_batch"],
    value_vars=["correctly_classified", "incorrectly_classified"],
    var_name="Outcome",
    value_name="Proportion kept [%]",
)
g = sns.relplot(
    melt,
    x="distance_threshold",
    y="Proportion kept [%]",
    hue="Outcome",
    style="Outcome", # Style as well for grayscale printing
    estimator="mean",
    errorbar=("ci", 95),
    palette=set1_colors,
    kind="line",
    col="removed_batch",
    facet_kws=dict(sharex=False, sharey=True),
    legend=None
)

batch_titles = {"ELN": "Panel-shifted", "GEIL": "Time-shifted"}

for batch, ax in zip(g.col_names, g.axes[0]):
    ax.vlines(
        x=best_threshold,
        ymin=0,
        ymax=100,
        linestyles="dotted",
        color="black",
        linewidth=2,
        label="Rejection threshold",
    )

    ax.set_xlabel("Jeffreys divergence [nats]")
    ax.set_ylabel("Proportion retained [%]")
    ax.set_title(batch_titles.get(batch, batch))


legend_data = {
    "Correctly classified": Line2D(
        [], [], color=set1_colors[0], linewidth=2
    ),
    "Incorrectly classified": Line2D(
        [], [], color=set1_colors[1], linestyle="dashed", linewidth=2
    ),
    "Rejection threshold": Line2D(
        [], [], color="black", linestyle="dotted", linewidth=2
    ),
}

g.axes[0, 0].legend(
    handles=list(legend_data.values()),
    labels=list(legend_data.keys()),
    loc="lower right",
    bbox_to_anchor=(1.1, 0., 0., 0.)
)

plt.savefig("figures/rejection_curves.png", dpi=800)
plt.close()
# %%

highest_distance_row = curve_data[
    curve_data.distance_threshold < best_threshold
].sort_values(
    by="distance_threshold", ascending=False
).reset_index().iloc[0]
print("Threshold", best_threshold)
print("Correctly classified kept at threshold:", highest_distance_row.correctly_classified)
print("Incorrectly classified kept at threshold:", highest_distance_row.incorrectly_classified)
# %%
# How many subjects are rejected in total at the optimal threshold?
rejected_per_run = curve_data[curve_data.distance_threshold > best_threshold].sort_values(
    "distance_threshold"
).groupby("run")["removed_subjects"].first()
print(f"Subjects rejected at best threshold: {rejected_per_run.mean():.1f} (SD {rejected_per_run.std():.1f})")
subjects_per_run = curve_data.groupby("run")["removed_subjects"].max()
print(f"Subjects evaluated per run: {subjects_per_run.mean():.1f} (SD {subjects_per_run.std():.1f})")


# %%
