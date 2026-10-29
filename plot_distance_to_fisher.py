# %%
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D


rc = {
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["TeX Gyre Termes", "Nimbus Roman", "Liberation Serif", "Times New Roman", "STIXGeneral"],
    "mathtext.fontset": "stix",
}

sns.set_theme(
    context="paper",
    style="darkgrid",
    rc=rc,
)
# For correctly vs incorrectly patients,
# we want a good vs bad kind of palette, so blue first
set1_colors = list(sns.color_palette("Set1", n_colors=2))[::-1]

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
    return fold_change


with open("output/distance_to_fisher_results.json", "r") as input_file:
    saved_results = json.load(input_file)
classification_analysis = pd.DataFrame.from_records(
    saved_results["classification_analysis"]
)
results = pd.DataFrame.from_records(saved_results["results"])
curves = saved_results["curves"]
best_method = (
    results.groupby("name")["criterion_on_validation"].mean().idxmax()
)
best_method_results = results[results["name"] == best_method].copy()

best_method_results["misclassification_rate_fold_change"] = [
    compute_misclassification_rates_from_curve(
        curves[int(row.run)][best_method],
        row.accuracy_before_rejection,
        row.optimal_threshold,
    ) for row in best_method_results.itertuples()
]


best_method_results["accuracy_delta"] = (
    best_method_results["accuracy_after_rejection"]
    - best_method_results["accuracy_before_rejection"]
)
accuracy_summary = best_method_results[[
    "accuracy_before_rejection",
    "accuracy_after_rejection",
    "accuracy_delta",
]].agg(["mean", "std"]).T
print(f"Evaluation accuracy for best method ({best_method}):")
print(accuracy_summary)
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

x = np.linspace(0, 2*np.pi)
df = pd.DataFrame(x, columns=["x"])
df["sin"] = np.sin(x)
df["cos"] = np.cos(x)
df["2cos"] = 2 * np.cos(x)
df = df.melt(id_vars="x")
sns.lineplot(
    df,
    x="x",
    y="value",
    style="variable",
    markers=True
)
plt.close()

# %%
# Gather rejection curves for best method
best_threshold = results[results.name == best_method].optimal_threshold.median()
curve_data = []
for run_index, curve_result in enumerate(curves):
    best_curve = pd.DataFrame(curve_result[best_method])
    best_curve["run"] = run_index
    curve_data.append(best_curve)
curve_data = pd.concat(curve_data)

# Put all runs on the same reference distance scale
# because otherwise seaborn won't display error shades
run_threshold_ranges = curve_data.groupby("run")["distance_threshold"].agg(
    ["min", "max"]
)
shared_min = run_threshold_ranges["min"].max()
shared_max = run_threshold_ranges["max"].min()
shared_thresholds = np.sort(
    curve_data.loc[
        curve_data["distance_threshold"].between(shared_min, shared_max),
        "distance_threshold",
    ].unique()
)
aligned_curves = []
for _, run_curve in curve_data.groupby("run"):
    run_curve = run_curve.sort_values("distance_threshold")
    indices = np.searchsorted(
        run_curve["distance_threshold"].to_numpy(), shared_thresholds, side="left"
    )
    aligned_curve = run_curve.iloc[indices].copy()
    aligned_curve["distance_threshold"] = shared_thresholds
    aligned_curves.append(aligned_curve)
plot_data = pd.concat(aligned_curves, ignore_index=True)

# %%
# Display curves and best threshold
plot_data["Rejected"] = plot_data["distance_threshold"] > best_threshold
melt = plot_data.melt(
    id_vars=["distance_threshold", "Rejected", "run"],
    value_vars=["correctly_classified", "incorrectly_classified"],
    var_name="Outcome",
    value_name="Proportion kept [%]",
)
ax = sns.lineplot(
    melt,
    x="distance_threshold",
    y="Proportion kept [%]",
    hue="Outcome",
    style="Outcome", # Style as well for grayscale printing
    estimator="mean",
    errorbar=("ci", 95),
    palette=set1_colors
)
curve_handles = ax.get_legend().get_lines()

ax.vlines(
    x=best_threshold,
    ymin=0,
    ymax=100,
    linestyles="dotted",
    color="black",
    label="Tuned rejection threshold",
)
ax.legend(
    handles=[
        *curve_handles,
        Line2D([], [], color="black", linestyle="--"),
    ],
    loc="lower right",
    labels=[
        "Correctly classified",
        "Incorrectly classified",
        "Tuned rejection threshold",
    ]
)
plt.xlabel("Jeffreys divergence [nats]", fontsize=13)
plt.ylabel("Proportion kept [%]", fontsize=13)
plt.savefig("figures/rejection_curves.png", dpi=800)
plt.close()

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
