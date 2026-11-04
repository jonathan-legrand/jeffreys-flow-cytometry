"""Load and preview vector-form NPM pairwise distances and perform stat
tests to determine whether our distances detect batch effects
We use a PERMANOVA. Ref:
Anderson, M. J. (2017). Permutational Multivariate Analysis of Variance (PERMANOVA). In Wiley StatsRef: Statistics Reference Online (pp. 1–15). John Wiley & Sons, Ltd. https://doi.org/10.1002/9781118445112.stat07841
"""
# %%

import pandas as pd
import numpy as np

# Min possible P-value is 1/(1+N_PERMS)
N_PERMS = 1000

# Output of compare_npm_distances.py
DISTANCES_PATH = "output/npm_pairwise_distances_k-8_20261029_104154.csv"
#DISTANCES_PATH = "output/npm_pairwise_distances_k-1_20261103_164506.csv"

all_distances = pd.read_csv(DISTANCES_PATH)
# Construct a subject: batch table
# That's what needs to be permuted
mapping = pd.concat([
    all_distances[["subject_i", "batch_i"]].rename(
        columns={"subject_i": "subject", "batch_i": "batch"}
    ),
    all_distances[["subject_j", "batch_j"]].rename(
        columns={"subject_j": "subject", "batch_j": "batch"}
    ),
]).drop_duplicates().set_index("subject")["batch"]
subject_positions = pd.Series(np.arange(len(mapping)), index=mapping.index)
group_codes, groups = pd.factorize(mapping, sort=False)
group_sizes = np.bincount(group_codes, minlength=len(groups))
group_to_code = {group: code for code, group in enumerate(groups)}
N = len(mapping)
a = len(groups)

subject_i = all_distances["subject_i"].map(subject_positions).to_numpy()
subject_j = all_distances["subject_j"].map(subject_positions).to_numpy()
distance_squared = all_distances["distance"].to_numpy() ** 2

def compute_pseudo_F(distances_squared, index_i, index_j, labels):
    labels_i = labels[index_i]
    labels_j = labels[index_j]
    within_mask = labels_i == labels_j
    within_sums = np.bincount(
        labels_i[within_mask],
        weights=distances_squared[within_mask],
        minlength=a,
    )
    ss_total = distances_squared.sum() / N
    ss_within = (within_sums / group_sizes).sum()

    numerator = (ss_total - ss_within) / (a - 1)
    denominator = ss_within / (N - a)
    pseudo_F = numerator / denominator
    R_squared = 1 - ss_within / ss_total
    return pseudo_F, R_squared


for method, distances in all_distances.groupby("method"):
    row_indices = distances.index.to_numpy()
    method_distances_squared = distance_squared[row_indices]
    method_subject_i = subject_i[row_indices]
    method_subject_j = subject_j[row_indices]
    realized_stat, R_squared = compute_pseudo_F(
        method_distances_squared,
        method_subject_i,
        method_subject_j,
        group_codes,
    )
    null_stats = []
    for perm_iter in range(N_PERMS):
        permuted_labels = mapping.sample(
            frac=1, random_state=perm_iter
        ).map(group_to_code).to_numpy()
        null_stat_iter, _ = compute_pseudo_F(
            method_distances_squared,
            method_subject_i,
            method_subject_j,
            permuted_labels,
        )
        null_stats.append(null_stat_iter)

    null_stats = np.array(null_stats)
    p_value = (1 + (null_stats >= realized_stat).sum()) / (1 + N_PERMS)
    print(f"Method {method}: p = {p_value} (R squared = {R_squared})")
    
