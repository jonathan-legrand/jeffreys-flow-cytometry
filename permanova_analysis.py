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

all_distances = pd.read_csv(DISTANCES_PATH)
batchs = all_distances.batch_i.unique()
a = len(batchs)
N = len(np.unique(all_distances[["subject_i", "subject_j"]].values.ravel()))


# Construct a subject: batch table
# That's what needs to be permuted
mapping = pd.concat([
    all_distances[["subject_i", "batch_i"]].rename(
        columns={"subject_i": "subject", "batch_i": "batch"}
    ),
    all_distances[["subject_j", "batch_j"]].rename(
        columns={"subject_j": "subject", "batch_j": "batch"}
    ),
]).drop_duplicates().set_index("subject")
n_l = mapping.reset_index().groupby("batch").count().rename({"subject":"subject_count"}, axis=1)

def compute_pseudo_F(distances):
    intra_batch_mask = distances.batch_i == distances.batch_j
    SS_total = distances.distance.pow(2).sum() / N

    within_batch = distances.loc[intra_batch_mask].copy()
    within_batch["distance_squared"] = within_batch["distance"] ** 2
    W_l = within_batch.groupby("batch_i")["distance_squared"].sum() / n_l.subject_count

    SS_within = W_l.sum()

    numerator = (SS_total - SS_within) / (a-1)
    denominator = (SS_within) / (N-a)
    pseudo_F = numerator / denominator
    R_squared = 1 - SS_within / SS_total # Effect size
    return pseudo_F, R_squared


for method, distances in all_distances.groupby("method"):
    realized_stat, R_squared = compute_pseudo_F(distances)
    null_stats = []
    for perm_iter in range(N_PERMS):

        null_distances = distances.copy()

        # Permute subject -> batch mapping
        shuffled_mapping = mapping.sample(frac=1, random_state=perm_iter).reset_index(drop=True)

        null_distances["batch_i"] = distances["subject_i"].apply(
            lambda subject_idx: shuffled_mapping.loc[subject_idx, "batch"]
        ).values
        null_distances["batch_j"] = distances["subject_j"].apply(
            lambda subject_idx: shuffled_mapping.loc[subject_idx, "batch"]
        ).values

        null_stat_iter, _ =  compute_pseudo_F(null_distances)
        null_stats.append(null_stat_iter)

    null_stats = np.array(null_stats)
    p_value = (1 + (null_stats >= realized_stat).sum()) / (1 + N_PERMS)
    print(f"Method {method}: p = {p_value} (R squared = {R_squared})")
    
