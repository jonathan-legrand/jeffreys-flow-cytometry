"""Compute pairwise Jeffreys GMM divergences for the NPM dataset."""
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.manifold import MDS
from sklearn.mixture import GaussianMixture

from run_config.simple_config import SEED
from run_distance_to_fisher import jeffreys_divergence
from source.dataset.cross_domain import PanelChangeWithGEILTestDataset
from source.scaling import CytoScaler


# Match tex document font
N_COMPONENTS = 8
DISTANCE_METHODS = {
    "Jeffreys GMM": jeffreys_divergence,
}
OUTCOME_LABELS = {
    0: "NPM1-WT",
    1: "NPM1-mut",
}

def load_npm_dataset():
    samples = []
    outcomes = []
    groups = []
    train_count = 0

    for split in ("train", "test"):
        dataset = PanelChangeWithGEILTestDataset(
            split=split,
            target_col="NPM",
            n_cells=1000,
            random_state=SEED,
        )
        X, y, metadata = dataset.make_X_y_meta()
        samples.extend(X)
        outcomes.extend(y.tolist())
        if split == "train":
            groups.extend(["GEIL train"] * len(X))
            train_count = len(samples)
        else:
            groups.extend(metadata["protocol"].tolist())

    return (
        np.asarray(samples),
        np.asarray(outcomes),
        np.asarray(groups),
        train_count,
    )


def fit_gmms(X):
    gmms = []
    for sample in X:
        gmms.append(
            GaussianMixture(
                n_components=N_COMPONENTS,
                random_state=SEED,
                max_iter=100,
            ).fit(sample)
        )

    return gmms


def pairwise_gmm_distances(gmms, distance_function):
    distances = np.zeros((len(gmms), len(gmms)))
    for first in range(len(gmms)):
        for second in range(first + 1, len(gmms)):
            distance = distance_function(gmms[first], [gmms[second]])[0]
            distances[first, second] = distance
            distances[second, first] = distance
    return distances


def main():
    X, outcomes, groups, train_count = load_npm_dataset()
    print(f"Preparing NPM: {len(X)} subjects", flush=True)
    scaler = CytoScaler().fit(X[:train_count])
    X_scaled = scaler.transform(X)

    print("Fitting subject GMMs", flush=True)
    gmms = fit_gmms(X_scaled)

    distances = {}
    for method, distance_function in DISTANCE_METHODS.items():
        print(f"Computing {method} distances", flush=True)
        distances[method] = pairwise_gmm_distances(gmms, distance_function)

    plot_data = pd.DataFrame(
        {
            "subject_index": np.arange(len(outcomes)),
            "outcome": outcomes,
            "group": groups,
        }
    )
    for method, distance_matrix in distances.items():
        print(f"Computing {method} embedding", flush=True)
        embedding = MDS(
            n_components=2,
            metric="precomputed",
            metric_mds=method != "Jeffreys GMM",
            random_state=SEED,
            init="classical_mds",
            n_init=1,
        ).fit_transform(distance_matrix)
        prefix = "ot_gmm" if method == "OT-GMM" else "jeffreys_gmm"
        plot_data[f"{prefix}_mds1"] = embedding[:, 0]
        plot_data[f"{prefix}_mds2"] = embedding[:, 1]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Export distances in vector form for stat analysis
    subject_i, subject_j = np.triu_indices(len(outcomes), k=1)
    pairwise_data = pd.concat(
        [
            pd.DataFrame(
                {
                    "method": method,
                    "subject_i": subject_i,
                    "batch_i": groups[subject_i],
                    "outcome_i": outcomes[subject_i],
                    "subject_j": subject_j,
                    "batch_j": groups[subject_j],
                    "outcome_j": outcomes[subject_j],
                    "distance": distance_matrix[subject_i, subject_j],
                }
            )
            for method, distance_matrix in distances.items()
        ],
        ignore_index=True,
    )
    pairwise_path = f"output/npm_pairwise_distances_k-{N_COMPONENTS}_{timestamp}.csv"
    pairwise_data.to_csv(pairwise_path, index=False)
    print("Pairwise distances exported to", pairwise_path)

    output_path = f"output/npm_distance_embeddings_k-{N_COMPONENTS}_{timestamp}.csv"
    plot_data.to_csv(output_path, index=False)
    print("Plot data exported to", output_path)


if __name__ == "__main__":
    main()
