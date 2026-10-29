# %%
import json

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import laplacian_kernel
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

from run_config.simple_config import (
    DISTANCE_CONFIGS,
    N_CELLS,
    N_RUNS,
    SEED,
    VALIDATION_FRACTION,
    models,
    tasks,
)
from source.dataset.cross_domain import PanelChangeWithGEILTestDataset
from source.scaling import CytoScaler
from source.models import FisherEmbedding

def monte_carlo_kullback_leibler_divergence(gmm_a, gmm_b, n_samples=256):
    x_a, _ = gmm_a.sample(n_samples)
    kl_divergence = gmm_a.score(x_a) - gmm_b.score(x_a)
    return kl_divergence

def kl_divergence(reference_gmm, subject_gmms):
    divergences = []
    for subject_gmm in subject_gmms:
        divergence = monte_carlo_kullback_leibler_divergence(reference_gmm, subject_gmm)
        divergences.append(divergence)
    
    return np.array(divergences)

def jeffreys_divergence(reference_gmm, subject_gmms):
    divergences = []
    for subject_gmm in subject_gmms:
        kl_a_b = monte_carlo_kullback_leibler_divergence(reference_gmm, subject_gmm)
        kl_b_a = monte_carlo_kullback_leibler_divergence(subject_gmm, reference_gmm)
        divergences.append(kl_a_b + kl_b_a)
    return np.array(divergences)


name_dist_mapping = {
    "gmm_jeffreys": jeffreys_divergence,
}

task = tasks["NPM_train-GEIL_infer-GEIL-and-ELN"]
dataset_class = task["dataset_class"]

tasks = {
    "NPM_train-GEIL_infer-GEIL-and-ELN": {
        "dataset_class": PanelChangeWithGEILTestDataset,
        "kwargs": dict(
            target_col="NPM",
            n_cells=N_CELLS
        ),
    }
}

def compute_retention_curve(distances, correct, top_fraction=0.8):
    """
    Average the correct-retention minus error-retention gap over farthest cutoffs.
    """
    n_correct = int(correct.sum())
    n_incorrect = int((~correct).sum())
    max_removed_rank = int(np.ceil(len(distances) * top_fraction))
    curve = []
    removed = 0
    for threshold in np.sort(np.unique(distances))[::-1]:
        if removed >= max_removed_rank:
            break
        removed_mask = distances >= threshold
        kept_correct = int(np.sum(correct & ~removed_mask))
        kept_incorrect = int(np.sum(~correct & ~removed_mask))
        correct_retained = kept_correct / n_correct * 100
        incorrect_retained = kept_incorrect / n_incorrect * 100
        curve.append(
            {
                "distance_threshold": float(threshold),
                "removed_subjects": int(removed_mask.sum()),
                "correctly_classified": correct_retained,
                "incorrectly_classified": incorrect_retained,
                "retention_gap": correct_retained - incorrect_retained,
            }
        )
        removed = int(removed_mask.sum())

    criterion = float(np.mean([point["retention_gap"] for point in curve]))
    return {
        "criterion": criterion,
        "n_cutoffs": len(curve),
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
        "curve": curve,
    }

def run_analysis(seed):

    train_set = dataset_class(
        split="train", random_state=seed, **task["kwargs"]
    )
    X_train, y_train, _ = train_set.make_X_y_meta()
    test_set = dataset_class(
        split="test", random_state=seed, **task["kwargs"]
    )
    X_test, y_test, tabular_test = test_set.make_X_y_meta()

    # Train, infer, show that incorrectly classified are from other batch?
    classifier = make_pipeline(
        CytoScaler(),
        FisherEmbedding(),
        SVC(class_weight="balanced", kernel=laplacian_kernel),
    )

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = (y_pred == y_test).mean()

    print(f"Accuracy: {accuracy:.2f}")

    classification_analysis = pd.DataFrame(tabular_test.protocol.values, columns=["Batch"])
    classification_analysis["Correctly classified"] = (y_pred == y_test)
    print(classification_analysis.groupby("Batch").mean())

    # We need a validation set to tune the rejection threshold
    validation_indices, evaluation_indices = train_test_split(
        np.arange(len(y_test)),
        test_size=1 - VALIDATION_FRACTION,
        random_state=seed,
        stratify=y_test,
    )
    validation_indices = np.sort(validation_indices)
    evaluation_indices = np.sort(evaluation_indices)
    
    fisher_embedding = classifier.named_steps["fisherembedding"]
    scaler = classifier.named_steps["cytoscaler"]
    clf_gmm = fisher_embedding.gmm_

    # Compute distances between the classifiers
    # GMM and the subjects GMM
    gmm_cache = dict()
    curves = dict()
    results = []
    for distance_config in DISTANCE_CONFIGS:
        params = distance_config["parameters"]
        test_scaled = scaler.transform(X_test)
        K = params["n_components"]
        cov_type = params["covariance_type"]

        # Avoid fitting gmm multiple times
        # since jeffrey and ot both use gmms
        if (K, cov_type) in gmm_cache.keys():
            subject_gmms = gmm_cache[(K,cov_type)]
        else:
            subject_gmms = []
            for test_cells in test_scaled:
                target_gmm = GaussianMixture(
                    n_components=K,
                    covariance_type=cov_type,
                    random_state=seed
                ).fit(test_cells)
                subject_gmms.append(target_gmm)

            gmm_cache[(K, cov_type)] = subject_gmms

        distances = name_dist_mapping[distance_config["kind"]](clf_gmm, subject_gmms)

        # Tune rejection threshold
        retention_analysis_validation = compute_retention_curve(
            distances[validation_indices],
            y_test[validation_indices] == y_pred[validation_indices],
            top_fraction=1
        )
        curve_data = pd.DataFrame(retention_analysis_validation["curve"])
        optimal_threshold = curve_data.sort_values(
            by="retention_gap", ascending=False
        ).reset_index(drop=True).loc[0, "distance_threshold"] # That but just on validation

        evaluation_distances = distances[evaluation_indices]
        evaluation_correct = (
            y_test[evaluation_indices] == y_pred[evaluation_indices]
        )
        retained = evaluation_distances <= optimal_threshold
        accuracy_before_rejection = evaluation_correct.mean()
        accuracy_after_rejection = evaluation_correct[retained].mean()
        
        # Compute and export rejection results on test data
        retention_analysis_evaluation = compute_retention_curve(
            evaluation_distances,
            evaluation_correct,
            top_fraction=1,
        )

        results.append(
            {
                "name": distance_config["name"],
                "kind": distance_config["kind"],
            }
            | distance_config["parameters"]
            | {
                "criterion_on_validation": retention_analysis_validation["criterion"],
                "criterion": retention_analysis_evaluation["criterion"],
                "optimal_threshold": optimal_threshold,
                "accuracy_before_rejection": accuracy_before_rejection,
                "accuracy_after_rejection": accuracy_after_rejection,
            }
        )
        # We store curves here to keep results flat
        curves[distance_config["name"]] = retention_analysis_evaluation["curve"]
    return classification_analysis, pd.DataFrame(results), curves


if __name__ == "__main__":
    classification_analysis, results, curves = [], [], []
    for seed_offset in range(N_RUNS):
        run_clf_analysis, run_results, run_curves = run_analysis(SEED + seed_offset)
        run_results["run"] = seed_offset
        run_clf_analysis["run"] = seed_offset
        results.append(run_results)
        curves.append(run_curves)
        classification_analysis.append(run_clf_analysis)

    output = {
        "classification_analysis": pd.concat(classification_analysis).to_dict(orient="records"),
        "results": pd.concat(results).to_dict(orient="records"),
        "curves": curves,
    }
    with open("output/distance_to_fisher_results.json", "w") as output_file:
        json.dump(output, output_file, indent=2)
