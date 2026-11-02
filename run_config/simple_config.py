from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics.pairwise import laplacian_kernel

from source.models.fisher_kernel import FisherEmbedding
from source.scaling import CytoScaler
from source.dataset import (
    PanelChangeWithGEILTestDataset,
)
from source.utils import get_config

config = get_config()

# Matplotlib params
rc = {
    "text.usetex": False,
    "font.family": "serif",
    "font.size": 30,
    "font.serif": ["TeX Gyre Termes", "Nimbus Roman", "Liberation Serif", "Times New Roman", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "axes.labelsize":20,
    "axes.titlesize":20,
    "xtick.labelsize":15,
    "ytick.labelsize":15,
    "legend.fontsize":15,
    "legend.title_fontsize":15,
    "legend.title_fontsize":15,
}


SEED = 0

# The number of cells to sample from each patient
N_CELLS = 1000

# Validation for threshold tuning
VALIDATION_FRACTION = 0.2

# The analysis is run multiple times to have SD estimates
N_RUNS = 20

DISTANCE_CONFIGS = [
    {
        "name": f"{kind}_k-{n_components}_{covariance_type}",
        "kind": kind,
        "parameters": {
            "n_components": n_components,
            "covariance_type": covariance_type,
            "n_samples": 256,
        },
    }
    for kind in ("gmm_jeffreys",)
    for n_components in (1,4,8)
    for covariance_type in ("full","diag")
]

tasks = {
    "NPM_train-GEIL_infer-GEIL-and-ELN": {
        "dataset_class": PanelChangeWithGEILTestDataset,
        "kwargs": dict(
            target_col="NPM",
            n_cells=N_CELLS
        ), # We do not have many cells in some GEIL tubes
        "random_state": SEED
    },
}

models = {
    "fisher_svm": {
        "estimator": make_pipeline(
            CytoScaler(),
            FisherEmbedding(),
            SVC(class_weight="balanced", kernel=laplacian_kernel),
        ),
        "param_grid": dict()
    },
}
