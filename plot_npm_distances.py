"""Plot precomputed NPM distance embeddings from a CSV file."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns

CSV_PATH = Path("output/npm_distance_embeddings_k-8_20261029_104154.csv")


rc = {
    "text.usetex": False,
    "font.family": "serif",
    "legend.fontsize": 13,
    "legend.title_fontsize": 13,
    "font.serif": ["TeX Gyre Termes", "Nimbus Roman", "Liberation Serif", "Times New Roman", "STIXGeneral"],
    "mathtext.fontset": "stix",   # Times-compatible math
}

sns.set_theme(
    context="paper",
    style="darkgrid",
    rc=rc,
)


OUTCOME_LABELS = {
    0: "NPM1-WT",
    1: "NPM1-mut",
}
DISTANCE_METHODS = {
    #"OT-GMM": ("ot_gmm_mds1", "ot_gmm_mds2"),
    "Jeffreys GMM": ("jeffreys_gmm_mds1", "jeffreys_gmm_mds2"),
}

categories_mapping = {
    "GEIL train": "Source",
    "GEIL": "Time-shifted",
    "ELN": "Panel-shifted"
}



data = pd.read_csv(CSV_PATH)

figure, axes = plt.subplots(1, len(DISTANCE_METHODS), figsize=(7*len(DISTANCE_METHODS), 6), squeeze=False)
outcomes = data["outcome"].to_numpy()
groups = data["group"].astype(str).to_numpy()
categories, group_codes = np.unique(groups, return_inverse=True)
palette = sns.cubehelix_palette(
    n_colors=3, start=2, rot=1, dark=0.5, light=0.9
)
colormap = ListedColormap(palette)
outcome_values = np.unique(outcomes)
#markers = ("o", "^")
group_handles = [
    Line2D(
        [],
        [],
        marker="o",
        markeredgecolor="black",
        markersize=40,
        linestyle="None",
        color=palette[index],
        label=categories_mapping[category]
    )
    for index, category in enumerate(categories)
]
#outcome_handles = [
#    Line2D(
#        [], [], linestyle="None",
#        color="black", label=OUTCOME_LABELS.get(outcome, str(outcome)),
#    )
#    for index, outcome in enumerate(outcome_values)
#]

for axis, (method, columns) in zip(axes[0], DISTANCE_METHODS.items()):
    embedding = data[list(columns)].to_numpy()
    for outcome_index, outcome in enumerate(outcome_values):
        mask = outcomes == outcome
        axis.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=group_codes[mask],
            cmap=colormap,
            vmin=-0.5,
            vmax=len(categories) - 0.5,
            s=24,
            edgecolors="black"
        )

    #axis.set_title(method, )
    axis.set_xlabel("MDS 1", fontsize=13)
    axis.set_ylabel("MDS 2", fontsize=13)
    axis.set_xticks([-1, 0, 1])
    axis.set_yticks([-1, 0, 1])

axes[0, 0].legend(
    handles=group_handles,# + outcome_handles,
    title="Dataset",
    title_fontproperties={'weight':'bold'},
    loc="upper right",
)
for image_format in ("eps", "png"):
    output_path = Path("figures/mds").with_suffix(f".{image_format}")
    figure.savefig(output_path, bbox_inches='tight')
    print("Figure exported to", output_path)
plt.show()


# %%
