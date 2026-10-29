from pathlib import Path
import pandas as pd
import numpy as np
from source.utils import get_config
import numpy as np
import json

from sklearn.preprocessing import LabelEncoder
from .eln import ELNDataset, ELNSplitSet, list_to_array
from .geil import GEILTube, GeilTestSet


config = get_config()
MPATH = config["ELN_META"]
DPATH = config["ELN_FCS"]

common_dyes_geil = [
    "FS-A",
    "SS-A",
    "CD13 ECD",
    "CD33 PC5.5",
    "CD34 PC7",
    "CD45 KO",
    "CD7 APC700",
]
common_dyes_eln = [
    "FS INT",
    "SS INT",
    "CD13 PE",
    "CD33 PC7",
    "CD34 FITC",
    "CD45 KO",
    "CD7 PC5.5",
]

class PanelChangeDataset:
    """
    That dataset is for training and evaluating models that need to adapt
    to a change of fluorochrome. Here, we have the example of training on
    GEIL tube A and testing on ELN.
    We don't want to inherit from CytometryDataset because otherwise
    the getattr overriding would not work. Maybe that's not great, but
    it's the most pragmatic option.
    """
    def __init__(self, split="train", **kwargs):
        self.group_col = None
        if split == "train":
            self._ds = GEILTube(
                config["GEIL_FCS"],
                config["GEIL_META"],
                tube="A",
                **kwargs
            )
            self.common_dyes = common_dyes_geil
        elif split =="test":
            self._ds = ELNDataset(
                config["ELN_FCS"],
                config["ELN_META"],
                **kwargs
            )
            self.common_dyes = common_dyes_eln
        else:
            raise ValueError(f"Invalid split value {split}")

    def __getattr__(self, name):
        return getattr(self._ds, name)

    def make_X_y_meta(self, to_array=True):
        """
        We want to keep marker intersection between ELN and GEIL protocols
        """
        X_complete, y, meta = self._ds.make_X_y_meta(to_array=False)
        X= [x[self.common_dyes] for x in X_complete]
        if to_array:
            X = list_to_array(X)
        return X, y, meta


class PanelChangeWithGEILTestDataset:
    """Train on GEIL and test on the combined ELN and GEIL test sets."""
    def __init__(self, split="train", **kwargs):
        self.group_col = None
        if split == "train":
            self._datasets = [(GEILTube(
                config["GEIL_FCS"],
                config["GEIL_META"],
                tube="A",
                **kwargs
            ), common_dyes_geil, 0)]
        elif split == "test":
            self.group_col = "batch_idx"
            self._datasets = [
                (ELNDataset(config["ELN_FCS"], config["ELN_META"], **kwargs), common_dyes_eln, 1),
                (GeilTestSet(
                    config["GEIL_TEST_FCS"],
                    config["GEIL_TEST_META"],
                    tube="A",
                    **kwargs
                ), common_dyes_geil, 0),
            ]
        else:
            raise ValueError(f"Invalid split value {split}")

    def make_X_y_meta(self, to_array=True):
        X, y, meta = [], [], []
        for dataset, common_dyes, batch_idx in self._datasets:
            dataset_X, dataset_y, dataset_meta = dataset.make_X_y_meta(to_array=False)
            X.extend(sample[common_dyes] for sample in dataset_X)
            y.append(dataset_y)
            dataset_meta = dataset_meta.copy()
            dataset_meta["batch_idx"] = batch_idx
            dataset_meta["protocol"] = dataset.protocol
            meta.append(dataset_meta)

        if to_array:
            X = list_to_array(X)
        return X, np.concatenate(y), pd.concat(meta, ignore_index=True)


class TemporalMismatchDataset:
    """
    That dataset is for training and evaluating models that need to adapt
    to domain shifts. Here, we have the example of training on
    Camilla's data and testing on the FCS that has been collected a few years
    before. We hypothesized that some performance drop for tube B was due to domain
    mismatch
    """
    def __init__(self, split="train", **kwargs):
        if split == "train":
            self._ds = GEILTube(
                config["GEIL_FCS"],
                config["GEIL_META"],
                **kwargs
            )
        elif split =="test":
            self._ds = GeilTestSet(
                config["GEIL_TEST_FCS"],
                config["GEIL_TEST_META"],
                **kwargs
            )
        else:
            raise ValueError(f"Invalid split value {split}")

    def __getattr__(self, name):
        return getattr(self._ds, name)
