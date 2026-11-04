from pathlib import Path
import pandas as pd
import numpy as np
from source.utils import get_config
import numpy as np
import joblib
import json

from sklearn.base import BaseEstimator


config = get_config()
MPATH = config["ELN_META"]
DPATH = config["ELN_FCS"]

LABEL_MAPPING = {
            "WT": 0,
            "ITD": 1,
            "ITD + TKD": 1,
            "ITD+TKD": 1,
            "TKD": 0,
        }

NPM_MAPPING = {'neg': 0,
 'NPM type D': 1,
 'NPM type A': 1,
 'NPM type B': 1,
 'NPM1 muté': 1,
 'NPM muté type A': 1,
 'NPM1 muté type Nm': 1,
 'NPM1 muté type A': 1,
 'NPM type Ht': 1,
 'NPM type Om non suivable': 1,
 'NPM non suivable': 1,
 'NPM1 muté type Z (pas de suivi MRD possible bio mol)': 1,
 'muté type 4/Ht': 1,
 'muté type A': 1
}

class CytometryDataset:
    def __init__(
            self,
            dpath:str,
            mpath:str=MPATH,
            scaler:BaseEstimator=None,
            n_cells:int=5000,
            resample_cells:bool=False,
            random_state:int=1234,
            return_patient_id:bool=False,
            tabular_features:tuple=None,
            target_col:str="FLT3",
        ):
        if isinstance(dpath, str):
            dpath = Path(dpath)
        self._build_fc_dataset(dpath, mpath, n_cells)
        self.n_cells = n_cells
        self.rng = np.random.default_rng(random_state)
        self.scaler = scaler
        self.resample_cells = resample_cells
        self.return_patient_id = return_patient_id
        self.tabular_features = tabular_features
        self.target_col = target_col

        if target_col == "FLT3":
            self.label_mapping = LABEL_MAPPING
        elif target_col == "NPM":
            self.label_mapping = NPM_MAPPING
        else:
            raise NotImplementedError("Unsupported target column")

        print("before", self.__len__())
        self._validate_n_tubes()
        self._drop_nf()
        print("after", self.__len__())

    def _build_fc_dataset(self, dpath, mpath, n_cells):
        raise NotImplementedError()
        

    def _validate_n_tubes(self):
        raise NotImplementedError()
        
    def _drop_nf(self):
        for _, row in self.metadata.iterrows():
            target = row[self.target_col]
            if "NF" in target or "NR" in target:
                patient = row.Patient
                print(f"{patient} has NF status, removing")
                self.fc_dataset.pop(patient)
                drop_idx = self.metadata[self.metadata.Patient == patient].index
                self.metadata.drop(index=drop_idx, inplace=True)
    
    def get_df(self, index):
        raise NotImplementedError()


    def get_numpy(self, index):
        raise NotImplementedError()

    def get_label(self, row):
        return self.label_mapping[row[self.target_col]]

    def __len__(self):
        return len(self.fc_dataset)

    def make_X_y_meta(self, to_array=True):
        X, y, metadata = [], [], []
        for i in range(len(self)):
            features, label, row = self.get_df(i)
            if "CD7 PC5.5" not in features.columns:
                features.rename({"CD7 PerCP C5.5": "CD7 PC5.5"}, inplace=True, axis=1)
            X.append(features)
            y.append(label)
            metadata.append(row)
        y = np.vstack(y).ravel()
        metadata = pd.DataFrame(metadata).reset_index(drop=True)
        if to_array is True:
            X = list_to_array(X)
        return X, y, metadata


class ELNDataset(CytometryDataset):
    protocol = "ELN"
    def __init__(self, dpath = DPATH, mpath = MPATH, scaler = None, n_cells = 5000, resample_cells = False, random_state = 1234, return_patient_id = False, tabular_features = None, target_col = "FLT3"):
        super().__init__(dpath, mpath, scaler, n_cells, resample_cells, random_state, return_patient_id, tabular_features, target_col)

    def _build_fc_dataset(self, dpath, mpath, n_cells):
        metadata = pd.read_csv(mpath, sep="\t", index_col=0, dtype={"Patient": str})
        self.metadata = metadata
        
        # Filter fc_dataset to include only patients present in the metadata
        patients = set(self.metadata.Patient.astype(str).to_list())
        fc_dataset = dict()
        
        def load_sample(sample_path):
            """Load a single sample file and return (patient_id, sample) if patient is in patients."""
            sample_format = sample_path.name.split(".")[-1]
            if sample_format == "csv":
                sample = pd.read_csv(sample_path, index_col=0)
            elif sample_format == "parquet":
                sample = pd.read_parquet(sample_path)
            elif sample_format == "fcs":
                import flowkit as fk # flowkit is not required in env.yml
                fk_sample = fk.Sample(str(sample_path))  # Load all events
                sample = fk_sample.as_dataframe(
                    source="raw",
                    subsample=False,
                    col_multi_index=False,
                    col_names=fk_sample.pns_labels
                )
            else:
                raise ValueError(f"Unrecognized sample format {sample_format}")

            patient_id = sample_path.name.split("_")[0].split("-")[1]
            if patient_id in patients:
                return patient_id, sample
            return None
        
        # Collect all sample paths
        sample_paths = list(dpath.iterdir())
        
        # Load samples in parallel
        results = joblib.Parallel(n_jobs=4, verbose=1)(
            joblib.delayed(load_sample)(sample_path) for sample_path in sample_paths
        )
        
        # Build fc_dataset from results
        for result in results:
            if result is not None:
                patient_id, sample = result
                fc_dataset[patient_id] = sample

        print(f"Loaded {len(fc_dataset)} fc samples.")
        
        self.fc_dataset = fc_dataset
    
    def _validate_n_tubes(self):
        pass

    def get_df(self, index):
        row = self.metadata.iloc[index, :]
        sample = self.fc_dataset[row.Patient]
        
        # Since all samples are now DataFrames, handle subsampling directly
        if self.resample_cells:
            # Randomly sample n_cells rows
            sampled_indices = self.rng.choice(
                len(sample), size=self.n_cells, replace=False
            )
            events = sample.iloc[sampled_indices]
        else:
            # Use all events, potentially subsample to n_cells if needed
            if len(sample) > self.n_cells:
                sampled_indices = self.rng.choice(
                    len(sample), size=self.n_cells, replace=False
                )
                events = sample.iloc[sampled_indices]
            else:
                events = sample
        
        # Clean up columns
        events = events.drop(
            ["TIME", " ", "", "FL7 INT", "APC700", "FS PEAK LIN"],
            axis=1,
            errors="ignore"
        )
        
        label = self.get_label(row)
        return events, label, row

    def get_numpy(self, index):
        events, label, row = self.get_df(index)
        label = self.get_label(row)
        features = events.values
        if self.scaler is not None:
            features = self.scaler.transform(features).squeeze()
        return features, label, row

class ELNSplitSet(ELNDataset):
    def __init__(
            self,
            dpath = DPATH,
            mpath = MPATH,
            scaler = None,
            n_cells = 5000,
            resample_cells = False,
            random_state = 1234,
            return_patient_id = False,
            tabular_features = None,
            target_col = "FLT3",
            split="train"
        ):
        self.split = split
        self.group_col = None # If we group by year we end up with a test fold of 4 individuals and that's not great
        super().__init__(dpath, mpath, scaler, n_cells, resample_cells, random_state, return_patient_id, tabular_features, target_col)

    def _build_fc_dataset(self, dpath, mpath, n_cells):
        metadata = pd.read_csv(mpath, sep="\t", index_col=0, dtype={"Patient": str})
        clinisight = metadata["Numero clinisight"].astype(str).to_list()
        years = np.array([cl[:4] for cl in clinisight])
        metadata["years"] = years

        # We don't want our splits to be random, but rather
        # to be temporal, to assess whether our classifiers
        # can predict actual future samples
        # We use year 2022 as test year, and previous years
        # as training
        if self.split == "test":
            target_years = {"2022"}
        elif self.split == "train":
            target_years = set(np.unique(years)) - {"2022"}
        else:
            raise ValueError("Invalid split name ", self.split)

        # Keep only metadata rows that belong to the target years
        mask = np.isin(years, list(target_years))
        filtered_metadata = metadata[mask].copy()
        filtered_metadata.drop_duplicates(subset="Patient", inplace=True)
        self.metadata = filtered_metadata.reset_index(drop=True)

        # Filter fc_dataset to include only patients present in the filtered metadata
        patients = set(self.metadata.Patient.astype(str).to_list())
        fc_dataset = dict()
        
        def load_sample(sample_path):
            """Load a single sample file and return (patient_id, sample) if patient is in patients."""
            sample_format = sample_path.name.split(".")[-1]
            if sample_format == "csv":
                sample = pd.read_csv(sample_path, index_col=0)
            elif sample_format == "parquet":
                sample = pd.read_parquet(sample_path)
            elif sample_format == "fcs":
                import flowkit as fk # flowkit is not required in env.yml
                fk_sample = fk.Sample(str(sample_path))  # Load all events
                sample = fk_sample.as_dataframe(
                    source="raw",
                    subsample=False,
                    col_multi_index=False,
                    col_names=fk_sample.pns_labels
                )
            else:
                raise ValueError(f"Unrecognized sample format {sample_format}")

            patient_id = sample_path.name.split("_")[0].split("-")[1]
            if patient_id in patients:
                return patient_id, sample
            return None
        
        # Collect all sample paths
        sample_paths = list(dpath.iterdir())
        
        # Load samples in parallel
        results = joblib.Parallel(n_jobs=4, verbose=1)(
            joblib.delayed(load_sample)(sample_path) for sample_path in sample_paths
        )
        
        # Build fc_dataset from results
        for result in results:
            if result is not None:
                patient_id, sample = result
                fc_dataset[patient_id] = sample

        print(f"Split '{self.split}': kept {len(self.metadata)} metadata rows and {len(fc_dataset)} fc samples.")

        self.fc_dataset = fc_dataset
        
def list_to_array(X):
    X_numpy = [x.to_numpy() for x in X]
    return np.stack(X_numpy)
