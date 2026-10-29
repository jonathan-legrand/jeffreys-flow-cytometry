
import joblib
import numpy as np
import pandas as pd
from source.dataset import CytometryDataset
from source.dataset.eln import list_to_array

pns_dict_camilla = {'A': ['FS-A',
  'SS-A',
  'CD65 FITC',
  'CD14 PE',
  'CD13 ECD',
  'CD33 PC5.5',
  'CD34 PC7',
  'CD117 APC',
  'CD7 APC700',
  'CD11b APC750',
  'CD16 PB',
  'CD45 KO'],
 'B': ['FS-A',
  'SS-A',
  'CD64 FITC',
  'CD10 PE',
  'CD4 ECD',
  'CD33 PC5.5',
  'CD34 PC7',
  'CD123 APC',
  'CD56 APC700',
  'CD19 APC750',
  'CD38 PB',
  'CD45 KO'],
 'C': ['FS-A',
  'SS-A',
  'CD36 FITC',
  'CD61 PE',
  'ECD',
  'CD33 PC5.5',
  'CD34 PC7',
  'CD2 APC',
  'APC700',
  'CD71 APC750',
  'HLA-DR PB',
  'CD45 KO']}


pns_dict = {'A': ['FS-A',
  'SS-A',
  'CD65 FITC',
  'CD14 PE',
  'CD13 ECD',
  'CD33 PC5.5',
  'CD34 PC7',
  'CD117 APC',
  'CD7 APC700',
  'CD11b APC750',
  'CD16 PB',
  'CD45 KO'],
 'B': ['FS-A',
  'SS-A',
  'CD64 FITC',
  'CD10 PE',
  'CD4 ECD',
  'CD33 PC5.5',
  'CD34 PC7',
  'CD123 APC',
  'CD56 APC700',
  'CD19 APC750',
  'CD38PB',
  'CD45 KO'],
 'C': ['FS-A',
  'SS-A',
  'CD36 FITC',
  'CD61 PE',
  'ECD',
  'CD33 PC5.5',
  'CD34 PC7',
  'CD2 APC',
  'APC700',
  'CD71 APC750',
  'HLA-DR PB',
  'CD45 KO']}

        
        
class GEILTube(CytometryDataset):
    protocol = "GEIL"
    def __init__(self, dpath, mpath = ..., tube="A", scaler = None, n_cells = 5000, resample_cells = False, random_state = 1234, return_patient_id = False, tabular_features = None, target_col = "FLT3"):
        self.tube = tube
        super().__init__(dpath, mpath, scaler, n_cells, resample_cells, random_state, return_patient_id, tabular_features, target_col)
        try:
            self.metadata["years"] = self.metadata.analyse.apply(lambda x: x.year)
            self.group_col = "years"
        except AttributeError:
            print("No keyword for temporal split, setting groups to None")
            self.group_col = None

    def _build_fc_dataset(self, dpath, mpath, n_cells):
        metadata = pd.read_excel(mpath, dtype=dict(Patient=str))
        self.metadata = metadata
        patients = metadata.Patient.to_list()
        fc_dataset = {patient_id: {} for patient_id in patients}
        # Build frame
        for path in dpath.iterdir():
            sample = pd.read_parquet(path)
            patient = path.name.split("_")[0]
            tube = path.name.split("_")[1].split(" ")[-1]
            fc_dataset[patient][tube] = sample

        self.fc_dataset = fc_dataset

    def _validate_n_tubes(self):
        """
        Check number of tubes for each patient and
        remove patients who do not have tube A
        """
        patients = self.metadata.Patient.to_list()
        for patient in patients:
            patient_keys = self.fc_dataset[patient].keys()
            if self.tube not in patient_keys:
                print(f"{patient} does not have tube {self.tube}, removing")
                self.fc_dataset.pop(patient)
                drop_idx = self.metadata[self.metadata.Patient == patient].index
                self.metadata.drop(index=drop_idx, inplace=True)


    def get_df(self, index):
        row = self.metadata.iloc[index, :]
        patient_samples = self.fc_dataset[row.Patient]
        sample = patient_samples[self.tube]

        # Handle subsampling directly on the DataFrame
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
            ["TIME", " ", "", "FS PEAK LIN", "FS-H"], axis=1, errors="ignore"
        )
        events.rename({"CD38PB":"CD38 PB", "CD7 PerCP C5.5": "CD7 PC5.5"}, axis=1, inplace=True)
        if (len(events.columns) != 12):
            raise ValueError(f"{row.Patient} tube {self.tube} has {len(events.columns)} columns")
        try:
            events = events.loc[:, pns_dict_camilla[self.tube]] # Make sure we always have the correct order
        except KeyError as err:
            print(events)
            raise err

        label = self.get_label(row)
        return events, label, row


def match_labels(pns_labels, tube):
    for label in pns_dict[tube]:
        if label not in pns_labels:
            return False
    return True

class GeilTestSet(GEILTube):
    """
    Dataset for GEIL tubes that are used as a test set
    """
    def _build_fc_dataset(self, dpath, mpath, n_cells):
        metadata = pd.read_excel(mpath, dtype={"Identifiant patient":str})
        metadata.rename(columns={"Identifiant patient":"Patient"}, inplace=True)
        metadata["FLT3"] = np.where(metadata["Mutation FLT3-ITD"] == "Oui", "ITD", "WT")
        metadata["NPM"] = np.where(
            metadata["Mutation de NPM1 (Biologie moléculaire)"] == "Oui", "NPM1 muté", "neg"
        )

        self.metadata = metadata
        patients = set(metadata.Patient.to_list())
        fc_dataset = {patient_id: {} for patient_id in patients}
        
        # We lose some subjects in parquet because of the assumptions
        # made by the preproc R script
        def load_sample(sample_path):
            """Load a single sample file and return (patient_id, tube, sample) if patient is in patients."""
            sample_format = sample_path.name.split(".")[-1]
            
            if sample_format == "csv":
                sample = pd.read_csv(sample_path, index_col=0)
            elif sample_format == "parquet":
                sample = pd.read_parquet(sample_path)
            elif sample_format == "fcs":
                import flowkit as fk
                fk_sample = fk.Sample(
                    str(sample_path),
                    subsample=False,
                    null_channel_list=["FS PEAK", "FS-H", "FS PEAK LIN", ""]
                )
                if fk_sample.pns_labels[2] == "-FITC":
                    print(sample_path.name, " has truncated pns names, skipping")
                    return None
                
                sample = fk_sample.as_dataframe(
                    source="raw",
                    subsample=False,
                    col_multi_index=False,
                    col_names=fk_sample.pns_labels
                )
            else:
                raise ValueError(f"Unrecognized sample format {sample_format}")
            
            # Apply -A convention to column names for all formats
            sample.columns = [x.replace(" INT LIN", "-A") for x in sample.columns]

            # Extract patient information from file name
            tags = sample_path.name.split("_")
            patient_id = tags[0]
            tube = tags[1][4]
            assert tube in {"A", "B", "C"}, f"Unrecognized tube {tube}"
            
            if not match_labels(sample.columns, tube):
                #if tube == "B":
                #    #if patient_id in {"20130030", "20130038", "20130055"}:
                #    breakpoint()
                print(sample_path.name, "Tube does not have required pns labels, skipping")
                return None
                
            return patient_id, tube, sample
        
        # Collect all sample paths
        sample_paths = list(dpath.iterdir())
        
        # Load samples in parallel
        results = joblib.Parallel(n_jobs=1, verbose=1)(
            joblib.delayed(load_sample)(sample_path) for sample_path in sample_paths
        )
        
        # Build fc_dataset from results
        for result in results:
            if result is not None:
                patient_id, tube, sample = result
                if patient_id in patients:
                    fc_dataset[patient_id][tube] = sample
                else:
                    breakpoint()
        
        print(f"Loaded {sum(len(tubes) for tubes in fc_dataset.values())} fc samples.")
        
        self.fc_dataset = fc_dataset
