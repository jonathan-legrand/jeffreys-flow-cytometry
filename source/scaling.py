import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted

# TODO Pass pandas df directly? How does the standard scaler do?
class CytoScaler(TransformerMixin, BaseEstimator):
    """
    Assume shape (batch_size, n_cells, n_markers)
    """
    def __init__(self, include_tabular=False):
        self.include_tabular = include_tabular

    def fit(self, X, y=None, tabular_data=None):
        
        # We want value per marker => shape (1, 1, n_markers)
        target_shape = (1, 1, X.shape[-1])
        self.mean_ = X.mean(axis=(0, 1)).reshape(target_shape)
        self.scale_ = X.std(axis=(0, 1)).reshape(target_shape)

        if self.include_tabular:
            if tabular_data is None:
                raise ValueError("Expected tabular metadata")
            
            if tabular_data.shape[0] != X.shape[0]:
                raise ValueError(
                    f"""
                    Cytometry and tabular arrrays have inconsistent
                    shapes. Along first dimension, X is of size
                    {X.shape[0]} and tabular data is of shape
                    {tabular_data.shape[0]}
                    """)

            self.mean_tabular_ = tabular_data.mean(axis=0)
            self.scale_tabular_ = tabular_data.std(axis=0)

        elif tabular_data is not None:
            raise ValueError(
                "Scaler was not initialized for tabular data"
            )
        return self

    def transform(self, X, y=None, tabular_data=None):
        X_scaled = X - self.mean_
        X_scaled /= self.scale_
        if self.include_tabular:
            if tabular_data is None:
                raise ValueError("Expected tabular metadata")
            tab_scaled = tabular_data - self.mean_tabular_
            tab_scaled /= self.scale_tabular_
            return X_scaled.squeeze(), tab_scaled.squeeze()
        return X_scaled # TODO I removed the squeeze and I do not know what it can break

    def inverse_transform(self, X):
        check_is_fitted(self)
        X_unscaled = X * self.scale_
        X_unscaled += self.mean_
        if self.include_tabular:
            raise NotImplementedError()
        return X_unscaled
