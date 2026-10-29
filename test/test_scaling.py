"""
Test sklearn's like Z-scoring functions.
"""
import numpy as np
import pytest
from source.scaling import CytoScaler

@pytest.fixture
def synthetic_data():
    # shape: (n_patients, n_cells, n_markers)
    np.random.seed(1234)
    X = np.random.randn(200, 1000, 12) * 2 + 5  # mean=5, std=2
    return X


def test_cytoscaler_transform(synthetic_data):
    scaler = CytoScaler(include_tabular=False)
    # Have mock train and mock test!
    scaler.fit(synthetic_data)
    X_scaled = scaler.transform(synthetic_data)
    
    mean = X_scaled.mean(axis=(0,1))
    std = X_scaled.std(axis=(0,1))
    np.testing.assert_allclose(mean, np.zeros_like(mean), atol=1e-7)
    np.testing.assert_allclose(std, np.ones_like(std), atol=1e-7)

def test_cytoscaler_inverse_transform_roundtrip(synthetic_data):
    scaler = CytoScaler()
    scaler.fit(synthetic_data)
    X_scaled = scaler.transform(synthetic_data)
    X_recovered = scaler.inverse_transform(X_scaled)
    np.testing.assert_allclose(X_recovered, synthetic_data, atol=1e-7)
