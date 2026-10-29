"""test functions from the Fisher kernel embedding object
"""
import pytest
import numpy as np
import matplotlib.pyplot as plt # For pdb
from source.models.fisher_kernel import FisherKernelGenerator
from scipy.stats import norm, multivariate_normal, ttest_1samp


rng = np.random.default_rng(seed=1234)
loc_a = -2
loc_b = 2
scale_a = 2
scale_b = 1
n_features = 100

@pytest.fixture
def gen():
    class MockFittedGMM:
        n_components = 2
        weights_ = [0.5, 0.5]
        means_ = np.array([[loc_a, loc_b]] * n_features).T
        covariances_ = np.array([[scale_a ** 2, scale_b ** 2]] * n_features).T
    
    return FisherKernelGenerator(MockFittedGMM(), l2_normalize=False)


def test_gamma(gen):
    """
    Generate artificial clusters, 
    create a fake GMM accordingly and check
    if gamma function correctly attributes 
    points generated from the same distribution.
    Since we know the true values of mu and sigma,
    we can compute the correct soft attribution value.
    """

    mean_vec_a = np.repeat(loc_a, n_features)
    mean_vec_b = np.repeat(loc_b, n_features)
    rv_a = multivariate_normal(mean=mean_vec_a, cov=scale_a ** 2)
    rv_b = multivariate_normal(mean=mean_vec_b, cov=scale_b ** 2)

    point_cluster_a = rv_a.rvs()
    point_cluster_b = rv_b.rvs()

    # Check attribution of point a
    u_a = rv_a.pdf(point_cluster_a)
    u_b = rv_b.pdf(point_cluster_a)
    true_gamma = np.array([u_a / (u_a + u_b), u_b / (u_a + u_b)])
    computed_gamma = gen._precompute_gamma(point_cluster_a)
    assert np.all(np.isclose(true_gamma, computed_gamma)), "Wrong attribution for point a"

    # Check attribution of point b
    u_a = rv_a.pdf(point_cluster_b)
    u_b = rv_b.pdf(point_cluster_b)
    true_gamma = np.array([u_a / (u_a + u_b), u_b / (u_a + u_b)])
    computed_gamma = gen._precompute_gamma(point_cluster_b)
    assert np.all(np.isclose(true_gamma, computed_gamma)), "Wrong attribution for point b"


def test_gamma_precomputation(gen):
    """
    We create an imbalanced dataset from two random variables with different means
    and check whether the attributions match with initial proportions.

    """

    n_X_1, n_X_2 = 10, int(1e3)
    g_1_ratio = (n_X_1) / (n_X_1 + n_X_2)

    X_1 = rng.normal(loc=loc_a, scale=scale_a, size=(1, n_X_1, n_features))
    X_2 = rng.normal(loc=loc_b, scale=scale_b, size=(1, n_X_2, n_features))
    X = np.concatenate((X_1, X_2), axis=1)

    gamma = gen._precompute_gamma(X)
    K = gen.n_components


    assert np.all(np.isclose(sum(gamma), 1.0)), "Attributions do not sum to one"
    assert np.isclose(
        np.mean(gamma[0]),
        g_1_ratio,
        atol=1e-2
    ), "Attribution proportions are inconsistent"


def test_mu_gradient(gen):
    """
    Test the fisher score embedding itself.
    The Fisher score measures the direction in which
    the parameters should shift to better generate
    the observed data.
    """
    
    delta = 0.1
    X = rng.normal(loc=loc_a - delta, scale=scale_a, size=(1, 5000, n_features))

    embeddings = gen.G(X)

    # G_mu contain negative elements because this population
    # mostly belongs to cluster a but was generated
    # with a loc of loc_a - delta.
    # We know that G_mu_a in this case has
    # an expected value of -delta / (sigma_a * sqrt(w_a))
    G_mu_a = embeddings[:n_features]
    expected_value = -delta / (scale_a * np.sqrt(gen.w[0]))
    result = ttest_1samp(G_mu_a, expected_value, alternative="two-sided")
    assert result.pvalue > 0.1, "Gradient scores are not consistent with the given sample and GMM parameters"

    # The generated population does not belong to cluster b, so E(G_mu_b) == 0
    G_mu_b = embeddings[2*n_features:3 * n_features]
    assert np.all(np.isclose(G_mu_b, 0)), "Gradient scores are not consistent with the given sample and GMM parameters"
    

def test_sigma_gradient(gen): 
    delta = 0.1
    X = rng.normal(loc=loc_a, scale=scale_a - delta, size=(1, 5000, n_features))

    embeddings = gen.G(X)
    G_sigma_a = embeddings[n_features:2*n_features]
    
    # When X was generated with mean_a, we know that all gamma(a) == 1
    # and expectation of G_sigma_a is the following
    factor = 1 / (np.sqrt(2 * gen.w[0]))
    inner_expectation = (scale_a - delta) ** 2
    expected_value = factor * (inner_expectation / (scale_a ** 2) - 1)
    
    result = ttest_1samp(G_sigma_a, expected_value, alternative="two-sided")
    assert result.pvalue > 0.1, "Gradient scores are not consistent with the given sample and GMM parameters"

    # The generated population does not belong to cluster b, so E(G_sigma_b) == 0
    G_sigma_b = embeddings[3*n_features:-1]
    assert np.all(np.isclose(G_sigma_b, 0)), "Gradient scores are not consistent with the given sample and GMM parameters"


