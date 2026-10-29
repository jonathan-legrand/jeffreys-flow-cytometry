import numpy as np

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import validate_data
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal


class FisherKernelGenerator:
    """
    Generate a gradient embedding from a GMM.
    Ref : Florent Perronnin, Jorge Sánchez, and Thomas Mensink,
    ‘Improving the Fisher Kernel for Large-Scale Image Classification’
    """
    def __init__(self, gmm, l2_normalize=True):
        self.w, self.mu, self.covariances = gmm.weights_, gmm.means_, gmm.covariances_
        self.sigma = np.sqrt(self.covariances)
        self.n_components = gmm.n_components
        self.l2_normalize = l2_normalize


    def _precompute_gamma(self, X):
        """Iterating on the number of clusters is okay,
        iterating on cells is computationally
        prohibitive, so we vectorize operations which 
        happen along cells.

        Args:
            X (np.ndarray): shape (1, T, D)
        """
        gamma = []
        scalings = []
        for i in range(self.n_components):
            rv = multivariate_normal(mean=self.mu[i], cov=self.covariances[i])
            scaling_i = self.w[i] * rv.pdf(X)
            scalings.append(scaling_i)
        
        scalings = np.stack(scalings, axis=0).sum(axis=0)

        for i in range(self.n_components):
            rv = multivariate_normal(mean=self.mu[i], cov=self.covariances[i])
            u_i = rv.pdf(X)
            gamma_i = (self.w[i] * u_i) / scalings
            gamma.append(gamma_i)
            
        self.gamma_ = gamma
        return gamma
        

    def G(self, X):
        """
        Returns the concatenated array of features from flow cytometry sample

        Args:
            X (np.ndarray): Features for one sample,
            of shape (1, T, D)

        Returns:
            G (np.ndarray): Feature vector from Fisher gradients
            of shape (2 * K * D,)
        """

        X = X.copy().squeeze()
        self._precompute_gamma(X)
        gradients = []
        for i in range(self.n_components):
            # shape (T,). Reads as : how much a given cell 
            # belongs to gausssian i
            attribution = self.gamma_[i]

            T = len(X)
        
            # Compute gradient of Fisher score wrt mean
            factor = 1 / (T * np.sqrt(self.w[i]))
            X_term = (X - self.mu[i]) / self.sigma[i]
            cell_sum = attribution.T @ X_term
            G_mu = factor * cell_sum

            # Compute gradient of Fisher score wrt std
            factor = 1 / (T * np.sqrt(2 * self.w[i]))
            X_term = np.square((X - self.mu[i])) / self.covariances[i]
            cell_sum = attribution.T @ (X_term - 1)
            G_sigma = factor * cell_sum

            G_i = np.concatenate((G_mu, G_sigma))
            gradients.append(G_i)
        return np.concatenate(gradients)



class FisherEmbedding(TransformerMixin, BaseEstimator):
    def __init__(self, n_components=5, random_state=1234):
        self.n_components = n_components
        self.random_state = random_state
    
    def fit(self, X, y=None):
        """Fit the gmm and pass it to Fisher Kernel Generator

        Args:
            X (np.ndarray): (n_samples, n_cells, n_features)
            y (np.ndarray, optional): Just for passing through the pipeline. Defaults to None.

        Returns:
            self: object
            Returns the instance itself
        """

        # Fit GMMs on shape (n_cells * n_samples, n_features)
        self.gmm_ = GaussianMixture(
                n_components=self.n_components,
                covariance_type="diag",
                random_state=self.random_state
        )

        X_flat = X.copy().reshape(
            (-1, X.shape[-1])
        )

        self.gmm_.fit(X_flat)

        self.kernel_gen_ = FisherKernelGenerator(
            self.gmm_,
        )

        return self
    
    def transform(self, X, y=None):
        X = validate_data(
            self,
            X=X,
            reset=False,
            ensure_2d=False,
            allow_nd=True
        ).copy()
        # Validate X shape somehow?
        X_out = []
        for sample in X:
            X_out.append(self.kernel_gen_.G(sample))
        return np.stack(X_out, axis=0)
