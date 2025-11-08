import numpy as np
from scipy.linalg import svd
from sklearn.kernel_approximation import Nystroem as sk_Nystroem
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

from ...utils.torch_utils import to_numpy
from .kernel import Kernel


class Nystroem(sk_Nystroem):
    def __init__(
        self,
        kernel: Kernel,
        *,
        n_components=100,
        random_state=None,
    ):
        super().__init__(
            kernel=kernel,
            n_components=n_components,
            random_state=random_state,
        )

    def fit(self, X, y=None):
        """Fit estimator to data.

        Samples a subset of training points, computes kernel
        on these and computes normalization matrix.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like, shape (n_samples,) or (n_samples, n_outputs), \
                default=None
            Target values (None for unsupervised transformations).

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self.kernel.fit(X, y)
        rnd = check_random_state(self.random_state)
        n_samples = X.shape[0]

        # get basis vectors
        if self.n_components > n_samples:
            # XXX should we just bail?
            n_components = n_samples
        else:
            n_components = self.n_components
        n_components = min(n_samples, n_components)
        inds = rnd.permutation(n_samples)
        basis_inds = inds[:n_components]
        basis = X[basis_inds]
        basis_kernel = self.kernel(basis)
        basis_kernel = to_numpy(basis_kernel)

        # sqrt of kernel matrix on basis vectors
        U, S, V = svd(basis_kernel)
        S = np.maximum(S, 1e-12)
        self.normalization_ = np.dot(U / np.sqrt(S), V)
        self.components_ = basis
        self.component_indices_ = basis_inds
        self._n_features_out = n_components
        return self

    def transform(self, X):
        """Apply feature map to X.

        Computes an approximate feature map using the kernel
        between some training points and X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        check_is_fitted(self)
        embedded = self.kernel(X, self.components_)
        embedded = to_numpy(embedded)
        return np.dot(embedded, self.normalization_.T)

    def freeze(self):
        self.kernel.freeze()
        return self
