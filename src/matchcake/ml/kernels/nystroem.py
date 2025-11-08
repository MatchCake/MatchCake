import numpy as np
from scipy.linalg import svd
from sklearn.kernel_approximation import Nystroem as sk_Nystroem
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

from ...utils.torch_utils import to_numpy
from .kernel import Kernel


class Nystroem(sk_Nystroem):
    """
    Encapsulation of the Nystroem method for kernel approximation.

    This class provides an implementation of the Nystroem method, which
    is used to approximate a kernel map for a given dataset. It is designed
    to reduce computation cost while maintaining a good approximation of the
    original kernel matrix. It involves sampling a subset of the data and
    computing the kernel function on these sampled points to derive the
    approximation.

    :ivar n_components: Number of components (basis functions) used to
        approximate the kernel map. If greater than the number of samples,
        it will be automatically set to the number of samples in the data.
    :type n_components: int
    :ivar kernel: Kernel function to be used for computing the kernel similarity.
    :type kernel: Kernel
    :ivar random_state: Controls the randomness of the sampling of the subset
        of training data. Can be an integer, RandomState instance, or None.
    :type random_state: int or RandomState or None
    """

    def __init__(
        self,
        kernel: Kernel,
        *,
        n_components=100,
        random_state=None,
    ):
        """
        Initializes an instance of the class with specified kernel, number of components,
        and random state. This class is used for applications requiring these parameters
        to process data or perform computations efficiently. The kernel defines the
        transformation applied, `n_components` specifies the dimensionality, and
        `random_state` ensures reproducibility.

        :param kernel: An instance of a Kernel defining the transformation method to be applied.
        :param n_components: Number of components for dimensionality reduction or computation.
            Defaults to 100.
        :param random_state: Seed or random state for ensuring reproducible results. Defaults to None.
        """
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
            n_components = n_samples  # pragma: no cover
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
        """
        Freezes the kernel to retain its current state and settings.

        This method ensures that the kernel does not alter its configuration
        or parameters anymore, providing stability for subsequent operations.

        :return: The current instance of the class.
        :rtype: Same as the class of the current instance.
        """
        self.kernel.freeze()
        return self
