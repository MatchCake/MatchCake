from typing import Optional, Type, Union

import numpy as np
import pennylane as qml
from sklearn import svm
from tqdm import tqdm

from ..utils import torch_utils
from .kernels.ml_kernel import MLKernel
from .std_estimator import StdEstimator


class SimpleSVC(StdEstimator):
    def __init__(
        self,
        kernel_cls: Union[Type[MLKernel]],
        kernel_kwargs: Optional[dict] = None,
        cache_size: int = 1024,
        random_state: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.kernel_cls = kernel_cls
        self.kernel_kwargs = kernel_kwargs or {}
        self.cache_size = cache_size
        self.random_state = random_state
        self.estimator_ = None
        self._train_gram_matrix = None
        self.kernel: Optional[MLKernel] = None
        self.memory = []

    @property
    def train_gram_matrix(self):
        return self._train_gram_matrix

    @train_gram_matrix.setter
    def train_gram_matrix(self, value):
        self._train_gram_matrix = torch_utils.to_numpy(value)

    def fit(self, X, y=None, train_gram_matrix=None, **kwargs):
        """
        Fit the model to the input data of shape (n_samples, n_features). This process involves
        building the kernel object, computing the gram matrix, and fitting the SVM model. If the
        train_gram_matrix is provided, the kernel object will not be built and the gram matrix will
        be used directly.

        :param X: The input data of shape (n_samples, n_features).
        :param y: The target labels of shape (n_samples,).
        :param train_gram_matrix: The precomputed gram matrix of shape (n_samples, n_samples). This input is
            optional and is used to avoid recomputing the gram matrix.
        :param kwargs: Additional keyword arguments to be passed to the kernel object and the compute_gram_matrix method.
        :return: The fitted model.
        """
        super().fit(X, y, **kwargs)
        if train_gram_matrix is None:
            self.kernel = self.kernel_cls(**self.kernel_kwargs).fit(X, y, **kwargs)
            self.train_gram_matrix = self.kernel.compute_gram_matrix(X, **kwargs)
        else:
            self.train_gram_matrix = train_gram_matrix
        self.estimator_ = svm.SVC(
            kernel="precomputed",
            random_state=self.random_state,
            cache_size=self.cache_size,
        )
        self.estimator_.fit(self.train_gram_matrix, y)
        return self

    def get_gram_matrix_from_memory(self, X, **kwargs):
        if qml.math.shape(X) == qml.math.shape(self.X_) and qml.math.allclose(self.X_, X):
            return self.train_gram_matrix
        if self.memory is None:
            return None
        for i, (x, gram_matrix) in enumerate(self.memory):
            if qml.math.shape(X) == qml.math.shape(x) and qml.math.allclose(x, X):
                return gram_matrix
        return None

    def get_is_in_memory(self, X):
        return self.get_gram_matrix_from_memory(X) is not None

    def push_to_memory(self, X, gram_matrix):
        if getattr(self, "memory", None) is None:
            self.memory = []
        if not self.get_is_in_memory(X):
            self.memory.append((X, gram_matrix))
        return self

    def predict(self, X, **kwargs):
        """
        Predict the labels of the input data of shape (n_samples, n_features).

        :param X: The input data of shape (n_samples, n_features).
        :type X: np.ndarray or array-like
        :param kwargs: Additional keyword arguments.

        :keyword cache: If True, the computed gram matrices will be stored in memory. Default is False.

        :return: The predicted labels of the input data.
        :rtype: np.ndarray of shape (n_samples,)
        """
        self.check_is_fitted()
        gram_matrix = self.get_gram_matrix_from_memory(X, **kwargs)
        if gram_matrix is None:
            gram_matrix = self.kernel.pairwise_distances(X, self.X_, **kwargs)
        if kwargs.get("cache", False):
            self.push_to_memory(X, gram_matrix)
        return self.estimator_.predict(gram_matrix)

    def score(self, X, y, **kwargs):
        """
        Get the accuracy of the model on the input data.

        :param X: The input data of shape (n_samples, n_features).
        :param y: The target labels of shape (n_samples,).
        :param kwargs: Additional keyword arguments to be passed to the predict method.
        :return: The accuracy of the model on the input data.
        """
        self.check_is_fitted()
        pred = self.predict(X, **kwargs)
        return np.mean(np.isclose(pred, y).astype(float))


class FixedSizeSVC(StdEstimator):
    def __init__(
        self,
        kernel_cls: Union[Type[MLKernel], str],
        kernel_kwargs: Optional[dict] = None,
        max_gram_size: Optional[int] = np.inf,
        cache_size: int = 1024,
        random_state: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.kernel_cls = kernel_cls
        self.kernel_kwargs = kernel_kwargs or {}
        self.max_gram_size = max_gram_size or np.inf
        self.cache_size = cache_size
        self.random_state = random_state
        self.kernels = None
        self.estimators_ = None
        self.train_gram_matrices = None
        self.memory = []

    @property
    def kernel(self):
        return self.kernel_cls

    @property
    def n_kernels(self):
        return len(self.kernels)

    @property
    def kernel_size(self):
        if self.kernels is None:
            return None
        return getattr(self.kernels[0], "size", None)

    @property
    def kernel_n_ops(self):
        if self.kernels is None:
            return None
        return getattr(self.kernels[0], "n_ops", None)

    @property
    def kernel_n_params(self):
        if self.kernels is None:
            return None
        return getattr(self.kernels[0], "n_params", None)

    @property
    def n_features(self):
        if self.X_ is None:
            return None
        return qml.math.shape(self.X_)[-1]

    def __getattr__(self, item):
        if item in self.__dict__:
            return self.__dict__[item]
        if item.startswith("kernel_"):
            if self.kernels is None:
                return None
            return getattr(self.kernels[0], item[7:])
        raise AttributeError(f"{self.__class__.__name__} has no attribute {item}.")

    def split_data(self, x, y=None):
        x_shape = qml.math.shape(x)
        if x_shape[0] <= self.max_gram_size:
            return [x], [y]
        n_splits = int(np.ceil(x_shape[0] / self.max_gram_size))
        x_splits = np.array_split(x, n_splits)
        if y is not None:
            y_splits = np.array_split(y, n_splits)
        else:
            y_splits = [None for _ in range(n_splits)]
        return x_splits, y_splits

    def get_gram_matrices(self, X, **kwargs):
        p_bar: Optional[tqdm] = kwargs.get("p_bar", None)
        p_bar_postfix_str = p_bar.postfix if p_bar is not None else ""
        x_splits, _ = self.split_data(X)
        gram_matrices = []
        for i, sub_x in enumerate(x_splits):
            if p_bar is not None:
                p_bar.set_postfix_str(f"{p_bar_postfix_str}:[{i + 1}/{len(x_splits)}]")
            gram_matrices.append(self.kernels[i].compute_gram_matrix(sub_x, **kwargs))
        if p_bar is not None:
            p_bar.set_postfix_str(p_bar_postfix_str)
        return gram_matrices

    def get_gram_matrix(self, X):
        gram_matrices = self.get_gram_matrices(X)
        return qml.math.block_diag(gram_matrices)

    def get_pairwise_distances_matrices(self, x0, x1, **kwargs):
        x0_splits, _ = self.split_data(x0)
        x1_splits, _ = self.split_data(x1)
        pairwise_distances = []
        for i, (sub_x0, sub_x1) in enumerate(zip(x0_splits, x1_splits)):
            pairwise_distances.append(self.kernels[i].pairwise_distances(sub_x0, sub_x1, **kwargs))
        return pairwise_distances

    def get_pairwise_distances(self, x0, x1):
        pairwise_distances = self.get_pairwise_distances_matrices(x0, x1)
        return qml.math.block_diag(pairwise_distances)

    def fit(self, X, y=None, **kwargs):
        super().fit(X, y, **kwargs)
        x_splits, y_splits = self.split_data(X, y)
        self.kernels = [
            self.kernel_cls(**self.kernel_kwargs).fit(sub_x, sub_y, **kwargs)
            for i, (sub_x, sub_y) in enumerate(zip(x_splits, y_splits))
        ]
        self.train_gram_matrices = self.get_gram_matrices(X, **kwargs)
        self.estimators_ = [
            svm.SVC(
                kernel="precomputed",
                random_state=self.random_state,
                cache_size=self.cache_size,
            )
            for _ in range(self.n_kernels)
        ]
        for i, (gram_matrix, sub_y) in enumerate(zip(self.train_gram_matrices, y_splits)):
            self.estimators_[i].fit(gram_matrix, sub_y)
        return self

    def get_gram_matrices_from_memory(self, X, **kwargs):
        if qml.math.shape(X) == qml.math.shape(self.X_) and qml.math.allclose(self.X_, X):
            return self.train_gram_matrices
        if getattr(self, "memory", None) is None:
            return None
        for i, (x, gram_matrices) in enumerate(self.memory):
            if qml.math.shape(X) == qml.math.shape(x) and qml.math.allclose(x, X):
                return gram_matrices
        return None

    def get_is_in_memory(self, X):
        return self.get_gram_matrices_from_memory(X) is not None

    def push_to_memory(self, X, gram_matrices):
        if getattr(self, "memory", None) is None:
            self.memory = []
        if not self.get_is_in_memory(X):
            self.memory.append((X, gram_matrices))
        return self

    def predict(self, X, **kwargs):
        """
        Predict the labels of the input data of shape (n_samples, n_features).

        :param X: The input data of shape (n_samples, n_features).
        :type X: np.ndarray or array-like
        :param kwargs: Additional keyword arguments.

        :keyword cache: If True, the computed gram matrices will be stored in memory. Default is False.

        :return: The predicted labels of the input data.
        :rtype: np.ndarray of shape (n_samples,)
        """
        self.check_is_fitted()
        gram_matrices = self.get_gram_matrices_from_memory(X, **kwargs)
        if gram_matrices is None:
            gram_matrices = self.get_pairwise_distances_matrices(X, self.X_, **kwargs)
        if kwargs.get("cache", False):
            self.push_to_memory(X, gram_matrices)
        votes = np.zeros((qml.math.shape(X)[0], len(self.classes_)), dtype=int)
        predictions_stack = []
        for i, gram_matrix in enumerate(gram_matrices):
            predictions = self.estimators_[i].predict(gram_matrix)
            predictions_stack.append(predictions)
        predictions_stack = np.concatenate(predictions_stack, axis=0)
        votes[np.arange(qml.math.shape(X)[0]), predictions_stack] += 1
        return self.classes_[np.argmax(votes, axis=-1)]

    def score(self, X, y, **kwargs):
        self.check_is_fitted()
        pred = self.predict(X, **kwargs)
        return np.mean(np.isclose(pred, y).astype(float))
