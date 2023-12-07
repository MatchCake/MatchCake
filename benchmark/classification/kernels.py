from typing import Optional

import numpy as np
from pennylane import AngleEmbedding
from sklearn.utils.estimator_checks import check_estimator
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics.pairwise import pairwise_kernels
import pennylane as qml
import pythonbasictools as pbt


class ClassicalKernel(BaseEstimator):
    def __init__(
            self,
            *,
            embedding_dim: Optional[int] = None,
            seed: Optional[int] = 0,
            encoder_matrix: Optional[np.ndarray] = None,
            metric: Optional[str] = "linear",
    ):
        self._embedding_dim = embedding_dim
        self._seed = seed
        self._random_state = np.random.RandomState(seed=self._seed)
        self._encoder_matrix = encoder_matrix
        self._metric = metric

    def fit(self, X, y=None):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y

        if self._encoder_matrix is None:
            if self._embedding_dim is None:
                self._embedding_dim = X.shape[-1]

            if X.shape[-1] == self._embedding_dim:
                self._encoder_matrix = np.eye(X.shape[-1])
            else:
                self._encoder_matrix = self._random_state.randn(X.shape[-1], self._embedding_dim)
        self._embedding_dim = self._encoder_matrix.shape[-1]
        return self

    def transform(self, x):
        check_is_fitted(self)
        x = check_array(x)
        result = np.dot(x, self._encoder_matrix)
        return result

    def inverse_transform(self, x):
        check_is_fitted(self)
        x = check_array(x)
        return np.dot(x, self._encoder_matrix.T)  # Note: not sure about that

    def fit_transform(self, x, y=None):
        self.fit(x, y)
        return self.transform(x)

    def kernel(self, x0, x1):
        x0_t = self.transform(x0)
        x1_t = self.transform(x1)
        return pairwise_kernels(x0_t, x1_t, metric=self._metric)


class PennylaneQuantumKernel(BaseEstimator):
    UNPICKLABLE_ATTRIBUTES = ['_device', ]

    def __init__(
            self,
            *,
            embedding_dim: Optional[int] = None,
            seed: Optional[int] = 0,
            encoder_matrix: Optional[np.ndarray] = None,
            interface: Optional[str] = "auto",
            device: Optional[str] = "lightning.qubit",
            shots: int = 1,
            nb_workers: int = 0,
    ):
        self._embedding_dim = embedding_dim
        self._seed = seed
        self._random_state = np.random.RandomState(seed=self._seed)
        self._encoder_matrix = encoder_matrix
        self._interface = interface
        self._device = device
        self._shots = shots
        self._nb_workers = nb_workers

        self._dev_kernel = None
        self._qnode = None

    def __getstate__(self):
        state = {
            k: v
            for k, v in self.__dict__.items()
            if k not in self.UNPICKLABLE_ATTRIBUTES
        }
        return state

    def fit(self, X, y=None):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y

        if self._encoder_matrix is None:
            if self._embedding_dim is None:
                self._embedding_dim = X.shape[-1]
            if X.shape[-1] == self._embedding_dim:
                self._encoder_matrix = np.eye(X.shape[-1])
            else:
                self._encoder_matrix = self._random_state.randn(X.shape[-1], self._embedding_dim)

        self._embedding_dim = self._encoder_matrix.shape[-1]
        self._dev_kernel = qml.device(self._device, wires=self._embedding_dim, shots=self._shots)
        self._qnode = qml.QNode(self.circuit, self._dev_kernel, interface=self._interface)

        return self

    def transform(self, x):
        check_is_fitted(self)
        x = np.asarray(x)
        return np.dot(x, self._encoder_matrix)

    def inverse_transform(self, x):
        check_is_fitted(self)
        x = np.asarray(x)
        return np.dot(x, self._encoder_matrix.T)

    def fit_transform(self, x, y=None):
        self.fit(x, y)
        return self.transform(x)

    def circuit(self, x0, x1):
        x0_t = self.transform(x0)
        x1_t = self.transform(x1)

        projector = np.zeros((2 ** self._embedding_dim, 2 ** self._embedding_dim))
        projector[0, 0] = 1

        AngleEmbedding(x0_t, wires=range(self._embedding_dim))
        qml.adjoint(AngleEmbedding)(x1_t, wires=range(self._embedding_dim))
        return qml.expval(qml.Hermitian(projector, wires=range(self._embedding_dim)))

    def kernel(self, x0, x1, **kwargs):
        x0 = check_array(x0)
        x1 = check_array(x1)
        check_is_fitted(self)

        _list_results = pbt.apply_func_multiprocess(
            func=self._qnode,
            iterable_of_args=[
                (a, b)
                for a in x0
                for b in x1
            ],
            iterable_of_kwargs=[
                kwargs
                for _ in range(len(x0) * len(x1))
            ],
            nb_workers=self._nb_workers,
            verbose=False,
        )
        _result = np.asarray(_list_results).reshape((len(x0), len(x1)))
        return _result

