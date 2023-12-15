from typing import Optional
import sys
import os
import numpy as np
from pennylane import AngleEmbedding
from pennylane.ops.qubit.observables import BasisStateProjector
from sklearn.utils.estimator_checks import check_estimator
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics.pairwise import pairwise_kernels
import pennylane as qml
import pythonbasictools as pbt

from ..devices.nif_device import NonInteractingFermionicDevice
from ..operations import MAngleEmbedding


class MLKernel(BaseEstimator):
    def __init__(
            self,
            size: Optional[int] = None,
            **kwargs
    ):
        self._size = size
        self.nb_workers = kwargs.get("nb_workers", 0)
        self.kwargs = kwargs
        
        self.X_, self.y_, self.classes_ = None, None, None
    
    def fit(self, X, y=None):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y

        if self._size is None:
            self._size = X.shape[-1]
        return self
    
    def transform(self, x):
        check_is_fitted(self)
        x = check_array(x)
        return x
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
    
    def single_distance(self, x0, x1):
        raise NotImplementedError(f"This method is not implemented for {self.__class__.__name__}.")
    
    def batch_distance(self, x0, x1):
        raise NotImplementedError(f"This method is not implemented for {self.__class__.__name__}.")
        
    def pairwise_distances(self, x0, x1, **kwargs):
        x0 = check_array(x0)
        x1 = check_array(x1)
        check_is_fitted(self)
        _list_results = pbt.apply_func_multiprocess(
            func=self.batch_distance,
            iterable_of_args=[(x0, b) for b in x1],
            iterable_of_kwargs=[kwargs for _ in range(len(x1))],
            nb_workers=self.nb_workers,
            verbose=False,
        )
        # _result = np.asarray(_list_results).reshape((len(x0), len(x1)))
        _result = np.stack(_list_results, axis=-1)
        return _result


class NIFKernel(MLKernel):
    UNPICKLABLE_ATTRIBUTES = ['_device', ]
    
    def __init__(
            self,
            size: Optional[int] = None,
            **kwargs
    ):
        super().__init__(size=size, **kwargs)
        self.qnode = None
        self._device = None
        
    @property
    def size(self):
        return self._size
        
    def __getstate__(self):
        state = {
            k: v
            for k, v in self.__dict__.items()
            if k not in self.UNPICKLABLE_ATTRIBUTES
        }
        return state
    
    def fit(self, X, y=None):
        super().fit(X, y)
        self._device = NonInteractingFermionicDevice(wires=self.size)
        self.qnode = qml.QNode(self.circuit, self._device, **self.kwargs.get("qnode_kwargs", {}))
        return self
    
    def circuit(self, x0, x1):
        MAngleEmbedding(x0, wires=range(self.size))
        qml.adjoint(MAngleEmbedding)(x1, wires=range(self.size))
        projector: BasisStateProjector = qml.Projector(np.zeros(self.size), wires=range(self.size))
        return qml.expval(projector)
    
    def single_distance(self, x0, x1):
        return self.qnode(x0, x1)
    
    def batch_distance(self, x0, x1):
        return self.qnode(x0, x1)
