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
from pennylane import broadcast
try:
    import msim
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    import msim
    
from msim.ml.ml_kernel import MLKernel, NIFKernel, MPQCKernel


class ClassicalKernel(MLKernel):
    def __init__(
            self,
            size: Optional[int] = None,
            **kwargs
    ):
        super().__init__(size=size, **kwargs)
        self._metric = kwargs.get("metric", "linear")
        
    def single_distance(self, x0, x1):
        return pairwise_kernels(x0, x1, metric=self._metric)
    
    def batch_distance(self, x0, x1):
        return pairwise_kernels(x0, x1, metric=self._metric)

    def pairwise_distances(self, x0, x1, **kwargs):
        return pairwise_kernels(x0, x1, metric=self._metric)


class MPennylaneQuantumKernel(NIFKernel):
    def __init__(self, size: Optional[int] = None, **kwargs):
        super().__init__(size=size, **kwargs)
        self._device_name = kwargs.get("device", "default.qubit")

    def fit(self, X, y=None):
        MLKernel.fit(self, X, y)
        self._device = qml.device(self._device_name, wires=self.size)
        self.qnode = qml.QNode(self.circuit, self._device, **self.kwargs.get("qnode_kwargs", {}))
        return self


class CPennylaneQuantumKernel(MPennylaneQuantumKernel):
    def __init__(self, size: Optional[int] = None, **kwargs):
        super().__init__(size=size, **kwargs)
        self._device_name = kwargs.get("device", "default.qubit")
        self._embedding_rotation = kwargs.get("embedding_rotation", "X")
    
    def circuit(self, x0, x1):
        AngleEmbedding(x0, wires=range(self.size), rotation=self._embedding_rotation)
        # broadcast(unitary=qml.RX, pattern="pyramid", wires=range(self.size), parameters=x0)
        qml.adjoint(AngleEmbedding)(x1, wires=range(self.size))
        # qml.adjoint(broadcast)(unitary=qml.RX, pattern="pyramid", wires=range(self.size), parameters=x1)
        projector: BasisStateProjector = qml.Projector(np.zeros(self.size), wires=range(self.size))
        return qml.expval(projector)
