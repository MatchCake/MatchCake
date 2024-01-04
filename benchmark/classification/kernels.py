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
from pennylane.templates.broadcast import wires_pyramid, PATTERN_TO_NUM_PARAMS, PATTERN_TO_WIRES
from pennylane import numpy as pnp
try:
    import msim
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    import msim
    
from msim.ml.ml_kernel import MLKernel, NIFKernel, FermionicPQCKernel, PennylaneFermionicPQCKernel


class ClassicalKernel(MLKernel):
    def __init__(
            self,
            size: Optional[int] = None,
            **kwargs
    ):
        super().__init__(size=size, **kwargs)
        self._metric = kwargs.get("metric", "linear")
        
    def single_distance(self, x0, x1, **kwargs):
        return pairwise_kernels(x0, x1, metric=self._metric)
    
    def batch_distance(self, x0, x1, **kwargs):
        return pairwise_kernels(x0, x1, metric=self._metric)

    def pairwise_distances(self, x0, x1, **kwargs):
        return pairwise_kernels(x0, x1, metric=self._metric)


class MPennylaneQuantumKernel(NIFKernel):
    def __init__(self, size: Optional[int] = None, **kwargs):
        super().__init__(size=size, **kwargs)
        self._device_name = kwargs.get("device", "default.qubit")
        self._device_kwargs = kwargs.get("device_kwargs", {})

    def pre_initialize(self):
        self._device = qml.device(self._device_name, wires=self.size, **self._device_kwargs)
        self.qnode = qml.QNode(self.circuit, self._device, **self.kwargs.get("qnode_kwargs", {}))


class CPennylaneQuantumKernel(MPennylaneQuantumKernel):
    def __init__(self, size: Optional[int] = None, **kwargs):
        super().__init__(size=size, **kwargs)
        # self._device_name = kwargs.get("device", "lightning.qubit")
        self._embedding_rotation = kwargs.get("embedding_rotation", "X")
    
    def circuit(self, x0, x1):
        AngleEmbedding(x0, wires=range(self.size), rotation=self._embedding_rotation)
        # broadcast(unitary=qml.RX, pattern="pyramid", wires=range(self.size), parameters=x0)
        qml.adjoint(AngleEmbedding)(x1, wires=range(self.size))
        # qml.adjoint(broadcast)(unitary=qml.RX, pattern="pyramid", wires=range(self.size), parameters=x1)
        projector: BasisStateProjector = qml.Projector(np.zeros(self.size), wires=range(self.size))
        return qml.expval(projector)


class PQCKernel(MPennylaneQuantumKernel):
    def __init__(self, size: Optional[int] = None, **kwargs):
        super().__init__(size=size, **kwargs)
        self._device_name = kwargs.get("device", "default.qubit")
        self._data_scaling = kwargs.get("data_scaling", np.pi / 2)
        self._parameter_scaling = kwargs.get("parameter_scaling", np.pi / 2)
        self._depth = kwargs.get("depth", None)
        self._rotations = kwargs.get("rotations", "Y,Z").split(",")

    @property
    def depth(self):
        return self._depth

    @property
    def data_scaling(self):
        return self._data_scaling

    @property
    def rotations(self):
        return self._rotations

    def _compute_default_size(self):
        return max(2, int(np.ceil(np.log2(self.X_.shape[-1] + 2) - 1)))

    def initialize_parameters(self):
        self._depth = self.kwargs.get("depth", max(1, (self.X_.shape[-1]//self.size) - 1))
        self.parameters = pnp.random.uniform(0.0, 1.0, size=self.X_.shape[-1])

    def ansatz(self, x):
        wires_double = PATTERN_TO_WIRES["double"](self.wires)
        wires_double_odd = PATTERN_TO_WIRES["double_odd"](self.wires)
        wires_patterns = [wires_double, wires_double_odd]
        for layer in range(self.depth):
            sub_x = x[..., layer * self.size: (layer + 1) * self.size]
            qml.AngleEmbedding(sub_x, wires=self.wires, rotation=self._rotations[0])
            qml.AngleEmbedding(sub_x, wires=self.wires, rotation=self._rotations[1])
            cnot_wires = wires_patterns[layer % len(wires_patterns)]
            for wires in cnot_wires:
                qml.CNOT(wires=wires)

    def circuit(self, x0, x1):
        theta_x0 = self._parameter_scaling * self.parameters + self.data_scaling * x0
        theta_x1 = self._parameter_scaling * self.parameters + self.data_scaling * x1

        self.ansatz(theta_x0)
        qml.adjoint(self.ansatz)(theta_x1)

        projector: BasisStateProjector = qml.Projector(np.zeros(self.size), wires=self.wires)
        return qml.expval(projector)


class LightningPQCKernel(PQCKernel):
    def __init__(self, size: Optional[int] = None, **kwargs):
        super().__init__(size=size, **kwargs)
        self._device_name = kwargs.get("device", "lightning.qubit")
        self._device_kwargs = kwargs.get("device_kwargs", {"batch_obs": True})


class NeighboursFermionicPQCKernel(FermionicPQCKernel):
    def pre_initialize(self):
        self._device = msim.NonInteractingFermionicDevice(wires=self.size, contraction_method="neighbours")
        self.qnode = qml.QNode(self.circuit, self._device, **self.kwargs.get("qnode_kwargs", {}))


class HFermionicPQCKernel(FermionicPQCKernel):
    pass
