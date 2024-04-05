from typing import Optional

import numpy as np
import pennylane as qml
from pennylane import AngleEmbedding
from pennylane import numpy as pnp
from pennylane.ops.qubit.observables import BasisStateProjector
from pennylane.templates.broadcast import PATTERN_TO_WIRES

from matchcake.ml.kernels.ml_kernel import (
    NIFKernel,
)


class MPennylaneQuantumKernel(NIFKernel):
    def __init__(self, size: Optional[int] = None, **kwargs):
        kwargs["use_cuda"] = False
        super().__init__(size=size, **kwargs)
        self._device_name = kwargs.get("device", "default.qubit")
        self._device_kwargs = kwargs.get("device_kwargs", {})

    def pre_initialize(self):
        self._device = qml.device(self._device_name, wires=self.size, **self._device_kwargs)
        self._qnode = qml.QNode(self.circuit, self._device, **self.qnode_kwargs)


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
    available_entangling_mth = {"cnot", "identity"}

    def __init__(self, size: Optional[int] = None, **kwargs):
        super().__init__(size=size, **kwargs)
        self._device_name = kwargs.get("device", "default.qubit")
        self._data_scaling = kwargs.get("data_scaling", np.pi / 2)
        self._parameter_scaling = kwargs.get("parameter_scaling", np.pi / 2)
        self._depth = kwargs.get("depth", None)
        self._rotations = kwargs.get("rotations", "Y,Z").split(",")
        self._entangling_mth = kwargs.get("entangling_mth", "cnot")
        if self._entangling_mth not in self.available_entangling_mth:
            raise ValueError(f"Unknown entangling method: {self._entangling_mth}.")

    @property
    def depth(self):
        return self._depth

    @property
    def data_scaling(self):
        return self._data_scaling

    @property
    def rotations(self):
        return self._rotations

    @property
    def entangling_mth(self):
        return getattr(self, "_entangling_mth", "cnot")

    def _compute_default_size(self):
        return max(2, int(np.ceil(np.log2(self.X_.shape[-1] + 2) - 1)))

    def initialize_parameters(self):
        self._depth = self.kwargs.get("depth", max(1, (self.X_.shape[-1] // self.size) - 1))
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
                if self.entangling_mth == "cnot":
                    qml.CNOT(wires=wires)
                elif self.entangling_mth == "identity":
                    pass
                else:
                    raise ValueError(f"Unknown entangling method: {self.entangling_mth}.")

    def circuit(self, x0, x1):
        theta_x0 = self._parameter_scaling * self.parameters + self.data_scaling * x0
        theta_x1 = self._parameter_scaling * self.parameters + self.data_scaling * x1

        self.ansatz(theta_x0)
        qml.adjoint(self.ansatz)(theta_x1)

        projector: BasisStateProjector = qml.Projector(np.zeros(self.size), wires=self.wires)
        return qml.expval(projector)


class IdentityPQCKernel(PQCKernel):
    # TODO: Use MPS simulator
    def __init__(self, size: Optional[int] = None, **kwargs):
        kwargs["entangling_mth"] = "identity"
        super().__init__(size=size, **kwargs)


class LightningPQCKernel(PQCKernel):
    def __init__(self, size: Optional[int] = None, **kwargs):
        super().__init__(size=size, **kwargs)
        self._device_name = kwargs.get("device", "lightning.qubit")
        self._device_kwargs = kwargs.get("device_kwargs", {"batch_obs": True})
