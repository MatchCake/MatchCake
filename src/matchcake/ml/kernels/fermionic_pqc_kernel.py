from typing import Optional

import numpy as np
import pennylane as qml
from pennylane.ops.qubit.observables import BasisStateProjector

try:
    from pennylane.templates.broadcast import PATTERN_TO_WIRES
except ImportError:
    # Hotfix for pennylane>0.39.0
    PATTERN_TO_WIRES = {
        "double": lambda wires: [wires.subset([i, i + 1]) for i in range(0, len(wires) - 1, 2)],
        "double_odd": lambda wires: [wires.subset([i, i + 1]) for i in range(1, len(wires) - 1, 2)],
    }

from ...operations import (
    CompHH,
    MAngleEmbedding,
    SptmAngleEmbedding,
    SptmCompHH,
    SptmCompZX,
    fSWAP,
)
from .nif_kernel import NIFKernel


class FermionicPQCKernel(NIFKernel):
    r"""

    Inspired from: https://iopscience.iop.org/article/10.1088/2632-2153/acb0b4/meta#artAbst


    By default, the size of the kernel is computed as

    .. math::
        \text{size} = \max\left(2, \lceil\log_2(\text{n features} + 2)\rceil\right)

    and the depth is computed as

    .. math::
        \text{depth} = \max\left(1, \left(\frac{\text{n features}}{\text{size}} - 1\right)\right)

    """

    available_entangling_mth = {"fswap", "identity", "hadamard"}

    def __init__(self, size: Optional[int] = None, **kwargs):
        super().__init__(size=size, **kwargs)
        self._data_scaling = kwargs.get("data_scaling", np.pi / 2)
        self._parameter_scaling = kwargs.get("parameter_scaling", np.pi / 2)
        self._depth = kwargs.get("depth", None)
        self._rotations = kwargs.get("rotations", "Y,Z")
        self._entangling_mth = kwargs.get("entangling_mth", "fswap")
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

    def _compute_default_size(self):
        _size = max(2, int(np.ceil(np.log2(self.X_.shape[-1] + 2) - 1)))
        if _size % 2 != 0:
            _size += 1
        return _size

    def initialize_parameters(self):
        super().initialize_parameters()
        self._depth = self.kwargs.get("depth", int(max(1, np.ceil(self.X_.shape[-1] / self.size))))
        self.parameters = self.parameters_rng.uniform(0.0, 1.0, size=self.X_.shape[-1])
        if self.qnode.interface == "torch":
            import torch

            self.parameters = torch.from_numpy(self.parameters).float().requires_grad_(True)

    def ansatz(self, x):
        wires_double = PATTERN_TO_WIRES["double"](self.wires)
        wires_double_odd = PATTERN_TO_WIRES["double_odd"](self.wires)
        wires_patterns = [wires_double, wires_double_odd]
        for layer in range(self.depth):
            sub_x = x[..., layer * self.size : (layer + 1) * self.size]
            SptmAngleEmbedding(sub_x, wires=self.wires, rotations=self.rotations)
            wires_list = wires_patterns[layer % len(wires_patterns)]
            for wires in wires_list:
                if self._entangling_mth == "fswap":
                    SptmCompZX(wires=wires)
                elif self._entangling_mth == "hadamard":
                    SptmCompHH(wires=wires)
                elif self._entangling_mth == "identity":
                    pass
                else:
                    raise ValueError(f"Unknown entangling method: {self._entangling_mth}")
        return

    def circuit(self, x0, x1):
        theta_x0 = self._parameter_scaling * self.parameters + self.data_scaling * x0
        theta_x1 = self._parameter_scaling * self.parameters + self.data_scaling * x1
        self.ansatz(theta_x0)
        qml.adjoint(self.ansatz)(theta_x1)
        projector: BasisStateProjector = qml.Projector(np.zeros(self.size), wires=self.wires)
        return qml.expval(projector)


class StateVectorFermionicPQCKernel(FermionicPQCKernel):
    def __init__(self, size: Optional[int] = None, **kwargs):
        super().__init__(size=size, **kwargs)
        self._device_name = kwargs.get("device", "default.qubit")
        self._device_kwargs = kwargs.get("device_kwargs", {})

    def pre_initialize(self):
        self._device = qml.device(self._device_name, wires=self.size, **self._device_kwargs)
        self._qnode = qml.QNode(self.circuit, self._device, **self.qnode_kwargs)
        if self.simpify_qnode:
            self._qnode = qml.simplify(self.qnode)

    def ansatz(self, x):
        wires_double = PATTERN_TO_WIRES["double"](self.wires)
        wires_double_odd = PATTERN_TO_WIRES["double_odd"](self.wires)
        wires_patterns = [wires_double, wires_double_odd]
        for layer in range(self.depth):
            sub_x = x[..., layer * self.size : (layer + 1) * self.size]
            MAngleEmbedding(sub_x, wires=self.wires, rotations=self.rotations)
            wires_list = wires_patterns[layer % len(wires_patterns)]
            for wires in wires_list:
                if self._entangling_mth == "fswap":
                    fSWAP(wires=wires)
                elif self._entangling_mth == "hadamard":
                    CompHH(wires=wires)
                elif self._entangling_mth == "identity":
                    pass
                else:
                    raise ValueError(f"Unknown entangling method: {self._entangling_mth}")
        return
