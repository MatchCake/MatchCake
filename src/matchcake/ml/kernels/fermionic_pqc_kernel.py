from typing import Optional, Union

import numpy as np
import pennylane as qml
import torch
from numpy._typing import NDArray
from torch.nn import Parameter

from ...utils.torch_utils import to_tensor

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
    DEFAULT_N_QUBITS = 12
    DEFAULT_GRAM_BATCH_SIZE = 10_000
    available_entangling_mth = {"fswap", "identity", "hadamard"}

    def __init__(
            self,
            *,
            gram_batch_size: int = DEFAULT_GRAM_BATCH_SIZE,
            random_state: int = 0,
            n_qubits: int = DEFAULT_N_QUBITS,
            depth: Optional[int] = None,
            rotations: str = "Y,Z",
            entangling_mth: str = "fswap",
    ):
        super().__init__(
            gram_batch_size=gram_batch_size,
            random_state=random_state,
            n_qubits=n_qubits,
        )
        self.depth = depth
        self.rotations = rotations
        self.entangling_mth = entangling_mth
        if self.entangling_mth not in self.available_entangling_mth:
            raise ValueError(f"Unknown entangling method: {self.entangling_mth}.")
        self.bias = None
        self.encoder = torch.nn.Flatten()
        self.data_scaling = None

    def fit(self, x_train: Union[NDArray, torch.Tensor], y_train: Optional[Union[NDArray, torch.Tensor]] = None):
        super().fit(x_train, y_train)
        n_inputs = int(np.prod(x_train.shape[1:]))
        self.bias = Parameter(torch.from_numpy(self.np_rn_gen.random(n_inputs))).to(dtype=self.R_DTYPE)
        self.data_scaling = torch.pi * Parameter(torch.ones(n_inputs)).to(dtype=self.R_DTYPE)
        if self.depth is None:
            self.depth = int(max(1, np.ceil(x_train.shape[-1] / self.n_qubits)))
        return self

    def ansatz(self, x):
        x = to_tensor(x, dtype=self.R_DTYPE).to(device=self.device)
        x = self.bias + self.data_scaling * self.encoder(x)

        wires_double = PATTERN_TO_WIRES["double"](self.wires)
        wires_double_odd = PATTERN_TO_WIRES["double_odd"](self.wires)
        wires_patterns = [wires_double, wires_double_odd]
        for layer in range(self.depth):
            sub_x = x[..., layer * self.n_qubits : (layer + 1) * self.n_qubits]
            yield from SptmAngleEmbedding(sub_x, wires=self.wires, rotations=self.rotations).decomposition()
            wires_list = wires_patterns[layer % len(wires_patterns)]
            for wires in wires_list:
                if self.entangling_mth == "fswap":
                    yield SptmCompZX(wires=wires)
                elif self.entangling_mth == "hadamard":
                    yield SptmCompHH(wires=wires)
                elif self.entangling_mth == "identity":
                    pass
                else:
                    raise ValueError(f"Unknown entangling method: {self.entangling_mth}")
        return


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
