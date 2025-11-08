from typing import Iterator, Optional, Tuple

import numpy as np
import pennylane as qml
import torch

from matchcake import NonInteractingFermionicDevice
from matchcake.utils.operators import adjoint_generator
from matchcake.utils.torch_utils import to_tensor

from .gram_matrix import GramMatrix
from .kernel import Kernel


class NIFKernel(Kernel):
    DEFAULT_N_QUBITS = 12
    DEFAULT_GRAM_BATCH_SIZE = 10_000

    def __init__(
        self,
        *,
        gram_batch_size: int = DEFAULT_GRAM_BATCH_SIZE,
        random_state: int = 0,
        n_qubits: int = DEFAULT_N_QUBITS,
    ):
        super().__init__(
            gram_batch_size=gram_batch_size,
            random_state=random_state,
        )
        self.R_DTYPE = torch.float32
        self._q_device = NonInteractingFermionicDevice(n_qubits, r_dtype=self.R_DTYPE)

    def forward(self, x0: torch.Tensor, x1: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x1 is None:
            x1 = x0
        x0, x1 = to_tensor(x0, dtype=self.R_DTYPE), to_tensor(x1, dtype=self.R_DTYPE)
        kernel = self.compute_similarities(x0, x1)
        return kernel

    def ansatz(self, x: torch.Tensor):
        raise NotImplementedError

    def compute_similarities(self, x0: torch.Tensor, x1: torch.Tensor):
        def _func(indices):
            b_x0, b_x1 = x0[indices[:, 0]], x1[indices[:, 1]]
            return self._q_device.execute_generator(
                self.circuit(b_x0, b_x1),
                observable=qml.Projector(np.zeros(self.n_qubits, dtype=int), wires=self.wires),
                output_type="expval",
                reset=True,
            )

        gram = GramMatrix((x0.shape[0], x1.shape[0]), requires_grad=self.training)
        gram.apply_(_func, batch_size=self.gram_batch_size, symmetrize=True)
        return gram.to_tensor().to(device=self.device)

    def circuit(self, x0, x1):
        yield from self.ansatz(x0)
        yield from adjoint_generator(self.ansatz(x1))
        return

    @property
    def wires(self):
        return self._q_device.wires

    @property
    def n_qubits(self) -> int:
        return len(self._q_device.wires)

    @n_qubits.setter
    def n_qubits(self, value: int):
        self._q_device = NonInteractingFermionicDevice(value, r_dtype=self.R_DTYPE)
