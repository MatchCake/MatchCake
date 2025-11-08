import numpy as np
import torch
from matchcake import NonInteractingFermionicDevice
from matchcake.operations import SingleParticleTransitionMatrixOperation
from matchcake.utils.torch_utils import to_tensor

from .nif_kernel import NIFKernel


class LinearNIFKernel(NIFKernel):
    DEFAULT_N_QUBITS = 12
    DEFAULT_GRAM_BATCH_SIZE = 10_000

    def __init__(
            self,
            *,
            gram_batch_size: int = DEFAULT_GRAM_BATCH_SIZE,
            random_state: int = 0,
            n_qubits: int = DEFAULT_N_QUBITS,
            bias: bool = True,
            encoder_activation: str = "Identity",
    ):
        super().__init__(
            gram_batch_size=gram_batch_size,
            random_state=random_state,
            n_qubits=n_qubits,
        )
        self._bias = bias
        self._encoder_activation = encoder_activation
        self.encoder = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.LazyLinear(
                self.encoder_out_indices[0].size,
                bias=self.bias,
                dtype=self.R_DTYPE
            ),
            getattr(torch.nn, self.encoder_activation)(),
        )

    def ansatz(self, x: torch.Tensor):
        x = to_tensor(x, dtype=self.R_DTYPE).to(device=self.device)
        h = torch.zeros(
            (x.shape[0], 2 * self.n_qubits, 2 * self.n_qubits),
            dtype=self.R_DTYPE, device=self.device
        )
        h[:, self.encoder_out_indices[0], self.encoder_out_indices[1]] = self.encoder(x)
        h[:, self.encoder_out_tril_indices[0], self.encoder_out_tril_indices[1]] = (
            -1.0 * h[:, self.encoder_out_tril_indices[1], self.encoder_out_tril_indices[0]]
        )
        sptm = torch.matrix_exp(h)
        yield SingleParticleTransitionMatrixOperation(sptm, wires=self.wires)

    @property
    def n_qubits(self) -> int:
        return len(self._q_device.wires)

    @n_qubits.setter
    def n_qubits(self, value: int):
        self._q_device = NonInteractingFermionicDevice(value, r_dtype=self.R_DTYPE)
        self.encoder[1] = torch.nn.LazyLinear(
            self.encoder_out_indices[0].size,
            bias=self.bias,
            dtype=self.R_DTYPE
        )

    @property
    def bias(self) -> bool:
        return self._bias

    @bias.setter
    def bias(self, value: bool):
        self._bias = value
        self.encoder[1] = torch.nn.LazyLinear(
            self.encoder_out_indices[0].size,
            bias=self._bias,
            dtype=self.R_DTYPE
        )

    @property
    def encoder_activation(self) -> str:
        return self._encoder_activation

    @encoder_activation.setter
    def encoder_activation(self, value: str):
        self._encoder_activation = value
        self.encoder[-1] = getattr(torch.nn, self._encoder_activation)()

    @property
    def encoder_out_indices(self):
        return np.triu_indices(2 * self.n_qubits, k=1)

    @property
    def encoder_out_tril_indices(self):
        return np.tril_indices(2 * self.n_qubits, k=-1)
