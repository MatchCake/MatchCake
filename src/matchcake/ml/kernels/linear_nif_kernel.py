import numpy as np
import torch

from matchcake import NonInteractingFermionicDevice
from matchcake.operations import SingleParticleTransitionMatrixOperation
from matchcake.utils.torch_utils import to_tensor

from .nif_kernel import NIFKernel


class LinearNIFKernel(NIFKernel):
    """
    Represents a specialized kernel model extending the NIFKernel class.

    This class provides a quantum kernel suitable for simulations utilizing a
    non-interacting fermionic hardware backend with customizable qubit
    configurations and encoder activations. The class heavily integrates
    PyTorch modules for defining the encoding layer and uses matrix operations
    to define the ansatz required for quantum computations. Allows fine-tuning
    of kernel parameters like bias, encoder activation function, and qubit
    distribution, making it highly adaptable to various quantum learning tasks.

    :ivar DEFAULT_N_QUBITS: Default number of qubits used in the kernel.
    :type DEFAULT_N_QUBITS: int
    :ivar DEFAULT_GRAM_BATCH_SIZE: Default size of the Gram matrix computation batch.
    :type DEFAULT_GRAM_BATCH_SIZE: int
    """

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
        """
        Initializes the class with configurable parameters for the model's encoder
        and quantum settings. It manages setup for batch processing, randomness seed,
        number of qubits, output bias and activation type of the encoder.

        :param gram_batch_size: Number of samples to include in each batch for
            processing. Optimally adjusts memory and computation requirements.
        :type gram_batch_size: int
        :param random_state: Seed value for random number generation to ensure
            reproducibility and consistency.
        :type random_state: int
        :param n_qubits: Number of qubits used in the quantum computation process.
        :type n_qubits: int
        :param bias: Determines if the linear layers in the encoder should include a
            bias term.
        :type bias: bool
        :param encoder_activation: The activation function applied in the encoder
            layers. Should match options available in torch.nn (e.g., "Identity").
        :type encoder_activation: str
        """
        super().__init__(
            gram_batch_size=gram_batch_size,
            random_state=random_state,
            n_qubits=n_qubits,
        )
        self._bias = bias
        self._encoder_activation = encoder_activation
        self.encoder = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.LazyLinear(self.encoder_out_indices[0].size, bias=self.bias, dtype=self.R_DTYPE),
            getattr(torch.nn, self.encoder_activation)(),
        )

    def ansatz(self, x: torch.Tensor):
        """
        Applies the ansatz to the input data for quantum computation. The method
        first processes the input tensor, creating a Hamiltonian matrix, and then
        calculates the single-particle transition matrix. Once computed, the method
        yields a single-particle transition matrix operation that can be applied to
        quantum wires.

        :param x: The input tensor containing data to be encoded in the ansatz.
        :type x: torch.Tensor
        :return: A generator yielding a single-particle transition matrix operation.
        :rtype: Generator[SingleParticleTransitionMatrixOperation, None, None]
        """
        x = to_tensor(x, dtype=self.R_DTYPE).to(device=self.device)
        h = torch.zeros((x.shape[0], 2 * self.n_qubits, 2 * self.n_qubits), dtype=self.R_DTYPE, device=self.device)
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
        self.encoder[1] = torch.nn.LazyLinear(self.encoder_out_indices[0].size, bias=self.bias, dtype=self.R_DTYPE)

    @property
    def bias(self) -> bool:
        return self._bias

    @bias.setter
    def bias(self, value: bool):
        self._bias = value
        self.encoder[1] = torch.nn.LazyLinear(self.encoder_out_indices[0].size, bias=self._bias, dtype=self.R_DTYPE)

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
