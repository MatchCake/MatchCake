from typing import Optional

import numpy as np
import pennylane as qml
import torch

from matchcake import NonInteractingFermionicDevice
from matchcake.utils.operators import adjoint_generator
from matchcake.utils.torch_utils import to_tensor

from .gram_matrix import GramMatrix
from .kernel import Kernel


class NIFKernel(Kernel):
    """
    Defines a quantum kernel using a non-interacting fermionic device for similarity
    computations between input data.

    This class encapsulates quantum kernel operations, leveraging a specific device
    implementation (NonInteractingFermionicDevice). It handles the computation of
    kernel values between input datasets, based on the provided similarity measure.
    The class supports flexible configuration of qubit numbers and processing batch
    sizes.

    :ivar R_DTYPE: Data type used for computation within the kernel.
    :type R_DTYPE: torch.dtype
    """

    DEFAULT_N_QUBITS = 12
    DEFAULT_GRAM_BATCH_SIZE = 10_000

    def __init__(
        self,
        *,
        gram_batch_size: int = DEFAULT_GRAM_BATCH_SIZE,
        random_state: int = 0,
        n_qubits: int = DEFAULT_N_QUBITS,
    ):
        """
        Initializes the class with specific parameters for quantum device configuration and
        random state.

        :param gram_batch_size: The batch size for the Gram computation.
        :param random_state: Seed for the random number generator to ensure reproducibility.
        :param n_qubits: The number of qubits for the non-interacting fermionic device.
        """
        super().__init__(
            gram_batch_size=gram_batch_size,
            random_state=random_state,
        )
        self.R_DTYPE = torch.float32
        self._q_device = NonInteractingFermionicDevice(n_qubits, r_dtype=self.R_DTYPE)

    def forward(self, x0: torch.Tensor, x1: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculates the similarities between input tensors and returns the kernel output.

        This method takes two tensors, converts them to the specified dtype, and computes
        their similarities using the `compute_similarities` method of the class. If the second
        input tensor is not provided, it defaults to the first tensor. The result is a kernel
        tensor representing the similarities between the inputs.

        :param x0: The first input tensor.
        :param x1: The second input tensor. Defaults to ``x0`` if not provided.
        :return: A tensor representing the computed kernel similarities.
        """
        if x1 is None:
            x1 = x0
        x0, x1 = to_tensor(x0, dtype=self.R_DTYPE), to_tensor(x1, dtype=self.R_DTYPE)
        kernel = self.compute_similarities(x0, x1)  # type: ignore
        return kernel

    def ansatz(self, x: torch.Tensor):
        """
        Represents an abstract method for implementing a specific ansatz.

        This method must be overridden in a subclass and is not implemented in
        the base class itself. It serves as a blueprint for any specific ansatz
        that is required by the framework or application.

        The responsibility of providing the implementation for the ansatz logic
        remains with the derived class.

        :param x: Input tensor.
        :type x: torch.Tensor
        :raises NotImplementedError: If the method is called without overriding
            it in a subclass.
        """
        raise NotImplementedError

    def compute_similarities(self, x0: torch.Tensor, x1: torch.Tensor):
        """
        Computes the similarity matrix between tensors `x0` and `x1`. The function evaluates
        quantum computational circuits for pairs of instances from `x0` and `x1` to compute a
        Gram matrix that represents the pairwise similarities. These values are then processed
        and provided as a tensor representation of the similarity matrix.

        :param x0: The first tensor containing a batch of samples for similarity computation.
        :type x0: torch.Tensor
        :param x1: The second tensor containing another batch of samples for similarity computation.
        :type x1: torch.Tensor
        :return: A tensor representing the computed similarity matrix.
        :rtype: torch.Tensor
        """

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
        """
        Generates a quantum circuit by applying an ansatz followed by its adjoint.

        The method first yields the result of applying the ansatz function to the
        first input, `x0`. Then, it yields the adjoint of the ansatz function applied
        to the second input, `x1`. This allows the generation of customized quantum
        circuits where specific operations are defined by the ansatz function.

        :param x0: The first parameter used for generating the circuit via the ansatz.
        :param x1: The second parameter used for generating the adjoint circuit via
            the ansatz.
        :return: Yields the quantum operations for constructing the circuit.
        """
        yield from self.ansatz(x0)
        yield from adjoint_generator(self.ansatz(x1))
        return

    @property
    def wires(self):
        return self.q_device.wires

    @property
    def n_qubits(self) -> int:
        return len(self.q_device.wires)

    @n_qubits.setter
    def n_qubits(self, value: int):
        """
        Sets the number of qubits to be used by the quantum device. Modifies
        the underlying quantum device accordingly.

        :param value: The number of qubits to be set for the quantum device.
        :type value: int
        """
        self._q_device = NonInteractingFermionicDevice(value, r_dtype=self.R_DTYPE)

    @property
    def q_device(self):
        return self._q_device
