from functools import cached_property, partial
from typing import Optional

import numpy as np
import pennylane as qml
import torch
from tqdm import tqdm

from ...devices.nif_device import NonInteractingFermionicDevice
from ...operations import SingleParticleTransitionMatrixOperation
from ...typing import TensorLike
from ...utils.torch_utils import to_tensor
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
        alignment: bool = False,
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
            alignment=alignment,
        )
        self.R_DTYPE = torch.float32
        self._q_device = NonInteractingFermionicDevice(n_qubits, r_dtype=self.R_DTYPE)

    def forward(self, x0: TensorLike, x1: Optional[TensorLike] = None, **kwargs) -> TensorLike:
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
        x0 = to_tensor(x0, dtype=self.R_DTYPE)  # type: ignore
        x1 = to_tensor(x1, dtype=self.R_DTYPE)  # type: ignore
        self.to(device=self.device)
        kernel = self.compute_similarities(x0, x1, **kwargs)  # type: ignore
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

    def compute_similarities(self, x0: torch.Tensor, x1: torch.Tensor, **kwargs):
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
        gram = GramMatrix((x0.shape[0], x1.shape[0]), requires_grad=self.training)
        p_bar = tqdm(
            total=int(np.prod(gram.shape)),
            desc="Computing Gram matrix",
            unit="element",
            disable=not kwargs.get("verbose", False),
        )
        p_bar.set_postfix_str("Compiling x0 circuits ...")
        sptm0 = self._x_to_sptm(x0)
        p_bar.set_postfix_str("Compiling x1 circuits ...")
        if kwargs.get("x1_train", False):
            sptm1 = self.sptms_train
        else:
            sptm1 = self._x_to_sptm(x1)
        p_bar.set_postfix_str("Evaluating Gram matrix ...")
        _func = partial(self._compute_similarities_func, sptm0=sptm0, sptm1=sptm1, p_bar=p_bar)
        gram.apply_(_func, batch_size=self.gram_batch_size, symmetrize=True)
        p_bar.set_postfix_str("Done.")
        p_bar.close()
        return gram.to_tensor().to(dtype=self.R_DTYPE)

    def circuit(self, sptm0: TensorLike, sptm1: TensorLike):
        """
        Applies a quantum circuit consisting of two operations to the specified wires.
        The circuit applies a single-particle transition matrix operation followed by its
        adjoint (conjugate transpose).

        :param sptm0: The first single-particle transition matrix operation to be applied.
            This operation acts on the specified wires in the circuit.
        :param sptm1: The second single-particle transition matrix operation to be
            applied, followed by its adjoint. This operation also acts on the specified wires.
        :return: Yields the sequential operations in the circuit application. This
            does not return a concrete value but facilitates the execution of the quantum
            circuit.
        """
        yield SingleParticleTransitionMatrixOperation(sptm0, wires=self.wires)
        yield SingleParticleTransitionMatrixOperation(sptm1, wires=self.wires).adjoint()
        return

    def _x_to_sptm(self, x: TensorLike) -> TensorLike:
        """
        Transforms the given input tensor into a single particle transition matrix (SPTM) by
        applying the user's ansatz. The function prepares the quantum device, executes the
        generator method on the ansatz with the given input, and retrieves the resulting
        SPTM.

        :param x: Input tensor to be transformed.
        :type x: TensorLike
        :return: The single particle transition matrix (SPTM) resulting from the ansatz execution.
        :rtype: TensorLike
        """
        x = to_tensor(x, dtype=self.R_DTYPE)  # type: ignore
        batched_indices = np.array_split(np.arange(len(x)), np.ceil(len(x) / self.gram_batch_size))
        sptms = []
        for batch_indices in batched_indices:
            bx = to_tensor(x[batch_indices], dtype=self.R_DTYPE, device=self.device)  # type: ignore
            self._q_device.execute_generator(self.ansatz(bx), reset=True)
            sptms.append(to_tensor(self._q_device.global_sptm.matrix(), dtype=self.R_DTYPE, device=x.device))
        stacked_sptms = qml.math.stack(sptms, axis=0).reshape(
            x.shape[0], 2 * self._q_device.num_wires, 2 * self._q_device.num_wires
        )
        return stacked_sptms

    def _compute_similarities_func(self, indices, sptm0, sptm1, p_bar):
        b_sptm0, b_sptm1 = sptm0[indices[:, 0]], sptm1[indices[:, 1]]
        b_sptm0 = to_tensor(b_sptm0, dtype=self.R_DTYPE, device=self.device)  # type: ignore
        b_sptm1 = to_tensor(b_sptm1, dtype=self.R_DTYPE, device=self.device)  # type: ignore
        expval = self._q_device.execute_generator(
            self.circuit(b_sptm0, b_sptm1),
            observable=qml.Projector(np.zeros(self.n_qubits, dtype=int), wires=self.wires),
            output_type="expval",
            reset=True,
        )
        p_bar.update(len(indices))
        return expval

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

    @cached_property
    def sptms_train(self) -> TensorLike:
        return self._x_to_sptm(self.x_train_)
