from typing import Optional, Union

import numpy as np
import torch
from numpy.typing import NDArray
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
    The FermionicPQCKernel class defines a Parameterized Quantum Circuit (PQC) kernel for use in
    quantum machine learning applications. It enables data embedding into a quantum feature
    space, supports various entangling methods, and allows customization of circuit parameters.

    This class incorporates layered circuit design with options to specify the rotation gates,
    entangling method, depth, and the number of qubits. It is designed to transform classical
    data into quantum embeddings for downstream tasks like classification or regression using
    quantum or hybrid models.


    Inspired from: https://iopscience.iop.org/article/10.1088/2632-2153/acb0b4/meta#artAbst


    By default, the size of the kernel is computed as

    .. math::
        \text{size} = \max\left(2, \lceil\log_2(\text{n features} + 2)\rceil\right)

    and the depth is computed as

    .. math::
        \text{depth} = \max\left(1, \left(\frac{\text{n features}}{\text{size}} - 1\right)\right)


    :ivar DEFAULT_N_QUBITS: Default number of qubits to use in the circuit.
    :type DEFAULT_N_QUBITS: int
    :ivar DEFAULT_GRAM_BATCH_SIZE: Default batch size for Gram matrix computations.
    :type DEFAULT_GRAM_BATCH_SIZE: int
    :ivar available_entangling_mth: Set of available entangling methods supported by the circuit.
    :type available_entangling_mth: set[str]
    :ivar depth: Number of entangling layers in the circuit. If set to None, it is dynamically
        computed based on the input data and number of qubits.
    :type depth: Optional[int]
    :ivar rotations: String describing the sequence of rotation gates used for embedding angles.
    :type rotations: str
    :ivar entangling_mth: Method used to introduce entanglement between qubits. Possible values
        include "fswap", "identity", and "hadamard".
    :type entangling_mth: str
    :ivar bias_: Bias tensor added to the input data for encoding. It is initialized during
        the `fit` method call.
    :type bias_: torch.nn.parameter.Parameter
    :ivar encoder: Encoder used to flatten the input data before embedding.
    :type encoder: torch.nn.Module
    :ivar data_scaling_: Scaling factor applied to the input data for encoding. It is
        initialized during the `fit` method call.
    :type data_scaling_: torch.nn.parameter.Parameter
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
        rotations: str = "Y,Z",
        entangling_mth: str = "fswap",
    ):
        """
        Initializes the class with specified parameters for quantum circuit design and
        data processing. Ensures that user-provided configurations such as the
        entangling method are valid.

        :param gram_batch_size: Size of the gram batch, used for processing data in batches.
        :param random_state: Seed for random number generator to ensure reproducibility.
        :param n_qubits: Number of qubits to be used in the quantum circuit.
        :param rotations: Types of rotations to be applied in the quantum circuit, specified
            as a comma-separated string (e.g., "Y,Z").
        :param entangling_mth: Method for entangling qubits in the quantum circuit. Must
            match one of the supported options.
        """
        super().__init__(
            gram_batch_size=gram_batch_size,
            random_state=random_state,
            n_qubits=n_qubits,
        )
        self.rotations = rotations
        self.entangling_mth = entangling_mth
        if self.entangling_mth not in self.available_entangling_mth:
            raise ValueError(f"Unknown entangling method: {self.entangling_mth}.")
        self.depth_: Optional[int] = None
        self.bias_: Optional[Parameter] = None
        self.encoder = torch.nn.Flatten()
        self.data_scaling_: Optional[Parameter] = None

    def fit(self, x_train: Union[NDArray, torch.Tensor], y_train: Optional[Union[NDArray, torch.Tensor]] = None):
        """
        Fits the model to the training data by initializing the parameters for the
        quantum operations and determining the depth of the operation layers based
        on the input shape and the number of qubits. The function adapts the model
        to the provided input shape and prepares it for further processing.

        :param x_train: Training input data. Can be of type NDArray or torch.Tensor.
        :param y_train: Optional parameter for training target/output data. Can be of
            type NDArray or torch.Tensor.
        :return: Updated instance of the class after fitting the training data.
        """
        super().fit(x_train, y_train)
        n_inputs = int(np.prod(x_train.shape[1:]))
        self.bias_ = Parameter(torch.from_numpy(self.np_rn_gen.random(n_inputs))).to(dtype=self.R_DTYPE)  # type: ignore
        self.data_scaling_ = torch.pi * Parameter(torch.ones(n_inputs)).to(dtype=self.R_DTYPE)  # type: ignore
        if self.depth_ is None:
            self.depth_ = int(max(1, np.ceil(x_train.shape[-1] / self.n_qubits)))
        return self

    def ansatz(self, x):
        """
        Generates a quantum circuit ansatz based on specific encoding, entangling methods, and configurations.

        The ansatz consists of layers of a circuit that perform decomposition of encoded inputs over wires.
        It includes a choice of entangling methods and alternates over defined patterns of wire connectivity
        for double and double_odd configurations. Parameters are scaled and biased based on internal attributes.

        :param x: Input to be processed. The input is tensor-compatible and undergoes encoding.
        :type x: Any tensor-compatible object
        :raises ValueError: If the specified entangling method (`entangling_mth`) is not supported.
        :yield: Decompositions of specified operations for inclusion in a quantum circuit.

        """
        x = to_tensor(x, dtype=self.R_DTYPE).to(device=self.device)
        x = self.bias_ + self.data_scaling_ * self.encoder(x)

        wires_double = PATTERN_TO_WIRES["double"](self.wires)
        wires_double_odd = PATTERN_TO_WIRES["double_odd"](self.wires)
        wires_patterns = [wires_double, wires_double_odd]
        for layer in range(self.depth_):
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
