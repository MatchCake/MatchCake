import numbers
import warnings
from typing import Any, Hashable, Optional, Union, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch.nn import Parameter

from ...typing import TensorLike
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
    SptmAngleEmbedding,
    SptmCompHH,
    SptmCompZX,
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

    DEFAULT_ROTATIONS = "Y,Z"
    DEFAULT_ENTANGLING_MTH = "fswap"
    available_entangling_mth = {"fswap", "identity", "hadamard"}
    PARAMETER_NAMES = ("bias_", "data_scaling_")
    MAX_DEPTH = 10_000
    # Default for a kernel pickled by an older version, which holds no `depth` attribute.
    depth: Optional[int] = None

    @classmethod
    def _validate_depth(cls, depth: Optional[int]) -> Optional[int]:
        """
        Checks that a number of layers is None or a positive integer of at most `MAX_DEPTH`.

        :param depth: The number of layers to check.
        :return: The number of layers as an int, or None.
        :rtype: Optional[int]
        :raises ValueError: If the number of layers is not None nor a positive integer of at
            most `MAX_DEPTH`.
        """
        if depth is None:
            return None
        is_integer = isinstance(depth, numbers.Integral) and not isinstance(depth, bool)
        if not is_integer or not 1 <= depth <= cls.MAX_DEPTH:
            raise ValueError(f"The depth must be None or a positive integer of at most {cls.MAX_DEPTH}, got {depth!r}.")
        return int(depth)

    def __init__(
        self,
        *,
        gram_batch_size: int = NIFKernel.DEFAULT_GRAM_BATCH_SIZE,
        random_state: int = NIFKernel.DEFAULT_RANDOM_STATE,
        alignment: bool = NIFKernel.DEFAULT_ALIGNMENT,
        alignment_iterations: int = NIFKernel.DEFAULT_ALIGNMENT_ITERATIONS,
        alignment_learning_rate: float = NIFKernel.DEFAULT_ALIGNMENT_LEARNING_RATE,
        alignment_early_stopping_patience: int = NIFKernel.DEFAULT_ALIGNMENT_EARLY_STOPPING_PATIENCE,
        alignment_early_stopping_threshold: float = NIFKernel.DEFAULT_ALIGNMENT_EARLY_STOPPING_THRESHOLD,
        n_qubits: int = NIFKernel.DEFAULT_N_QUBITS,
        r_dtype: Optional[torch.dtype] = None,
        c_dtype: Optional[torch.dtype] = None,
        rotations: str = DEFAULT_ROTATIONS,
        entangling_mth: str = DEFAULT_ENTANGLING_MTH,
        depth: Optional[int] = None,
    ):
        """
        Initializes the class with specified parameters for quantum circuit design and
        data processing. Ensures that user-provided configurations such as the
        entangling method are valid.

        :param gram_batch_size: Size of the gram batch, used for processing data in batches.
        :param random_state: Seed for random number generator to ensure reproducibility.
        :param alignment: A boolean flag indicating whether to perform kernel alignment during fitting.
        :param alignment_iterations: The maximum number of iterations for kernel alignment optimization.
        :param alignment_learning_rate: The learning rate for the optimizer used in kernel alignment.
        :param alignment_early_stopping_patience: The number of iterations to wait for improvement
            before stopping kernel alignment optimization.
        :param alignment_early_stopping_threshold: The threshold for determining improvement in kernel
            alignment optimization, used for early stopping criteria.
        :param n_qubits: Number of qubits to be used in the quantum circuit.
        :param r_dtype: The real floating-point dtype passed to the non-interacting fermionic device.
        :param c_dtype: The complex dtype passed to the non-interacting fermionic device.
        :param rotations: Types of rotations to be applied in the quantum circuit, specified
            as a comma-separated string (e.g., "Y,Z").
        :param entangling_mth: Method for entangling qubits in the quantum circuit. Must
            match one of the supported options.
        :param depth: Number of layers of the circuit, at most `MAX_DEPTH`. If None, it is computed
            at every fit from the number of features. A layer encodes `n_qubits` features, so a
            depth too small to encode every feature drops the last ones, with a warning.
        """
        super().__init__(
            gram_batch_size=gram_batch_size,
            random_state=random_state,
            alignment=alignment,
            alignment_iterations=alignment_iterations,
            alignment_learning_rate=alignment_learning_rate,
            alignment_early_stopping_patience=alignment_early_stopping_patience,
            alignment_early_stopping_threshold=alignment_early_stopping_threshold,
            n_qubits=n_qubits,
            r_dtype=r_dtype,
            c_dtype=c_dtype,
        )
        self.rotations = rotations
        self.entangling_mth = entangling_mth
        if self.entangling_mth not in self.available_entangling_mth:
            raise ValueError(f"Unknown entangling method: {self.entangling_mth}.")
        self.depth = depth
        self._depth: Optional[int] = self._validate_depth(depth)
        self.bias_: Optional[Parameter]
        self.register_parameter("bias_", None)
        self.encoder = torch.nn.Flatten()
        self.data_scaling_: Optional[Parameter]
        self.register_parameter("data_scaling_", None)

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Sets an attribute. A plain tensor assigned to `bias_` or `data_scaling_` is wrapped into
        a parameter, so both stay registered parameters of the kernel. The new parameter is
        trainable unless the one it replaces was frozen.

        :param name: Name of the attribute.
        :param value: Value of the attribute.
        """
        is_plain_tensor = isinstance(value, torch.Tensor) and not isinstance(value, Parameter)
        if name in self.PARAMETER_NAMES and is_plain_tensor:
            current = self._parameters.get(name)
            requires_grad = current.requires_grad if current is not None else True
            is_differentiable = value.is_floating_point() or value.is_complex()
            value = Parameter(value.detach(), requires_grad=requires_grad and is_differentiable)
        super().__setattr__(name, value)

    def fit(self, x_train: Union[NDArray, TensorLike], y_train: Optional[Union[NDArray, TensorLike]] = None):
        """
        Fits the model to the training data by initializing the parameters for the
        quantum operations and determining the depth of the operation layers based
        on the input shape and the number of qubits. The function adapts the model
        to the provided input shape and prepares it for further processing.

        Unless `depth` was set by the user, the number of layers is computed again at every fit
        from the number of features, so every feature of the new training data is encoded.

        The bias is drawn with `random_state` as the seed at every fit, so fitting again on the
        same data gives the same kernel.

        :param x_train: Training input data. Can be of type NDArray or torch.Tensor.
        :param y_train: Optional parameter for training target/output data. Can be of
            type NDArray or torch.Tensor.
        :return: Updated instance of the class after fitting the training data.
        """
        x_arr = cast(np.ndarray, x_train)
        if len(x_arr) == 0:
            raise ValueError("Cannot fit a kernel on zero samples.")
        depth = self._validate_depth(self.depth)
        n_inputs = int(np.prod(x_arr.shape[1:]))
        with self._restore_state_on_failure():
            bias = np.random.RandomState(seed=self.random_state).random(n_inputs)
            self.bias_ = Parameter(torch.as_tensor(bias, dtype=self.R_DTYPE, device=self.device))
            self.data_scaling_ = Parameter(torch.full((n_inputs,), torch.pi, dtype=self.R_DTYPE, device=self.device))
            self.__dict__.pop("depth_", None)
            self._depth = self._depth_in_use(depth, n_inputs)
            super().fit(x_train, y_train)
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
        x = to_tensor(x, dtype=self.R_DTYPE, device=self.device)
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

    def _depth_in_use(self, depth: Optional[int], n_inputs: int, stacklevel: int = 3) -> int:
        """
        Gives the number of layers of the circuit for a number of features. Warns when the given
        depth is too small to encode every feature.

        :param depth: The number of layers asked by the user, or None to compute it.
        :param n_inputs: The number of features to encode.
        :param stacklevel: The stack level of the warning, so it points to the caller's code.
        :return: The number of layers to use.
        :rtype: int
        """
        required_depth = int(max(1, np.ceil(n_inputs / self.n_qubits)))
        if depth is None:
            return required_depth
        if depth < required_depth:
            warnings.warn(
                f"A depth of {depth} encodes only the first {depth * self.n_qubits} of the {n_inputs} features. "
                f"A depth of {required_depth} is needed to encode all of them.",
                UserWarning,
                stacklevel=stacklevel,
            )
        return depth

    def _sptms_train_key(self) -> Hashable:
        return super()._sptms_train_key(), self.rotations, self.entangling_mth, self.depth_

    @property
    def depth_(self) -> Optional[int]:
        """
        Number of layers of the circuit in use. It is computed by `fit` from the number of
        features, unless `depth` is set. Setting it also sets `depth`, so `fit` and
        :func:`sklearn.base.clone` keep it; setting it to None hands it back to `fit`. On a
        fitted kernel, setting it to None computes it at once from the number of features of
        the training data.
        """
        # A kernel pickled by an older version holds `depth_` as a plain instance attribute.
        return self.__dict__.get("_depth", self.__dict__.get("depth_"))

    @depth_.setter
    def depth_(self, value: Optional[int]) -> None:
        depth = self._validate_depth(value)
        depth_in_use = depth
        if self.x_train_ is not None:
            n_inputs = int(np.prod(np.shape(self.x_train_)[1:]))  # type: ignore[arg-type]
            # The setter is reached through two `__setattr__` frames.
            depth_in_use = self._depth_in_use(depth, n_inputs, stacklevel=5)
        self.depth = depth
        self._depth = depth_in_use
