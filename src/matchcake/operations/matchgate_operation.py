from functools import cached_property
from typing import Any, Optional, Tuple, Union

import pennylane as qml
import torch
from pennylane.operation import Operation
from pennylane.wires import Wires, WiresLike

from ..matchgate_parameter_sets.matchgate_polar_params import MatchgatePolarParams
from ..matchgate_parameter_sets.matchgate_standard_params import (
    MatchgateParams,
    MatchgateStandardParams,
)
from ..typing import TensorLike
from ..utils import (
    make_single_particle_transition_matrix_from_gate,
    make_wires_continuous,
)
from ..utils.math import fermionic_operator_matmul
from ..utils.torch_utils import to_tensor
from .single_particle_transition_matrices.single_particle_transition_matrix import (
    SingleParticleTransitionMatrixOperation,
)


class MatchgateOperation(Operation):
    r"""
    A matchgate is a matrix of the form

    .. math::
        \begin{pmatrix}
            a & 0 & 0 & b \\
            0 & w & x & 0 \\
            0 & y & z & 0 \\
            c & 0 & 0 & d
        \end{pmatrix}

    where :math:`a, b, c, d, w, x, y, z \in \mathbb{C}`. The matrix M can be decomposed as

    .. math::
        A = \begin{pmatrix}
            a & b \\
            c & d
        \end{pmatrix}

    and

    .. math::
        W = \begin{pmatrix}
            w & x \\
            y & z
        \end{pmatrix}

    The matchgate is a unitary matrix if and only if the following conditions are satisfied:

    .. math::
        M^\dagger M = \mathbb{I} \quad \text{and} \quad MM^\dagger = \mathbb{I}

    where :math:`\mathbb{I}` is the identity matrix and :math:`M^\dagger` is the conjugate transpose of :math:`M`,
    and the following condition is satisfied:

    .. math::
        \det(A) = \det(W)
    """

    num_params = 1
    ndim_params = 2
    num_wires = 2
    par_domain = "A"

    grad_method = "A"
    grad_recipe = None

    generator = None

    @classmethod
    def from_std_params(
        cls,
        a: Optional[TensorLike] = None,
        b: Optional[TensorLike] = None,
        c: Optional[TensorLike] = None,
        d: Optional[TensorLike] = None,
        w: Optional[TensorLike] = None,
        x: Optional[TensorLike] = None,
        y: Optional[TensorLike] = None,
        z: Optional[TensorLike] = None,
        *,
        wires=None,
        dtype: torch.dtype = torch.complex128,
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> "MatchgateOperation":
        std_params = MatchgateStandardParams(a=a, b=b, c=c, d=d, w=w, x=x, y=y, z=z)
        matrix = std_params.matrix(dtype=dtype, device=device)
        return MatchgateOperation(matrix, wires=wires, **kwargs)

    @classmethod
    def from_sub_matrices(
        cls,
        outer_matrix: TensorLike,
        inner_matrix: TensorLike,
        *,
        wires=None,
        dtype: torch.dtype = torch.complex128,
        device: Optional[torch.device] = None,
        **kwargs,
    ):
        std_params = MatchgateStandardParams.from_sub_matrices(outer_matrix, inner_matrix)
        matrix = std_params.matrix(dtype=dtype, device=device)
        return MatchgateOperation(matrix, wires=wires, **kwargs)

    @classmethod
    def from_polar_params(
        cls,
        r0: Optional[TensorLike] = None,
        r1: Optional[TensorLike] = None,
        theta0: Optional[TensorLike] = None,
        theta1: Optional[TensorLike] = None,
        theta2: Optional[TensorLike] = None,
        theta3: Optional[TensorLike] = None,
        theta4: Optional[TensorLike] = None,
        *,
        wires=None,
        dtype: torch.dtype = torch.complex128,
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> "MatchgateOperation":
        r"""
        Matchgate from polar parameters.

        They are the parameters of a Matchgate operation in the standard form which is a 4x4 matrix

        .. math::

                \begin{bmatrix}
                    r_0 e^{i\theta_0} & 0 & 0 & (\sqrt{1 - r_0^2}) e^{-i(\theta_1+\pi)} \\
                    0 & r_1 e^{i\theta_2} & (\sqrt{1 - r_1^2}) e^{-i(\theta_3+\pi)} & 0 \\
                    0 & (\sqrt{1 - r_1^2}) e^{i\theta_3} & r_1 e^{-i\theta_2} & 0 \\
                    (\sqrt{1 - r_0^2}) e^{i\theta_1} & 0 & 0 & r_0 e^{-i\theta_0}
                \end{bmatrix}

            where :math:`r_0, r_1, \theta_0, \theta_1, \theta_2, \theta_3, \theta_4` are the parameters.

        The polar parameters will be converted to standard parameterization. The conversion is given by

        .. math::
            \begin{align}
                a &= r_0 e^{i\theta_0} \\
                b &= (\sqrt{1 - r_0^2}) e^{i(\theta_2+\theta_4-(\theta_1+\pi))} \\
                c &= (\sqrt{1 - r_0^2}) e^{i\theta_1} \\
                d &= r_0 e^{i(\theta_2+\theta_4-\theta_0)} \\
                w &= r_1 e^{i\theta_2} \\
                x &= (\sqrt{1 - r_1^2}) e^{i(\theta_2+\theta_4-(\theta_3+\pi))} \\
                y &= (\sqrt{1 - r_1^2}) e^{i\theta_3} \\
                z &= r_1 e^{i\theta_4}
            \end{align}
        """
        polar_params = MatchgatePolarParams(
            r0=r0,
            r1=r1,
            theta0=theta0,
            theta1=theta1,
            theta2=theta2,
            theta3=theta3,
            theta4=theta4,
        )
        matrix = polar_params.matrix(dtype=dtype, device=device)
        return MatchgateOperation(matrix, wires=wires, **kwargs)

    @classmethod
    def random_params(
        cls,
        *,
        batch_size: Optional[int] = None,
        dtype: torch.dtype = torch.float64,
        device: Optional[torch.device] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> Union[TensorLike, MatchgateParams]:
        """
        Generates a tensor of random parameters. This method allows creating a tensor
        with random elements drawn from uniform (0, 1] for the two first elements
        and drawn from a uniform distribution [0, :math:`2\pi`).
        If ``batch_size`` is None, a single parameter set is
        returned instead of a batch.

        :param batch_size: The number of parameter sets to generate. If None, only a
                           single parameter set is generated.
        :type batch_size: Optional[int]
        :param dtype: The data type of the output tensor. Defaults to torch.float64.
        :param device: The device where the tensor will be allocated. Defaults to
                       the current default device if not specified.
        :param seed: A manual seed for reproducible random generation. If None, the
                     random generator is not seeded manually.
        :type seed: Optional[int]
        :param kwargs: Additional keyword arguments for extended configuration.
        :return: A tensor containing the generated random parameters. If ``batch_size``
                 is specified, the shape will be ``(batch_size, 7)``. Otherwise, the
                 returned tensor has shape ``(7,)``.
        :rtype: torch.Tensor
        """
        eff_batch_size = batch_size if batch_size is not None else 1
        params = torch.zeros((eff_batch_size, 7), dtype=torch.float64, device=device)
        rn_generator = torch.Generator(device=device or "cpu")
        if seed is not None:
            rn_generator.manual_seed(seed)
        params[..., 0:2] = torch.clip(
            torch.rand((eff_batch_size, 2), generator=rn_generator, dtype=torch.float64, device=device) + 1e-12,
            min=torch.tensor(1e-12, dtype=torch.float64, device=device),
            max=torch.tensor(1.0, dtype=torch.float64, device=device),
        )
        params[..., 2:] = (2 * torch.pi) * torch.rand(
            (eff_batch_size, 5), generator=rn_generator, dtype=torch.float64, device=device
        )
        params = params.to(dtype=dtype, device=device)
        if batch_size is None:
            params = params[0]
        return MatchgatePolarParams(
            r0=params[..., 0],
            r1=params[..., 1],
            theta0=params[..., 2],
            theta1=params[..., 3],
            theta2=params[..., 4],
            theta3=params[..., 5],
            theta4=params[..., 6],
        )

    @classmethod
    def random(
        cls,
        wires: Wires,
        *,
        batch_size: Optional[int] = None,
        dtype: torch.dtype = torch.complex128,
        device: Optional[torch.device] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> "MatchgateOperation":
        """
        Generate a random `MatchgateOperation` instance.

        This class method creates a new instance of `MatchgateOperation` with
        randomly initialized parameters. The generated object can be used
        for testing, experimentation, or random state initialization.

        The random values for the parameters can be controlled using the provided
        seed. Additional configuration can be done using various optional parameters
        such as the batch size, data type, device, and any implementation-specific
        keyword arguments.

        :param wires: Wires that the operation acts on. Can be a sequence or scalar.
        :type wires: Wires
        :param batch_size: Optional size of generated random batches. If None,
            a single gate is returned.
        :param dtype: Torch data type used for the random parameters. Defaults to
            `torch.complex128`.
        :param device: Optional Torch device on which the parameters will be created.
            If None, parameters are created on the default device.
        :param seed: Optional integer seed for controlling randomness. If None,
            the seed is not fixed.
        :param kwargs: Additional keyword arguments for specific configuration or
            compatibility.
        :return: A new `MatchgateOperation` instance with randomly initialized parameters.
        :rtype: MatchgateOperation
        """
        params = cls.random_params(batch_size=batch_size, dtype=dtype, device=device, seed=seed, **kwargs)
        return cls(params, wires=wires, dtype=dtype, device=device, **kwargs)

    @staticmethod
    def compute_matrix(*params, **hyperparams) -> torch.Tensor:
        return params[0]

    @staticmethod
    def compute_decomposition(
        *params: TensorLike,
        wires: Optional[WiresLike] = None,
        **hyperparameters: dict[str, Any],
    ):
        return [qml.QubitUnitary(params[0], wires=wires)]

    def __init__(
        self,
        matrix: Union[TensorLike, MatchgateParams],
        wires=None,
        id=None,
        default_dtype: torch.dtype = torch.complex128,
        default_device: Optional[torch.device] = None,
        **kwargs,
    ):
        if isinstance(matrix, MatchgateParams):
            matrix = matrix.matrix(dtype=default_dtype, device=default_device)
        if wires is not None:
            wires = Wires(wires)
            assert len(wires) == 2, f"MatchgateOperation requires exactly 2 wires, got {len(wires)}."
            assert wires[-1] - wires[0] == 1, f"MatchgateOperation requires consecutive wires, got {wires}."

        if qml.math.get_interface(matrix) != "torch":
            matrix = to_tensor(matrix, dtype=default_dtype, device=default_device)
        self.draw_label_params = kwargs.get("draw_label_params", None)
        self.kwargs = kwargs
        super().__init__(matrix, wires=wires, id=id)
        self._check_is_matchgate()

    def __matmul__(self, other) -> Union["MatchgateOperation", SingleParticleTransitionMatrixOperation]:
        if isinstance(other, SingleParticleTransitionMatrixOperation):
            return fermionic_operator_matmul(self.to_sptm_operation(), other)

        if not isinstance(other, MatchgateOperation):
            raise ValueError(f"Cannot multiply MatchgateOperation with {type(other)}")

        if self.wires != other.wires:
            return fermionic_operator_matmul(self.to_sptm_operation(), other.to_sptm_operation())

        return MatchgateOperation.from_std_params(
            a=self.a * other.a + self.b * other.c,
            b=self.a * other.b + self.b * other.d,
            c=self.c * other.a + self.d * other.c,
            d=self.c * other.b + self.d * other.d,
            w=self.w * other.w + self.x * other.y,
            x=self.w * other.x + self.x * other.z,
            y=self.y * other.w + self.z * other.y,
            z=self.y * other.x + self.z * other.z,
            wires=self.wires,
            **self.hyperparameters,
        )

    def to_sptm_operation(self) -> SingleParticleTransitionMatrixOperation:
        return SingleParticleTransitionMatrixOperation(
            self.single_particle_transition_matrix,
            wires=self.wires,
            **self.hyperparameters,
        )

    def get_padded_single_particle_transition_matrix(self, wires=None) -> SingleParticleTransitionMatrixOperation:
        r"""
        Return the padded single particle transition matrix in order to have the block diagonal form where
        the block is the single particle transition matrix at the corresponding wires.

        :param wires: The wires of the whole system.
        :return: padded single particle transition matrix
        """
        return self.to_sptm_operation().pad(wires=wires)

    def adjoint(self):
        return MatchgateOperation.from_std_params(
            a=qml.math.conjugate(self.a),
            b=qml.math.conjugate(self.c),
            c=qml.math.conjugate(self.b),
            d=qml.math.conjugate(self.d),
            w=qml.math.conjugate(self.w),
            x=qml.math.conjugate(self.y),
            y=qml.math.conjugate(self.x),
            z=qml.math.conjugate(self.z),
            wires=self.wires,
            dtype=self.dtype,
            device=self.device,
            **self.hyperparameters,
        )

    def label(self, decimals=None, base_label=None, cache=None):
        if self.draw_label_params is None:
            return super().label(decimals=decimals, base_label=base_label, cache=cache)

        op_label = base_label or self.__class__.__name__
        return f"{op_label}({self.draw_label_params})"

    def _check_m_m_dagger_constraint(self) -> bool:
        with torch.no_grad():
            m_m_dagger = torch.einsum("...ij,...kj->...ik", self.matrix(), torch.conj(self.matrix()))
            expected_zero = m_m_dagger - torch.eye(4)
            check = torch.allclose(expected_zero, torch.zeros_like(expected_zero), atol=1e-5)
        if not check:
            raise ValueError(
                rf"The matchgate does not satisfy the M M^\dagger constraint. Expected zeros. Got {expected_zero}"
            )
        return check

    def _check_m_dagger_m_constraint(self) -> bool:
        with torch.no_grad():
            m_dagger_m = torch.einsum("...ji,...jk->...ik", torch.conj(self.matrix()), self.matrix())
            expected_zero = m_dagger_m - torch.eye(4)
            check = torch.allclose(expected_zero, torch.zeros_like(expected_zero), atol=1e-5)
        if not check:
            raise ValueError(
                rf"The matchgate does not satisfy the M^\dagger M constraint. Expected zeros. Got {expected_zero}"
            )
        return check

    def _check_det_constraint(self) -> bool:
        with torch.no_grad():
            outer_determinant = torch.linalg.det(self.outer_gate_data)
            inner_determinant = torch.linalg.det(self.inner_gate_data)
            check = qml.math.allclose(outer_determinant, inner_determinant, atol=1e-5)
        if not check:
            raise ValueError(
                rf"The matchgate does not satisfy the determinant constraint. "
                rf"Expected equal. Got {outer_determinant} != {inner_determinant}"
            )
        return check

    def _check_is_matchgate(self):
        self._check_m_m_dagger_constraint()
        self._check_m_dagger_m_constraint()  # pragma: no cover
        self._check_det_constraint()

    @cached_property
    def single_particle_transition_matrix(self):
        matrix = make_single_particle_transition_matrix_from_gate(self.matrix())
        return matrix

    @cached_property
    def batch_size(self) -> Optional[int]:
        if qml.math.ndim(self.matrix()) == 2:
            return None
        return qml.math.shape(self.matrix())[0]

    @property
    def shape(self) -> Union[Tuple[int, int], Tuple[int, int, int]]:
        if self.batch_size is None:
            return 4, 4
        return self.batch_size, 4, 4

    @cached_property
    def sorted_wires(self) -> Wires:
        return Wires(sorted(self.wires.tolist()))

    @cached_property
    def cs_wires(self) -> Wires:
        return Wires(make_wires_continuous(self.wires))

    @property
    def outer_gate_data(self):
        r"""
        The gate data is the matrix

        .. math::
            \begin{pmatrix}
                a & 0 & 0 & b \\
                0 & w & x & 0 \\
                0 & y & z & 0 \\
                c & 0 & 0 & d
            \end{pmatrix}

        where :math:`a, b, c, d, w, x, y, z \in \mathbb{C}`. The outer gate data is the following sub-matrix of the
        matchgate matrix:

        .. math::
            \begin{pmatrix}
                a & b \\
                c & d
            \end{pmatrix}

        :return: The outer gate data.
        """
        batch_size = self.batch_size or 1
        matrix = torch.zeros((batch_size, 2, 2), dtype=self.dtype, device=self.device)
        matrix[..., 0, 0] = self.a
        matrix[..., 0, 1] = self.b
        matrix[..., 1, 0] = self.c
        matrix[..., 1, 1] = self.d
        if self.batch_size is None:
            matrix = matrix[0]
        return matrix

    @property
    def inner_gate_data(self):
        r"""
        The gate data is the matrix

        .. math::
            \begin{pmatrix}
                a & 0 & 0 & b \\
                0 & w & x & 0 \\
                0 & y & z & 0 \\
                c & 0 & 0 & d
            \end{pmatrix}

        where :math:`a, b, c, d, w, x, y, z \in \mathbb{C}`. The inner gate data is the following sub-matrix of the
        matchgate matrix:

        .. math::
            \begin{pmatrix}
                w & x \\
                y & z
            \end{pmatrix}

        :return:
        """
        batch_size = self.batch_size or 1
        matrix = torch.zeros((batch_size, 2, 2), dtype=self.dtype, device=self.device)
        matrix[..., 0, 0] = self.w
        matrix[..., 0, 1] = self.x
        matrix[..., 1, 0] = self.y
        matrix[..., 1, 1] = self.z
        if self.batch_size is None:
            matrix = matrix[0]
        return matrix

    @property
    def a(self) -> torch.Tensor:
        return self.matrix()[..., 0, 0]

    @property
    def b(self) -> torch.Tensor:
        return self.matrix()[..., 0, 3]

    @property
    def c(self) -> torch.Tensor:
        return self.matrix()[..., 3, 0]

    @property
    def d(self) -> torch.Tensor:
        return self.matrix()[..., 3, 3]

    @property
    def w(self) -> torch.Tensor:
        return self.matrix()[..., 1, 1]

    @property
    def x(self) -> torch.Tensor:
        return self.matrix()[..., 1, 2]

    @property
    def y(self) -> torch.Tensor:
        return self.matrix()[..., 2, 1]

    @property
    def z(self) -> torch.Tensor:
        return self.matrix()[..., 2, 2]

    @property
    def dtype(self) -> torch.dtype:
        return self.matrix().dtype

    @property
    def device(self) -> torch.device:
        return self.matrix().device
