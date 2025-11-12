from functools import cached_property
from typing import Any, Optional, Tuple, Union

import pennylane as qml
import torch
from pennylane.operation import Operation
from pennylane.typing import TensorLike
from pennylane.wires import Wires, WiresLike

from .. import matchgate_parameter_sets as mps
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
        shapes = [qml.math.shape(p) for p in [a, b, c, d, w, x, y, z] if p is not None]
        batch_sizes = list(set([s[0] for s in shapes if len(s) > 0]))
        assert len(batch_sizes) <= 1, f"Expect the same batch size for every parameters. Got: {batch_sizes}."
        batch_size = batch_sizes[0] if len(batch_sizes) > 0 else 1
        a, b, c, d, w, x, y, z = [
            (
                to_tensor(p, dtype=dtype, device=device)
                if p is not None
                else torch.zeros((batch_size,), dtype=dtype, device=device)
            )
            for p in [a, b, c, d, w, x, y, z]
        ]
        matrix = torch.zeros((batch_size, 4, 4), dtype=dtype, device=device)
        matrix[..., 0, 0] = a
        matrix[..., 0, 3] = b
        matrix[..., 3, 0] = c
        matrix[..., 3, 3] = d
        matrix[..., 1, 1] = w
        matrix[..., 1, 2] = x
        matrix[..., 2, 1] = y
        matrix[..., 2, 2] = z
        if len(batch_sizes) == 0:
            matrix = matrix[0]
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
        return cls.from_std_params(
            a=outer_matrix[..., 0, 0],
            b=outer_matrix[..., 0, 1],
            c=outer_matrix[..., 1, 0],
            d=outer_matrix[..., 1, 1],
            w=inner_matrix[..., 0, 0],
            x=inner_matrix[..., 0, 1],
            y=inner_matrix[..., 1, 0],
            z=inner_matrix[..., 1, 1],
            wires=wires,
            dtype=dtype,
            device=device,
            **kwargs,
        )

    @classmethod
    def from_polar_params(
        cls,
        r: Optional[TensorLike] = None,
        *,
        wires=None,
        dtype: torch.dtype = torch.float64,
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> "MatchgateOperation": ...

    @classmethod
    def random_params(cls, batch_size=None, **kwargs):
        seed = kwargs.pop("seed", None)
        ...

    @classmethod
    def random(cls, wires: Wires, batch_size=None, **kwargs) -> "MatchgateOperation":
        return cls(
            cls.random_params(batch_size=batch_size, wires=wires, **kwargs),
            wires=wires,
            **kwargs,
        )

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
        matrix: TensorLike,
        wires=None,
        id=None,
        default_dtype: torch.dtype = torch.complex128,
        default_device: Optional[torch.device] = None,
        **kwargs,
    ):
        if wires is not None:
            wires = Wires(wires)
            assert len(wires) == 2, f"MatchgateOperation requires exactly 2 wires, got {len(wires)}."
            assert wires[-1] - wires[0] == 1, f"MatchgateOperation requires consecutive wires, got {wires}."

        if qml.math.get_interface(matrix) != "torch":
            matrix = to_tensor(matrix, dtype=default_dtype, device=default_device)
        self.draw_label_params = kwargs.get("draw_label_params", None)
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

    def get_padded_single_particle_transition_matrix(self, wires=None):
        r"""
        Return the padded single particle transition matrix in order to have the block diagonal form where
        the block is the single particle transition matrix at the corresponding wires.

        :param wires: The wires of the whole system.
        :return: padded single particle transition matrix
        """
        return self.to_sptm_operation().pad(wires=wires).matrix()

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
            return torch.allclose(expected_zero, torch.zeros_like(expected_zero), atol=1e-5)

    def _check_m_dagger_m_constraint(self) -> bool:
        with torch.no_grad():
            m_dagger_m = torch.einsum("...ji,...jk->...ik", torch.conj(self.matrix()), self.matrix())
            expected_zero = m_dagger_m - torch.eye(4)
            return torch.allclose(expected_zero, torch.zeros_like(expected_zero), atol=1e-5)

    def _check_det_constraint(self) -> bool:
        with torch.no_grad():
            outer_determinant = torch.linalg.det(self.outer_gate_data)
            inner_determinant = torch.linalg.det(self.inner_gate_data)
            return qml.math.allclose(outer_determinant, inner_determinant, atol=1e-5)

    def _check_is_matchgate(self):
        if not self._check_m_m_dagger_constraint():
            raise ValueError(r"The matchgate does not satisfy the M M^\dagger constraint.")
        if not self._check_m_dagger_m_constraint():
            raise ValueError(r"The matchgate does not satisfy the M^\dagger M constraint.")  # pragma: no cover
        if not self._check_det_constraint():
            raise ValueError(r"The matchgate does not satisfy the determinant constraint.")

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
