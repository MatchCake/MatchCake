import numpy as np
import pennylane as qml
from pennylane.operation import StatePrepBase
from pennylane.typing import TensorLike
from pennylane.wires import Wires

from ... import utils
from ...operations.state_preparation.product_state import ProductState
from ...utils.math import convert_and_cast_like
from .probability_strategy import ProbabilityStrategy


class ProductStateProbabilityStrategy(ProbabilityStrategy):
    r"""Probability strategy for arbitrary product-state inputs evolved by a matchgate circuit.

    For an initial product state evolved by a matchgate circuit, the evolved :math:`2n \times 2n`
    Majorana covariance matrix is

    .. math::

        \Lambda(t) = Q^\top \Lambda_0 Q

    (real, antisymmetric), where :math:`\Lambda_0` is the product state's covariance and :math:`Q`
    the single-particle transition matrix (SPTM). The probability of basis outcome
    :math:`y \in \{0,1\}^n` is

    .. math::

        p(y) = \frac{1}{2^n} \left| \mathrm{Pf}\!\left(\Lambda(t) + \Lambda_y\right) \right|
             = \frac{1}{2^n} \sqrt{\left| \det\!\left(\Lambda(t) + \Lambda_y\right) \right|}

    where :math:`\Lambda_y` is the basis state's block-diagonal covariance:

    .. math::

        (\Lambda_y)_{2k,\,2k+1} = -(-1)^{y_k}, \quad k = 0,\ldots,n-1,

    antisymmetric with all other entries zero. For a marginal on a subset of wires :math:`W`,
    restrict to the measured Majorana block:

    .. math::

        p(y_W) = \frac{1}{2^{|W|}} \sqrt{\left| \det\!\left(\Lambda(t)\big|_W + \Lambda_{y_W}\right) \right|}

    where :math:`\Lambda(t)\big|_W` is the principal submatrix on Majorana indices
    :math:`\{2w, 2w+1 : w \in W\}`.

    Complexity: building :math:`\Lambda(t) = Q^\top \Lambda_0 Q` costs :math:`O(n^3)` and is
    computed once per circuit, then reused across all outcomes. Per queried outcome: one
    :math:`O(n^3)` determinant of a :math:`2n \times 2n` matrix. Batched over :math:`B` outcomes:
    :math:`O(Bn^3)`. This is the resummed form of the Hilbert-Schmidt overlap
    :math:`\mathrm{Tr}[\rho_G(t)\rho_y]`; the naive sector sum has :math:`2^n` terms, which the
    single determinant/Pfaffian collapses to polynomial time.

    References:

    - D. J. Brod, "Efficient classical simulation of matchgate circuits with generalized inputs
      and measurements," Phys. Rev. A 93, 062332 (2016), arXiv:1602.03539.
    - S. Bravyi, "Lagrangian representation for fermionic linear optics," Quantum Inf. Comput. 5,
      216 (2005), arXiv:quant-ph/0404180.
    """

    NAME: str = "ProductState"
    REQUIRES_KWARGS = ["covariance_matrix", "all_wires"]

    @staticmethod
    def build_lambda_y(target_binary_state: TensorLike, n_wires: int) -> np.ndarray:
        r"""Build the basis-state covariance matrix :math:`\Lambda_y`.

        Constructs the real antisymmetric :math:`(2k) \times (2k)` matrix with
        :math:`(\Lambda_y)_{2j,\,2j+1} = -(-1)^{y_j}` for :math:`j = 0, \ldots, k-1`.

        Accepts a single outcome ``(k,)`` and returns ``(2k, 2k)``, or a batch ``(B, k)``
        and returns ``(B, 2k, 2k)``.

        :param target_binary_state: Binary outcome(s) of shape ``(n_wires,)`` or ``(B, n_wires)``.
        :type target_binary_state: TensorLike
        :param n_wires: Number of measured wires :math:`k`.
        :type n_wires: int
        :return: Antisymmetric matrix of shape ``(2*n_wires, 2*n_wires)`` or ``(B, 2*n_wires, 2*n_wires)``.
        :rtype: np.ndarray
        """
        bits = np.asarray(target_binary_state).astype(int)
        single = bits.ndim == 1
        bits = np.atleast_2d(bits)  # (B, k)
        B, k = bits.shape
        lambda_y = np.zeros((B, 2 * k, 2 * k))
        j = np.arange(k)
        vals = -((-1.0) ** bits)  # (B, k)
        lambda_y[:, 2 * j, 2 * j + 1] = vals
        lambda_y[:, 2 * j + 1, 2 * j] = -vals
        return lambda_y[0] if single else lambda_y

    @staticmethod
    def extract_majorana_submatrix(
        covariance_matrix: TensorLike,
        wire_indices: np.ndarray,
    ) -> TensorLike:
        r"""Extract the principal Majorana submatrix for the given wire positions.

        :param covariance_matrix: Evolved covariance matrix of shape ``(..., 2n, 2n)``.
        :type covariance_matrix: TensorLike
        :param wire_indices: Positions of the measured wires in the full wire list, shape ``(k,)``.
        :type wire_indices: np.ndarray
        :return: Principal submatrix of shape ``(..., 2k, 2k)``.
        :rtype: TensorLike
        """
        wire_indices = np.asarray(wire_indices)
        majorana_indices = np.stack([2 * wire_indices, 2 * wire_indices + 1], axis=1).ravel()
        return covariance_matrix[..., majorana_indices[:, None], majorana_indices[None, :]]

    @staticmethod
    def _extract_majorana_submatrix_batch(
        covariance_matrix: TensorLike,
        all_wires: Wires,
        batch_wires: np.ndarray,
        k: int,
    ) -> TensorLike:
        r"""Extract per-outcome Majorana submatrices for a batch of wire subsets.

        :param covariance_matrix: Evolved covariance matrix of shape ``(..., 2n, 2n)``.
        :type covariance_matrix: TensorLike
        :param all_wires: Full device wire list.
        :type all_wires: Wires
        :param batch_wires: Per-outcome wire labels of shape ``(B, k)``.
        :type batch_wires: np.ndarray
        :param k: Number of measured wires per outcome.
        :type k: int
        :return: Submatrix batch of shape ``(..., B, 2k, 2k)``.
        :rtype: TensorLike
        """
        wire_indices_batch = np.array([all_wires.indices(Wires(w)) for w in batch_wires])  # (B, k)
        majorana_batch = np.stack([2 * wire_indices_batch, 2 * wire_indices_batch + 1], axis=2).reshape(
            len(batch_wires), 2 * k
        )  # (B, 2k)
        return covariance_matrix[..., majorana_batch[:, :, None], majorana_batch[:, None, :]]  # (..., B, 2k, 2k)

    def __call__(
        self,
        *,
        state_prep_op: StatePrepBase,
        target_binary_states: TensorLike,
        wires: Wires,
        **kwargs,
    ) -> TensorLike:
        r"""Compute probabilities for one or a batch of basis outcomes.

        Accepts a single outcome ``(k,)`` and returns a scalar, or a batch ``(B, k)``
        and returns ``(B,)``. When all outcomes share the same wires (the device default),
        the batch is handled in one vectorized determinant call over a ``(B, 2k, 2k)`` stack.

        Pass ``pfaffian_chunk_size`` to bound peak memory: the Pfaffian over the
        ``(..., 2k, 2k)`` stack is then reduced in slices of at most that many matrices along the
        flattened batch axis instead of all at once, which caps the determinant/backward
        workspace. Defaults to ``None`` (no chunking).

        :param state_prep_op: State preparation operation (not used directly; the evolved
            covariance matrix is passed via ``covariance_matrix``).
        :type state_prep_op: StatePrepBase
        :param target_binary_states: Binary outcomes of shape ``(k,)`` or ``(B, k)``.
        :type target_binary_states: TensorLike
        :param wires: Measured wires — ``Wires`` for single, ``ndarray(B, k)`` for batch.
        :type wires: Wires
        :return: Scalar or ``(B,)`` probabilities.
        :rtype: TensorLike
        """
        self.check_required_kwargs(kwargs)

        target_arr = np.asarray(target_binary_states)
        is_single = target_arr.ndim == 1
        k = target_arr.shape[-1]

        covariance_matrix: TensorLike = kwargs["covariance_matrix"]  # (..., 2n, 2n)
        all_wires = Wires(kwargs["all_wires"])
        kwargs.pop("show_progress", False)

        if isinstance(wires, int):
            wires = [wires]
        batch_wires = np.asarray(wires)

        # ``build_lambda_y`` returns (2k, 2k) for a single outcome and (B_q, 2k, 2k) for a
        # batch of outcomes. The covariance submatrix carries its own (state-prep) batch
        # axes. The two batches are independent and must broadcast on separate axes so that
        # a batch of B_q outcomes queried against B state preparations yields (B_q, ..., B).
        same_wires = True
        if is_single:
            wire_indices = all_wires.indices(Wires(batch_wires))
            lambda_t_w = self.extract_majorana_submatrix(covariance_matrix, wire_indices)  # (..., 2k, 2k)
        else:
            if batch_wires.ndim == 1:
                batch_wires = np.broadcast_to(batch_wires, target_arr.shape)
            same_wires = len(target_arr) <= 1 or np.all(batch_wires == batch_wires[0])
            if same_wires:
                wire_indices = all_wires.indices(Wires(batch_wires[0]))
                lambda_t_w = self.extract_majorana_submatrix(covariance_matrix, wire_indices)  # (..., 2k, 2k)
            else:
                lambda_t_w = self._extract_majorana_submatrix_batch(
                    covariance_matrix, all_wires, batch_wires, k
                )  # (..., B, 2k, 2k)

        lambda_y = convert_and_cast_like(self.build_lambda_y(target_arr, k), covariance_matrix)

        # When the state preparation is batched, ``lambda_t_w`` has leading batch axes that
        # the outcome batch of ``lambda_y`` would otherwise collide with. In the shared-wires
        # case the outcome axis is not yet present in ``lambda_t_w``, so insert the covariance
        # batch axes after the outcome axis of ``lambda_y`` to keep the two batches separate.
        if not is_single and same_wires:
            covariance_batch_ndim = qml.math.ndim(lambda_t_w) - 2
            if covariance_batch_ndim > 0:
                lambda_y_shape = tuple(qml.math.shape(lambda_y))  # (B_q, 2k, 2k)
                lambda_y = qml.math.reshape(
                    lambda_y,
                    lambda_y_shape[:1] + (1,) * covariance_batch_ndim + lambda_y_shape[1:],
                )

        combined = lambda_t_w + lambda_y
        chunk_size = kwargs.get("pfaffian_chunk_size", None)
        prob = (1.0 / 2**k) * qml.math.real(utils.pfaffian(combined, sign=False, chunk_size=chunk_size))
        return convert_and_cast_like(prob, covariance_matrix)

    def can_execute(self, state_prep_op: StatePrepBase) -> bool:
        """Return True for any :class:`~matchcake.operations.ProductState` input.

        :param state_prep_op: State preparation operation.
        :type state_prep_op: StatePrepBase
        :return: True when ``state_prep_op`` is a ``ProductState``.
        :rtype: bool
        """
        return isinstance(state_prep_op, ProductState)

    def _compute_single(
        self,
        *,
        state_prep_op: StatePrepBase,
        target_binary_state: TensorLike,
        wires: Wires,
        **kwargs,
    ) -> TensorLike:
        """Single-state entry point, delegates to :meth:`__call__`.

        :param state_prep_op: State preparation operation.
        :type state_prep_op: StatePrepBase
        :param target_binary_state: Binary outcome of shape ``(k,)``.
        :type target_binary_state: TensorLike
        :param wires: Measured wires.
        :type wires: Wires
        :return: Scalar probability.
        :rtype: TensorLike
        """
        return self(
            state_prep_op=state_prep_op,
            target_binary_states=target_binary_state,
            wires=wires,
            **kwargs,
        )
