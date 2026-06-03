from typing import Literal, Optional

import numpy as np
import pennylane as qml
from pennylane.operation import StatePrepBase
from pennylane.typing import TensorLike
from pennylane.wires import Wires

from ... import utils
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

    TODO: merge batch call and call for ProbabilityStrategy.
    TODO: The shape of lambda_t_w: (2k, 2k) shouldn't be (B, 2k, 2k)?
    TODO: Verify that gradient flow is working
    """

    NAME: str = "ProductState"
    REQUIRES_KWARGS = ["covariance_matrix", "pfaffian_method", "all_wires"]

    @staticmethod
    def build_lambda_y(target_binary_state: TensorLike, n_wires: int) -> np.ndarray:
        r"""Build the basis-state covariance matrix :math:`\Lambda_y`.

        Constructs the real antisymmetric :math:`(2k) \times (2k)` matrix with
        :math:`(\Lambda_y)_{2j,\,2j+1} = -(-1)^{y_j}` for :math:`j = 0, \ldots, k-1`.

        :param target_binary_state: Binary outcome :math:`y` of shape ``(n_wires,)``.
        :type target_binary_state: TensorLike
        :param n_wires: Number of measured wires :math:`k`.
        :type n_wires: int
        :return: Antisymmetric matrix of shape ``(2*n_wires, 2*n_wires)``.
        :rtype: np.ndarray
        """
        lambda_y = np.zeros((2 * n_wires, 2 * n_wires))
        bits = np.asarray(target_binary_state).astype(int)
        for j in range(n_wires):
            val = -((-1) ** bits[j])
            lambda_y[2 * j, 2 * j + 1] = val
            lambda_y[2 * j + 1, 2 * j] = -val
        return lambda_y

    @staticmethod
    def extract_majorana_submatrix(
        covariance_matrix: TensorLike,
        wire_indices: np.ndarray,
    ) -> TensorLike:
        r"""Extract the principal Majorana submatrix for the given wire positions.

        For wire positions :math:`i_0, \ldots, i_{k-1}` in the full wire list, returns the
        :math:`(2k) \times (2k)` principal submatrix indexed by
        :math:`\{2i_j, 2i_j+1 : j = 0,\ldots,k-1\}`.

        :param covariance_matrix: Evolved covariance matrix of shape ``(..., 2n, 2n)``.
        :type covariance_matrix: TensorLike
        :param wire_indices: Positions of the measured wires in the full wire list, shape ``(k,)``.
        :type wire_indices: np.ndarray
        :return: Principal submatrix of shape ``(..., 2k, 2k)``.
        :rtype: TensorLike
        """
        majorana_indices = np.concatenate([[2 * i, 2 * i + 1] for i in wire_indices])
        return covariance_matrix[..., majorana_indices[:, None], majorana_indices[None, :]]

    def __call__(
        self,
        *,
        state_prep_op: StatePrepBase,
        target_binary_state: TensorLike,
        wires: Wires,
        **kwargs,
    ) -> TensorLike:
        r"""Compute the probability of a single basis outcome for a product-state input.

        :param state_prep_op: State preparation operation. Not used directly; the evolved
            covariance matrix must be supplied via the ``covariance_matrix`` keyword argument.
        :type state_prep_op: StatePrepBase
        :param target_binary_state: Binary outcome :math:`y` of shape ``(k,)``.
        :type target_binary_state: TensorLike
        :param wires: Measured wires.
        :type wires: Wires
        :return: Scalar probability.
        :rtype: TensorLike
        """
        self.check_required_kwargs(kwargs)
        if isinstance(wires, int):
            wires = [wires]
        wires = Wires(wires)
        k = len(wires)

        covariance_matrix: TensorLike = kwargs["covariance_matrix"]  # (..., 2n, 2n)
        pfaffian_method: Literal["det", "cuda_det", "PfaffianFDBPf"] = kwargs["pfaffian_method"]
        all_wires = Wires(kwargs["all_wires"])

        wire_indices = all_wires.indices(wires)

        # Extract Lambda(t)|_W: (..., 2k, 2k)
        lambda_t_w = self.extract_majorana_submatrix(covariance_matrix, wire_indices)
        lambda_y = convert_and_cast_like(self.build_lambda_y(target_binary_state, k), covariance_matrix)

        # p(y_W) = (1/2^k) * |Pf(Lambda(t)|_W + Lambda_y_W)|
        combined = lambda_t_w + lambda_y  # (..., 2k, 2k)
        prob = (1.0 / 2**k) * qml.math.real(utils.pfaffian(combined, method=pfaffian_method))
        return convert_and_cast_like(prob, covariance_matrix)

    def batch_call(
        self,
        *,
        state_prep_op: StatePrepBase,
        target_binary_states: TensorLike,
        batch_wires: Optional[Wires] = None,
        **kwargs,
    ) -> TensorLike:
        r"""Compute probabilities for a batch of basis outcomes.

        When all rows of ``batch_wires`` are identical (the common case from the device), the
        computation is vectorized: a single ``(B, 2k, 2k)`` stack of combined matrices is
        assembled and their Pfaffians computed in one call. Otherwise falls back to per-outcome
        calls.

        :param state_prep_op: State preparation operation.
        :type state_prep_op: StatePrepBase
        :param target_binary_states: Batch of binary outcomes of shape ``(B, k)``.
        :type target_binary_states: TensorLike
        :param batch_wires: Wires per outcome of shape ``(B, k)``. Defaults to all device wires
            broadcast to the batch shape.
        :type batch_wires: Optional[Wires]
        :return: Probability vector of shape ``(B,)``.
        :rtype: TensorLike
        """
        self.check_required_kwargs(kwargs)

        covariance_matrix: TensorLike = kwargs["covariance_matrix"]  # (..., 2n, 2n)
        pfaffian_method: Literal["det", "cuda_det", "PfaffianFDBPf"] = kwargs["pfaffian_method"]
        all_wires = Wires(kwargs["all_wires"])
        kwargs.pop("show_progress", False)

        target_binary_states = np.asarray(target_binary_states)  # (B, k)

        if batch_wires is None:
            batch_wires = np.broadcast_to(np.asarray(all_wires), target_binary_states.shape)
        batch_wires = np.asarray(batch_wires)

        n_batch = len(target_binary_states)
        k = target_binary_states.shape[-1]

        same_wires = n_batch == 0 or all(np.array_equal(batch_wires[i], batch_wires[0]) for i in range(1, n_batch))

        if not same_wires:
            probs = qml.math.stack(
                [
                    self(
                        state_prep_op=state_prep_op,
                        target_binary_state=tbs,
                        wires=Wires(wires),
                        **kwargs,
                    )
                    for tbs, wires in zip(target_binary_states, batch_wires)
                ]
            )
            return probs

        wire_indices = all_wires.indices(Wires(batch_wires[0]))

        # Extract Lambda(t)|_W once: (..., 2k, 2k)
        lambda_t_w = self.extract_majorana_submatrix(covariance_matrix, wire_indices)

        # Batch of Lambda_y matrices: (B, 2k, 2k)
        lambda_y_batch = np.stack([self.build_lambda_y(tbs, k) for tbs in target_binary_states])
        lambda_y_batch = convert_and_cast_like(lambda_y_batch, covariance_matrix)

        cov_ndim = len(qml.math.shape(covariance_matrix))
        if cov_ndim > 2:
            # Device batch dims present: fall back to per-outcome calls
            probs = qml.math.stack(
                [
                    self(
                        state_prep_op=state_prep_op,
                        target_binary_state=tbs,
                        wires=Wires(batch_wires[0]),
                        **kwargs,
                    )
                    for tbs in target_binary_states
                ]
            )
            return probs

        # lambda_t_w: (2k, 2k),  lambda_y_batch: (B, 2k, 2k)
        # Broadcasting gives combined: (B, 2k, 2k)
        combined = lambda_t_w + lambda_y_batch
        prob = (1.0 / 2**k) * qml.math.real(utils.pfaffian(combined, method=pfaffian_method))
        return convert_and_cast_like(prob, covariance_matrix)
