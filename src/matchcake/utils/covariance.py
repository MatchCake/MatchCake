from typing import Optional, Union

import numpy as np
import pennylane as qml
import torch

from ..typing import TensorLike
from .math import complex_dtype_name_like, convert_and_cast_like, convert_like_and_cast_to
from .torch_utils import infer_complex_dtype, infer_real_dtype

# |d| below this snaps to the zero-displacement (basis-state) path: b0 = cov @ d / m divides by
# m -> 0 as |d| -> 0, so the ancilla block is ill-conditioned below ~1e-6.
DISPLACEMENT_TOL = 1e-6

# Lambda_occ: covariance of the occupied configuration |1>_j |1>_k;
# (Lambda_occ)_{01} = (Lambda_occ)_{23} = +1.
_OCCUPIED_BLOCK = torch.tensor([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]], dtype=torch.float64)

# Reference precision at which the degeneracy threshold below was originally calibrated.
_REFERENCE_EPS = float(torch.finfo(torch.complex128).eps)

# Exponent relating the threshold to the working epsilon. The relative error of an
# overlap-normalized quantity was measured against a higher-precision reference over 375 branch
# pairs spanning squared overlaps 5e-3 to 0.8, and obeys ``err * overlap = eps`` to within a factor
# of 3 (median ratio 0.98, max 2.7). Splitting that budget evenly gives a threshold of
# ``eps ** (1/4)`` relative to the reference precision and a worst-case error of ``eps ** (3/4)`` at
# the threshold, which reproduces the calibrated complex128 pair (1e-4, ~1e-12).
# Pinned by ``TestCovariance.test_the_degeneracy_threshold_bounds_the_conditioning_error``.
_TOL_EPS_EXPONENT = 0.25

# Branch pairs with |<phi_a|phi_b>|^2 = 2^{-D/2} |Pf(Lambda_a + Lambda_b)| below this are treated
# as orthogonal: the overlap-normalized observables become ill-conditioned there (0 x inf), so the
# state is flagged degenerate. At this threshold the branch-path conditioning error is still ~1e-12.
#
# This is the ``complex128`` value. It is kept as a module constant for callers that work at that
# precision unconditionally; anything that honours the working dtype must call
# :func:`degenerate_overlap_tol` instead.
DEGENERATE_OVERLAP_TOL = 1e-4


def degenerate_overlap_tol(reference: Union[torch.dtype, TensorLike]) -> float:
    """Squared-overlap threshold below which a branch pair counts as orthogonal.

    The overlap-normalized branch observables divide by the pairwise overlap, so their error grows
    like ``eps / overlap`` as the pair approaches orthogonality. A threshold therefore trades two
    quantities against each other: lowering it keeps more pairs on the fast path, raising it keeps
    the worst-case error at the threshold smaller. Holding the ratio fixed across precisions gives
    ``tol(dtype) = DEGENERATE_OVERLAP_TOL * (eps(dtype) / eps(complex128)) ** 0.25``, which returns
    exactly :data:`DEGENERATE_OVERLAP_TOL` at ``complex128`` and roughly ``1.5e-2`` at
    ``complex64``.

    The bare :data:`DEGENERATE_OVERLAP_TOL` is dtype-blind and is sound at ``complex128`` only.
    Prefer this function wherever the working precision is not fixed.

    :param reference: A torch dtype, or any tensor whose complex working precision is inferred.
    :return: Threshold on the squared pairwise overlap, in ``(0, 1)``.
    :rtype: float
    :raises ValueError: if the scaled threshold reaches 1, which would flag every pair as
        degenerate and leave the certificate vacuous.
    """
    dtype = reference if isinstance(reference, torch.dtype) else infer_complex_dtype(reference)
    tol = DEGENERATE_OVERLAP_TOL * (float(torch.finfo(dtype).eps) / _REFERENCE_EPS) ** _TOL_EPS_EXPONENT
    if tol >= 1.0:
        raise ValueError(
            f"no sound degeneracy threshold exists at {dtype}: the scaled threshold is {tol:.3g}, "
            f"so every branch pair would be flagged degenerate. Work at a higher precision."
        )
    return tol


def lift_sptm(sptm: TensorLike) -> TensorLike:
    r"""Lift a ``(..., 2n, 2n)`` matchgate SPTM to the even ``(2n+2)`` frame ``sptm (+) I_2``.

    The two ancilla modes (the ``b0`` mode at index ``2n`` and the parity marker at ``2n+1``) are
    inert under matchgate evolution, so they ride along as an identity block:

    .. math::
        \widetilde{Q} = \begin{bmatrix} Q & 0 \\ 0 & I_2 \end{bmatrix}.

    This is the even ``(2n+2)`` analogue of
    :func:`~matchcake.devices.expval_strategies.m_pfaffian._extended_covariance.sptm_lift`, which
    appends a single parity row/col for the odd ``(2n+1)`` encoding. The two encodings serve
    different formalisms and are deliberately not one function with a mode flag.

    Supports arbitrary leading batch dimensions.

    :param sptm: Orthogonal single-particle transition matrix of shape ``(..., 2n, 2n)``.
    :return: Lifted SPTM of shape ``(..., 2n+2, 2n+2)``, on the backend/dtype of ``sptm``.
    :rtype: TensorLike
    """
    real_dtype = infer_real_dtype(sptm)
    sptm_real = torch.as_tensor(qml.math.real(sptm), dtype=real_dtype)
    batch = sptm_real.shape[:-2]
    two_n = sptm_real.shape[-1]

    lifted = torch.zeros(*batch, two_n + 2, two_n + 2, dtype=real_dtype, device=sptm_real.device)
    lifted[..., :two_n, :two_n] = sptm_real
    lifted[..., two_n, two_n] = 1.0
    lifted[..., two_n + 1, two_n + 1] = 1.0
    return convert_and_cast_like(lifted, sptm)


def lift_from_product_state(covariance: TensorLike, displacement: TensorLike) -> TensorLike:
    r"""Even ``(2n+2)`` orthogonal lift of a product state from its covariance and displacement.

    Given the physical covariance and displacement of a product state, returns the real orthogonal
    lift ``lifted`` of shape ``(2n+2, 2n+2)`` (``lifted @ lifted = -I``) whose physical block is the
    covariance, with the ``b0`` ancilla mode at index ``2n`` and the parity marker (``= -d``) at
    index ``2n+1``:

    .. math::
        M = \begin{bmatrix} \Lambda & b_0 & -d \\ -b_0^T & 0 & -m \\ d^T & m & 0 \end{bmatrix},
        \quad m^2 = -\frac{d^T \Lambda^2 d}{d^T d}, \quad b_0 = \frac{\Lambda d}{m}.

    When ``|displacement| < DISPLACEMENT_TOL`` the state is a basis state (up to per-qubit phase)
    and the ancilla decouples: ``lifted = covariance (+) [[0, 1], [-1, 0]]``. This is the
    zero-displacement special case, so a single code path serves basis and product inputs.

    :param covariance: Real antisymmetric covariance matrix of shape ``(2n, 2n)``.
    :param displacement: Real displacement vector ``d[mu] = <c_mu>`` of shape ``(2n,)``.
    :return: Real orthogonal lift of shape ``(2n+2, 2n+2)``, on the backend/dtype of ``covariance``.
    :rtype: TensorLike
    :raises NotImplementedError: if ``covariance`` carries leading batch dimensions.
    :raises ValueError: if the ancilla scale collapses to zero, which means the covariance and
        displacement do not describe a product state with this displacement.
    :raises AssertionError: if ``rank(I + covariance^2) > 2`` (the input is not a product state, so
        a single ancilla mode is insufficient).
    """
    real_dtype = infer_real_dtype(covariance)
    covariance_real = torch.as_tensor(qml.math.real(covariance), dtype=real_dtype)
    displacement_real = torch.as_tensor(qml.math.real(displacement), dtype=real_dtype)
    if covariance_real.ndim != 2:
        raise NotImplementedError(
            "lift_from_product_state expects an unbatched (2n, 2n) covariance; the initial product "
            "state is lifted once and then propagated, so batched parameters enter via the SPTMs."
        )
    dim = covariance_real.shape[-1]
    lifted = torch.zeros(dim + 2, dim + 2, dtype=real_dtype, device=covariance_real.device)
    lifted[:dim, :dim] = covariance_real

    norm_sq = torch.dot(displacement_real, displacement_real)
    if float(norm_sq) < DISPLACEMENT_TOL**2:
        lifted[dim, dim + 1], lifted[dim + 1, dim] = 1.0, -1.0
        return convert_and_cast_like(lifted, covariance)

    # Product-state invariant: a single ancilla mode always suffices (verified for n up to 6).
    rank = int(np.linalg.matrix_rank(qml.math.toarray(covariance_real @ covariance_real) + np.eye(dim), tol=1e-9))
    assert rank <= 2, (
        f"lift_from_product_state: rank(I + covariance^2) = {rank} > 2; the input is not a product "
        f"state, so a single ancilla mode is insufficient."
    )

    scale_sq = -(displacement_real @ (covariance_real @ (covariance_real @ displacement_real))) / norm_sq
    if float(scale_sq) <= 0.0:
        # b0 = Lambda d / m below divides by m, so a collapsed scale would return a silently NaN
        # lift. Genuine product states keep m well away from zero (measured minimum ~1.6e-5 over
        # random ProductState inputs); reaching here means the inputs are not a product state.
        raise ValueError(
            "lift_from_product_state: the ancilla scale m^2 = -d^T Lambda^2 d / d^T d collapsed to "
            f"{float(scale_sq):.3g}, so the covariance and displacement do not describe a product "
            "state with this displacement."
        )
    scale = torch.sqrt(scale_sq)
    ancilla_coupling = (covariance_real @ displacement_real) / scale
    lifted[:dim, dim] = ancilla_coupling
    lifted[dim, :dim] = -ancilla_coupling
    lifted[:dim, dim + 1] = -displacement_real
    lifted[dim + 1, :dim] = displacement_real
    lifted[dim, dim + 1] = scale
    lifted[dim + 1, dim] = -scale
    return convert_and_cast_like(lifted, covariance)


def transition_cov(cov_a: TensorLike, cov_b: TensorLike, marker: Optional[int] = None) -> TensorLike:
    r"""Complex transition covariance between branches, fully batched.

    Projector form, valid for the basis path (``D = 2n``) and the even lift (``D = 2n+2``) because
    both covariances are real orthogonal and even-dimensional:

    .. math::
        P_b = \tfrac{I + i M_b}{2}, \quad \bar P_a = \tfrac{I - i M_a}{2}, \quad
        \bar P_b = \tfrac{I - i M_b}{2}, \qquad
        \Gamma = i\,\bigl(2\,\bar P_a (\bar P_a + P_b)^{-1} \bar P_b\bigr).

    On the lifted path the marker row/col is rescaled by ``-i`` so it carries ``i d`` and the
    uniform Wick rule holds across both parities; on the basis path pass ``marker=None``.
    The result is antisymmetrized (``0.5 (Gamma - Gamma^T)``): the raw projector form has a nonzero
    imaginary diagonal that Pfaffian routines must not see; antisymmetrizing zeroes it while leaving
    every off-diagonal entry (hence every Pfaffian) unchanged, and gives ``transition_cov(M, M) = M``.

    ``cov_a`` and ``cov_b`` broadcast against each other, so passing ``cov[:, None]`` and
    ``cov[None, :]`` yields the full ``(chi, chi, ..., D, D)`` pair tensor in one call.

    :param cov_a: Real orthogonal covariance(s) of shape ``(..., D, D)``.
    :param cov_b: Real orthogonal covariance(s) of shape ``(..., D, D)``.
    :param marker: Index of the parity marker row/col to rescale by ``-i`` (lifted path), or
        ``None``.
    :return: Complex antisymmetric transition covariance of the broadcast shape ``(..., D, D)``.
    :rtype: TensorLike
    """
    complex_dtype = infer_complex_dtype(cov_a)
    matrix_a = torch.as_tensor(
        qml.math.toarray(cov_a) if not isinstance(cov_a, torch.Tensor) else cov_a, dtype=complex_dtype
    )
    matrix_b = torch.as_tensor(
        qml.math.toarray(cov_b) if not isinstance(cov_b, torch.Tensor) else cov_b, dtype=complex_dtype
    )
    dim = matrix_a.shape[-1]
    eye = torch.eye(dim, dtype=complex_dtype, device=matrix_a.device)

    proj_b = (eye + 1j * matrix_b) / 2
    proj_bar_a = (eye - 1j * matrix_a) / 2
    proj_bar_b = (eye - 1j * matrix_b) / 2
    # (proj_bar_a + proj_b) is singular exactly when the two branches are orthogonal. That case
    # only arises during branching, where the (finite, meaningless) pseudo-inverse result is
    # multiplied by the vanishing overlap weight and the state is flagged degenerate; pinv matches
    # inv whenever the pair is nonsingular.
    gamma = 1j * (2 * proj_bar_a @ torch.linalg.pinv(proj_bar_a + proj_b) @ proj_bar_b)

    if marker is not None:
        gamma[..., marker, :] = gamma[..., marker, :] * (-1j)
        gamma[..., :, marker] = gamma[..., :, marker] * (-1j)

    gamma = 0.5 * (gamma - gamma.transpose(-1, -2))
    # Cast to the complex working dtype rather than to ``cov_a``: the inputs are real orthogonal
    # covariances, so casting back to them would strip the imaginary part that is the whole result.
    return convert_like_and_cast_to(gamma, cov_a, dtype=complex_dtype_name_like(matrix_a))


def condition_occupied(covariance: TensorLike, j: int, k: int) -> TensorLike:
    r"""Covariance of branches with modes ``j, k`` projected to occupied, batched.

    With ``S4 = {2j, 2j+1, 2k, 2k+1}`` and ``R`` the rest (every Majorana mode not pinned; the two
    ancilla rows/cols stay in ``R`` on the lifted path), the block split ``cov = [[A, B], [-B^T, C]]``
    gives, with ``Lambda_occ`` the covariance of the occupied configuration ``|1>_j |1>_k``,

    .. math::
        \Lambda'|_{S4} = \Lambda_{\mathrm{occ}}, \quad \Lambda'|_{S4, R} = 0, \quad
        \Lambda'|_R = C + B^T (A + \Lambda_{\mathrm{occ}})^{-1} B.

    The last term is the fermionic Schur-complement back-action of the projection (the analogue of
    classical Gaussian conditioning).

    Acts on any leading batch (e.g. the whole ``(chi, ..., D, D)`` branch tensor) at once.

    :param covariance: Real antisymmetric covariance(s) of shape ``(..., D, D)``.
    :param j: First qubit index.
    :param k: Second qubit index, distinct from ``j``.
    :return: Conditioned covariance(s) of shape ``(..., D, D)``, on the backend of ``covariance``
        and on its dtype, except that an integer input is promoted to ``float64`` (see below).
    :rtype: TensorLike
    :raises ValueError: if ``covariance`` is complex, if ``j == k``, or if either qubit is out of
        range for the covariance size.
    """
    covariance_t = torch.as_tensor(
        qml.math.toarray(covariance) if not isinstance(covariance, torch.Tensor) else covariance
    )
    if torch.is_complex(covariance_t):
        raise ValueError(
            "condition_occupied expects a real covariance; the Schur complement below is real, so "
            "a complex input would have its imaginary part silently discarded."
        )
    # An integer covariance cannot hold the Schur back-action, which is generally non-integral, so
    # it is promoted here and the promoted precision is kept in the returned value.
    integer_input = not torch.is_floating_point(covariance_t)
    if integer_input:
        covariance_t = covariance_t.to(torch.float64)
    dim = covariance_t.shape[-1]
    # Without this, j == k builds a four-element pinned list holding two duplicate mode pairs. The
    # scatter write below happens to stay self-consistent, so the call returns a structurally
    # antisymmetric but physically meaningless covariance instead of failing.
    if j == k:
        raise ValueError(f"condition_occupied expects two distinct qubits, got j = k = {j}.")
    if min(j, k) < 0 or 2 * max(j, k) + 1 >= dim:
        raise ValueError(f"qubits j = {j} and k = {k} are out of range for a covariance of size {dim}.")
    occupied_modes = [2 * j, 2 * j + 1, 2 * k, 2 * k + 1]
    rest_modes = [mode for mode in range(dim) if mode not in occupied_modes]
    occupied_index = torch.as_tensor(occupied_modes, dtype=torch.long)
    rest_index = torch.as_tensor(rest_modes, dtype=torch.long)
    occupied_block = _OCCUPIED_BLOCK.to(dtype=covariance_t.dtype, device=covariance_t.device)

    block_pinned = covariance_t.index_select(-2, occupied_index).index_select(-1, occupied_index)
    block_cross = covariance_t.index_select(-2, occupied_index).index_select(-1, rest_index)
    block_rest = covariance_t.index_select(-2, rest_index).index_select(-1, rest_index)
    # block_pinned + occupied_block is invertible whenever the projected branch has nonzero weight;
    # it is singular exactly when the branch vanishes (e.g. swapping an ancilla pinned to |0>, q = 0),
    # in which case the conditioned covariance is irrelevant because the branch is pruned right after.
    # The pseudo-inverse gives a finite (pruned) result instead of raising.
    conditioned_rest = (
        block_rest + block_cross.transpose(-1, -2) @ torch.linalg.pinv(block_pinned + occupied_block) @ block_cross
    )

    conditioned = torch.zeros_like(covariance_t)
    conditioned[..., occupied_index[:, None], occupied_index[None, :]] = occupied_block
    conditioned[..., rest_index[:, None], rest_index[None, :]] = conditioned_rest
    if integer_input:
        return convert_like_and_cast_to(conditioned, covariance, dtype="float64")
    return convert_and_cast_like(conditioned, covariance)


def basis_state_covariance_block(
    bits: TensorLike,
    dim: int,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Physical-block basis-state covariance ``Lambda_y`` of shape ``(dim, dim)``.

    ``(Lambda_y)_{2k, 2k+1} = 2 y_k - 1 = -(-1)^{y_k}``, which is the convention
    :func:`~matchcake.utils._pfaffian_family.sample_outcomes` draws against. ``dim`` may exceed
    ``2 * len(bits)``, in which case the trailing rows and columns stay zero; this is what lets a
    physical outcome block be added to a lifted ``(2n+2)`` covariance.

    :param bits: Outcome bits of length ``n``.
    :param dim: Size of the (square) covariance block, ``>= 2n``.
    :param dtype: Working dtype of the returned tensor. Defaults to ``torch.float64``.
    :param device: Device of the returned tensor. Defaults to ``None`` (the current default
        device).
    :return: Antisymmetric basis-state covariance of shape ``(dim, dim)``.
    :rtype: torch.Tensor
    """
    bits = np.asarray(qml.math.toarray(bits)).astype(int).reshape(-1)
    n_qubits = len(bits)
    lambda_y = torch.zeros(dim, dim, dtype=dtype, device=device)
    values = torch.as_tensor(2 * bits - 1, dtype=dtype, device=device)
    qubit = torch.arange(n_qubits)
    lambda_y[2 * qubit, 2 * qubit + 1] = values
    lambda_y[2 * qubit + 1, 2 * qubit] = -values
    return lambda_y
