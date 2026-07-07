from typing import List, Optional, Tuple

import numpy as np
import pennylane as qml
import torch
from pennylane.operation import Operator

from ...typing import TensorLike
from ...utils._pfaffian import signed_pfaffian_complex
from ...utils.math import convert_and_cast_like
from ...utils.torch_utils import infer_complex_dtype
from .majorana_term_groups import MajoranaTermGroups


def transition_cov(cov_a: TensorLike, cov_b: TensorLike, marker: Optional[int] = None) -> TensorLike:
    r"""Complex transition covariance between branches (swap_injection_theory.md eq 23 / 9'), fully batched.

    Projector form, valid for the basis path (``D = 2n``) and the even lift (``D = 2n+2``) because
    both covariances are real orthogonal and even-dimensional:

    .. math::
        P_b = \tfrac{I + i M_b}{2}, \quad \bar P_a = \tfrac{I - i M_a}{2}, \quad
        \bar P_b = \tfrac{I - i M_b}{2}, \qquad
        \Gamma = i\,\bigl(2\,\bar P_a (\bar P_a + P_b)^{-1} \bar P_b\bigr).

    On the lifted path the marker row/col is rescaled by ``-i`` so it carries ``i d`` and the
    uniform Wick rule (eq 24) holds across both parities; on the basis path pass ``marker=None``.
    The result is antisymmetrized (``0.5 (Gamma - Gamma^T)``): the raw projector form has a nonzero
    imaginary diagonal that Pfaffian routines must not see; antisymmetrizing zeroes it while leaving
    every off-diagonal entry (hence every Pfaffian) unchanged, and gives ``transition_cov(M, M) = M``.

    ``cov_a`` and ``cov_b`` broadcast against each other, so passing ``cov[:, None]`` and
    ``cov[None, :]`` yields the full ``(chi, chi, ..., D, D)`` pair tensor in one call.

    :param cov_a: Real orthogonal covariance(s) of shape ``(..., D, D)``.
    :param cov_b: Real orthogonal covariance(s) of shape ``(..., D, D)``.
    :param marker: Index of the parity marker row/col to rescale by ``-i`` (lifted path), or ``None``.
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
    # multiplied by the vanishing overlap weight and the state is flagged degenerate (observables
    # then reroute to the string engine); pinv matches inv whenever the pair is nonsingular.
    gamma = 1j * (2 * proj_bar_a @ torch.linalg.pinv(proj_bar_a + proj_b) @ proj_bar_b)

    if marker is not None:
        gamma[..., marker, :] = gamma[..., marker, :] * (-1j)
        gamma[..., :, marker] = gamma[..., :, marker] * (-1j)

    gamma = 0.5 * (gamma - gamma.transpose(-1, -2))
    return convert_and_cast_like(gamma, matrix_a)


def basis_state_probability(
    branch_covariances: TensorLike,
    weights: TensorLike,
    target_state: TensorLike,
    measured_qubits: Optional[List[int]] = None,
    pair_mask: Optional[np.ndarray] = None,
) -> TensorLike:
    r"""Outcome probability ``p(y)`` from branch data (swap_injection_theory.md eq 15), vectorized over branch pairs.

    .. math::
        p(y) = \frac{\mathrm{Pf}(\Lambda_y)}{2^k}
               \sum_{a, b} W_{ab}\, \mathrm{Pf}\bigl(\Gamma_{ab}|_{\mathrm{meas}} + \Lambda_y\bigr),

    using the physical block of every transition covariance (the parity-even projector never appends
    the marker). For a marginal over ``k`` measured qubits, both ``Gamma`` and ``Lambda_y`` are
    restricted to the ``2k`` Majorana modes of those qubits. ``Pf(Lambda_y) = prod_k (2 y_k - 1)``.

    When ``pair_mask`` is given (the :attr:`SwapBranchState.string_pair_mask` of a degenerate
    state), the masked pairs are excluded: their weights are zeroed and their transition
    covariances replaced by a benign nonsingular placeholder, so neither the value nor the gradient
    ever touches the ill-defined ``0 x inf`` pairs. The caller adds those pairs back overlap-free.

    :param branch_covariances: Real branch covariance tensor of shape ``(chi, ..., D, D)``.
    :param weights: Complex Hermitian weight matrix of shape ``(chi, chi)`` (or ``(chi, chi, ...)``).
    :param target_state: Outcome bits of the measured qubits, an array of ``k`` bits.
    :param measured_qubits: Qubit indices the bits refer to. Defaults to ``range(k)``.
    :param pair_mask: Boolean ``(chi, chi)`` mask of branch pairs to exclude, or ``None``.
    :return: Real probability ``p(y)`` (scalar or ``(...)``).
    :rtype: TensorLike
    """
    bits = np.asarray(qml.math.toarray(target_state)).astype(int).reshape(-1)
    n_measured = len(bits)
    if measured_qubits is None:
        measured_qubits = list(range(n_measured))
    measured_modes = [2 * qubit + offset for qubit in measured_qubits for offset in (0, 1)]

    complex_dtype = infer_complex_dtype(branch_covariances[0])
    device = torch.as_tensor(qml.math.toarray(branch_covariances[0])).device
    weights = _weights_to_tensor(weights, complex_dtype, device)
    lambda_y = _build_lambda_y_block(bits, 2 * n_measured, complex_dtype, device)
    pf_lambda_y = float(np.prod(2 * bits - 1))  # Pf(Lambda_y), real
    mode_index = torch.as_tensor(measured_modes, dtype=torch.long)

    gamma = transition_cov(branch_covariances[:, None], branch_covariances[None, :])  # (chi, chi, ..., D, D)
    gamma, weights = _exclude_masked_pairs(gamma, weights, pair_mask)
    gamma_measured = gamma.index_select(-2, mode_index).index_select(-1, mode_index)
    pfaffians = signed_pfaffian_complex(gamma_measured + lambda_y)  # (chi, chi, ...)

    total = qml.math.sum(weights * pfaffians, axis=(0, 1)) * (2.0**-n_measured) * pf_lambda_y
    probability = qml.math.real(total)
    return convert_and_cast_like(probability, qml.math.real(branch_covariances[0]))


def hamiltonian_expval(
    branch_covariances: TensorLike,
    weights: TensorLike,
    observable: Operator,
    wires: List,
    marker: Optional[int] = None,
    pair_mask: Optional[np.ndarray] = None,
    pfaffian_chunk_size: Optional[int] = None,
) -> TensorLike:
    r"""Expectation value ``<H>`` of a Pauli-sum observable from branch data (swap_injection_theory.md eq 13 / 24).

    For each Pauli term ``P = coeff * kappa * c_{S}`` (with ``(S, kappa)`` from the Jordan-Wigner
    map) the uniform Wick rule gives ``sum_{a,b} W_{ab} coeff kappa i^{-t} Pf(Gamma_{ab}|_{S'})`` with
    ``(S', t) = (S, |S|/2)`` for even ``|S|`` and ``(S ∪ {marker}, (|S|+1)/2)`` for odd ``|S|``. On the
    basis path (``marker=None``) odd-``|S|`` terms vanish by parity superselection.

    The transition covariance of every branch pair is computed once (vectorized), then each term is a
    single batched Pfaffian over the pair grid; only the (necessary) sum over Hamiltonian terms is a
    Python loop.

    Only :func:`JordanWigner.pauli_to_majorana` is reused, not
    ``MPfaffianExpvalStrategy.extend_majorana_indices`` / ``get_payload`` (those bake the odd-rank
    parity phase and the real part for the ``(2n+1)`` raw-displacement encoding; the even lift carries
    ``i d`` in the marker column, so the branch path uses the plain ``i^{-t}`` and takes the real part
    only after the full complex branch sum).

    :param branch_covariances: Real branch covariance tensor of shape ``(chi, ..., D, D)``.
    :param weights: Complex Hermitian weight matrix of shape ``(chi, chi)`` (or ``(chi, chi, ...)``).
    :param observable: A Pauli-decomposable observable (exposing ``observable.terms()``).
    :param wires: Device wire labels in qubit order.
    :param marker: Parity-marker index ``D - 1`` on the lifted path, or ``None`` on the basis path.
    :param pair_mask: Boolean ``(chi, chi)`` mask of branch pairs to exclude (weights zeroed,
        transition covariances replaced by a benign placeholder, exactly as in
        :func:`basis_state_probability`), or ``None``.
    :param pfaffian_chunk_size: Max number of matrices reduced per batched Pfaffian call, forwarded as
        ``chunk_size`` to bound the reduction's memory (requires the chunked ``pfaffian``; the device passes its
        own ``pfaffian_chunk_size`` here). Defaults to ``None`` (no chunking).
    :return: Real expectation value ``<H>`` (scalar or ``(...)``).
    :rtype: TensorLike
    """
    wires = list(wires)

    complex_dtype = infer_complex_dtype(branch_covariances[0])
    device = torch.as_tensor(qml.math.toarray(branch_covariances[0])).device
    weights = _weights_to_tensor(weights, complex_dtype, device)

    gammas = transition_cov(
        branch_covariances[:, None], branch_covariances[None, :], marker=marker
    )  # (chi,chi,...,D,D)
    gammas, weights = _exclude_masked_pairs(gammas, weights, pair_mask)

    # Pfaffian calls are batched ACROSS TERMS as well as across branch pairs: the parsed Majorana term structure
    # is cached on the observable (MajoranaTermGroups), and each support-size group rides the Pfaffian batch
    # dimension as one call, so the kernel-dispatch count is the number of distinct support sizes per evaluation.
    term_groups = MajoranaTermGroups.from_observable(observable, wires, marker)
    total = 0.0 + 0.0j
    if term_groups.identity_weight != 0:
        total = total + term_groups.identity_weight * qml.math.sum(weights, axis=(0, 1))  # all-ones Pfaffian grid
    for size, index_tensor, weight_values in term_groups.groups:
        weight_tensor = weight_values.to(dtype=complex_dtype, device=device)
        rows = index_tensor[:, :, None].expand(-1, size, size)
        cols = index_tensor[:, None, :].expand(-1, size, size)
        submatrices = gammas[..., rows, cols]  # (chi, chi, ..., n_terms, size, size)
        pfaffian_kwargs = {} if pfaffian_chunk_size is None else {"chunk_size": pfaffian_chunk_size}
        pfaffians = signed_pfaffian_complex(submatrices, **pfaffian_kwargs)  # (chi, chi, ..., n_terms)
        term_sum = (pfaffians * weight_tensor).sum(-1)  # (chi, chi, ...)
        total = total + qml.math.sum(weights * term_sum, axis=(0, 1))

    expectation = qml.math.real(total)
    return convert_and_cast_like(expectation, qml.math.real(branch_covariances[0]))


def _weights_to_tensor(weights: TensorLike, dtype: torch.dtype, device) -> torch.Tensor:
    """Convert the weight matrix to a torch tensor without detaching it from the autograd graph.

    The weights carry the circuit parameters' gradient through the cross occupations of every
    branching, so converting through ``qml.math.toarray`` (which detaches) would silently drop
    ``dW / d theta`` from every branch-path gradient.

    :param weights: Complex weight matrix of shape ``(chi, chi)`` (or ``(chi, chi, ...)``).
    :param dtype: Complex working dtype.
    :param device: Torch device of the branch covariances.
    :return: The weight matrix as a torch tensor on ``device``.
    :rtype: torch.Tensor
    """
    if isinstance(weights, torch.Tensor):
        return weights.to(dtype=dtype, device=device)
    return torch.as_tensor(qml.math.toarray(weights), dtype=dtype, device=device)


def _exclude_masked_pairs(
    gamma: torch.Tensor,
    weights: torch.Tensor,
    pair_mask: Optional[np.ndarray],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Zero the weights and neutralize the transition covariances of the masked branch pairs.

    The masked pairs are (near-)orthogonal or of tainted lineage, so their ``gamma`` holds a
    meaningless pseudo-inverse result. Replacing it with the (nonsingular, antisymmetric) all-ones
    basis covariance keeps every downstream Pfaffian and its gradient finite, while the zeroed
    weight removes the pair's contribution; the caller adds the pair back overlap-free.

    :param gamma: Transition covariances of shape ``(chi, chi, ..., D, D)``.
    :param weights: Complex weight matrix of shape ``(chi, chi)`` (or ``(chi, chi, ...)``).
    :param pair_mask: Boolean ``(chi, chi)`` mask of pairs to exclude, or ``None`` for a no-op.
    :return: The neutralized ``(gamma, weights)`` pair.
    :rtype: Tuple[torch.Tensor, torch.Tensor]
    """
    if pair_mask is None or not pair_mask.any():
        return gamma, weights
    dim = gamma.shape[-1]
    benign = _build_lambda_y_block(np.ones(dim // 2, dtype=int), dim, gamma.dtype, gamma.device)
    mask = torch.as_tensor(pair_mask, device=gamma.device)
    gamma_mask = mask.reshape(mask.shape + (1,) * (gamma.ndim - 2))
    gamma = torch.where(gamma_mask, benign, gamma)
    weight_mask = mask.reshape(mask.shape + (1,) * (weights.ndim - 2))
    weights = torch.where(weight_mask, torch.zeros((), dtype=weights.dtype, device=weights.device), weights)
    return gamma, weights


def _build_lambda_y_block(bits: np.ndarray, dim: int, dtype, device) -> torch.Tensor:
    """Physical-block basis-state covariance ``Lambda_y`` of shape ``(dim, dim)``.

    ``(Lambda_y)_{2k, 2k+1} = 2 y_k - 1 = -(-1)^{y_k}``; matches
    :meth:`ProductStateProbabilityStrategy.build_lambda_y` and the oracle ``lam_y``.

    :param bits: Outcome bits of length ``n``.
    :param dim: Size of the (square) covariance block, ``>= 2n``.
    :param dtype: Working dtype of the returned tensor.
    :param device: Device of the returned tensor.
    :return: Antisymmetric basis-state covariance of shape ``(dim, dim)``.
    :rtype: torch.Tensor
    """
    bits = np.asarray(bits).astype(int).reshape(-1)
    n_qubits = len(bits)
    lambda_y = torch.zeros(dim, dim, dtype=dtype, device=device)
    values = torch.as_tensor(2 * bits - 1, dtype=dtype, device=device)
    qubit = torch.arange(n_qubits)
    lambda_y[2 * qubit, 2 * qubit + 1] = values
    lambda_y[2 * qubit + 1, 2 * qubit] = -values
    return lambda_y
