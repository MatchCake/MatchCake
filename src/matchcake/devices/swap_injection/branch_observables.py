from typing import List, Optional

import numpy as np
import pennylane as qml
import torch
from pennylane.operation import Operator, TermsUndefinedError
from pennylane.pauli import pauli_word_to_string

from ...typing import TensorLike
from ...utils import JordanWigner
from ...utils._pfaffian import signed_pfaffian_complex
from ...utils.math import convert_and_cast_like
from ...utils.torch_utils import infer_complex_dtype


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
    gamma = 1j * (2 * proj_bar_a @ torch.linalg.inv(proj_bar_a + proj_b) @ proj_bar_b)

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
) -> TensorLike:
    r"""Outcome probability ``p(y)`` from branch data (swap_injection_theory.md eq 15), vectorized over branch pairs.

    .. math::
        p(y) = \frac{\mathrm{Pf}(\Lambda_y)}{2^k}
               \sum_{a, b} W_{ab}\, \mathrm{Pf}\bigl(\Gamma_{ab}|_{\mathrm{meas}} + \Lambda_y\bigr),

    using the physical block of every transition covariance (the parity-even projector never appends
    the marker). For a marginal over ``k`` measured qubits, both ``Gamma`` and ``Lambda_y`` are
    restricted to the ``2k`` Majorana modes of those qubits. ``Pf(Lambda_y) = prod_k (2 y_k - 1)``.

    :param branch_covariances: Real branch covariance tensor of shape ``(chi, ..., D, D)``.
    :param weights: Complex Hermitian weight matrix of shape ``(chi, chi)`` (or ``(chi, chi, ...)``).
    :param target_state: Outcome bits of the measured qubits, an array of ``k`` bits.
    :param measured_qubits: Qubit indices the bits refer to. Defaults to ``range(k)``.
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
    weights = torch.as_tensor(qml.math.toarray(weights), dtype=complex_dtype, device=device)
    lambda_y = _build_lambda_y_block(bits, 2 * n_measured, complex_dtype, device)
    pf_lambda_y = float(np.prod(2 * bits - 1))  # Pf(Lambda_y), real
    mode_index = torch.as_tensor(measured_modes, dtype=torch.long)

    gamma = transition_cov(branch_covariances[:, None], branch_covariances[None, :])  # (chi, chi, ..., D, D)
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
    :return: Real expectation value ``<H>`` (scalar or ``(...)``).
    :rtype: TensorLike
    """
    wires = list(wires)
    n_qubits = len(wires)
    jordan_wigner = JordanWigner(n_qubits)
    wire_map = {wire: index for index, wire in enumerate(wires)}

    complex_dtype = infer_complex_dtype(branch_covariances[0])
    device = torch.as_tensor(qml.math.toarray(branch_covariances[0])).device
    weights = torch.as_tensor(qml.math.toarray(weights), dtype=complex_dtype, device=device)

    try:
        coefficients, operators = observable.terms()
    except TermsUndefinedError:
        coefficients, operators = [1.0], [observable]  # a bare Pauli word has no terms() decomposition

    gammas = transition_cov(
        branch_covariances[:, None], branch_covariances[None, :], marker=marker
    )  # (chi,chi,...,D,D)

    total = 0.0 + 0.0j
    for coefficient, operator in zip(coefficients, operators):
        pauli_str = pauli_word_to_string(operator, wire_map=wire_map)
        support, kappa = jordan_wigner.pauli_to_majorana(pauli_str, wires)
        rank = len(support)
        if rank % 2 == 0:
            indices, wick_exponent = list(support), rank // 2
        else:
            if marker is None:
                continue  # parity-odd term vanishes on the (parity-even) basis path
            indices, wick_exponent = list(support) + [marker], (rank + 1) // 2
        phase = (1j) ** (-wick_exponent)
        coefficient_value = complex(coefficient.item() if isinstance(coefficient, torch.Tensor) else coefficient)
        if indices:
            index_tensor = torch.as_tensor(indices, dtype=torch.long)
            submatrices = gammas.index_select(-2, index_tensor).index_select(-1, index_tensor)
            pfaffians = signed_pfaffian_complex(submatrices)  # (chi, chi, ...)
        else:
            pfaffians = torch.ones(gammas.shape[:-2], dtype=complex_dtype, device=device)
        total = total + qml.math.sum(weights * (coefficient_value * kappa * phase) * pfaffians, axis=(0, 1))

    expectation = qml.math.real(total)
    return convert_and_cast_like(expectation, qml.math.real(branch_covariances[0]))


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
