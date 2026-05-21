from typing import Sequence

import numpy as np
import pennylane as qml
import torch
from pennylane.operation import Operator, StatePrepBase, TermsUndefinedError
from pennylane.ops.qubit.observables import BasisStateProjector
from pennylane.pauli import pauli_word_to_string

from ....typing import TensorLike
from ....utils import JordanWigner
from ....utils._pfaffian import sector_pfaffian_features
from ....utils.math import convert_and_cast_like
from ....operations.state_preparation.product_state import ProductState
from ..expval_strategy import ExpvalStrategy


def extend_majorana_indices(
    mu: np.ndarray,
    alpha: complex,
    parity_index: int,
) -> tuple[np.ndarray, complex]:
    """Return (S_tilde, alpha_tilde) per the parity rule.

    For m even:  S_tilde = mu,                          alpha_tilde = alpha.
    For m odd:   S_tilde = concat(mu, [parity_index]),
                 alpha_tilde = alpha * i^m * (-1)^{m(m-1)/2}.

    Since parity_index = 2n is the largest index, appending it keeps S_tilde sorted.
    """
    m = len(mu)
    if m % 2 == 0:
        return mu, alpha
    extra_phase = (1j) ** m * (-1) ** ((m * (m - 1)) // 2)
    return np.concatenate([mu, [parity_index]]), alpha * extra_phase


class MPfaffianExpvalStrategy(ExpvalStrategy):
    """Compute <P> for arbitrary Pauli observables via the extended-encoding m-Pfaffian.

    Uses the extended (2n+1)-Majorana algebra so that arbitrary qubit product states
    are supported as initial states (not just computational basis states).

    The ``extended_covariance_matrix`` kwarg must be the EXTENDED matrix tilde_Lambda
    of shape (..., 2n+1, 2n+1), laid out with the parity index at position 2n:

        tilde_Lambda = [[ Lambda,  d   ],
                        [ -d^T,    0   ]]

    where Lambda is the 2n x 2n standard covariance matrix and d[mu] = <c_mu> is the
    displacement vector for the initial product state.

    The expectation value formula for each Pauli term P = coeff * pauli_op is:

        <P> = Re( coeff * alpha_tilde * (-i)^{|S_tilde|/2} * Pf(tilde_Lambda|_{S_tilde}) )

    Mid-circuit gates must all be matchgates (matchgate evolution preserves the
    lift's identity-block structure on the parity index). This is a device-level
    invariant enforced upstream.
    """

    NAME = "MPfaffianExpvalStrategy"

    @staticmethod
    def _to_hamiltonian(observable: Operator) -> qml.Hamiltonian:
        try:
            terms = observable.terms()
        except TermsUndefinedError:
            terms = (torch.ones(1),), [observable]
        return qml.Hamiltonian(*terms)

    def __call__(
        self,
        state_prep_op: StatePrepBase,
        observable: Operator,
        *,
        extended_covariance_matrix: TensorLike,
        **kwargs,
    ) -> TensorLike:
        if not self.can_execute(state_prep_op, observable):
            raise ValueError(
                f"Cannot execute {self.NAME} strategy for observable {observable} "
                f"with state_prep_op {state_prep_op}."
            )

        tilde_Lambda = extended_covariance_matrix  # (..., 2n+1, 2n+1)
        n_total = tilde_Lambda.shape[-1]           # 2n + 1
        parity_index = n_total - 1                 # 2n

        hamiltonian = self._to_hamiltonian(observable)
        h_coeffs, h_ops = hamiltonian.terms()

        device_wires = sorted(state_prep_op.wires.tolist())
        n_qubits = parity_index // 2
        jw = JordanWigner(n_qubits)

        global_wire_map = {w: i for i, w in enumerate(device_wires)}

        terms_by_sector: dict[int, list[tuple[np.ndarray, complex, int]]] = {}
        for term_idx, (coeff, op) in enumerate(zip(h_coeffs, h_ops)):
            pauli_str = pauli_word_to_string(op, wire_map=global_wire_map)
            mu, alpha = jw.pauli_to_majorana(pauli_str, device_wires)
            S_tilde, alpha_tilde = extend_majorana_indices(mu, alpha, parity_index)
            sector = len(S_tilde)
            terms_by_sector.setdefault(sector, []).append((S_tilde, alpha_tilde, term_idx))

        pf_values: dict[int, TensorLike] = {}
        for sector, items in terms_by_sector.items():
            S_list = np.stack([item[0] for item in items])  # (K, sector)
            if sector == 0:
                pf_values[sector] = torch.ones(len(items), dtype=torch.float64)
            elif sector == 2:
                i_idx = S_list[:, 0]
                j_idx = S_list[:, 1]
                tl = torch.as_tensor(
                    np.array(tilde_Lambda) if not isinstance(tilde_Lambda, torch.Tensor) else tilde_Lambda,
                    dtype=torch.float64,
                )
                pf_values[sector] = tl[..., i_idx, j_idx]  # (..., K)
            else:
                pf_values[sector] = sector_pfaffian_features(tilde_Lambda, S_list)  # (..., K)

        total_re = torch.tensor(0.0, dtype=torch.float64)
        for sector, items in terms_by_sector.items():
            wick_phase = (-1j) ** (sector // 2)
            pfs = pf_values[sector]   # (..., K)
            for k, (S_tilde, alpha_tilde, term_idx) in enumerate(items):
                coeff = complex(np.array(h_coeffs[term_idx]).item())
                pf_k = pfs[..., k]   # (...)
                scalar = float(np.real(coeff * complex(alpha_tilde) * wick_phase))
                total_re = total_re + scalar * pf_k

        return convert_and_cast_like(total_re, extended_covariance_matrix)

    def can_execute(
        self,
        state_prep_op: StatePrepBase,
        observable: Operator,
    ) -> bool:
        """Return True iff this strategy can compute <observable>.

        Conditions:
          1. state_prep_op is a BasisState or a ProductState.
          2. observable decomposes into a sum of Pauli strings.
        """
        from pennylane import BasisState
        if not isinstance(state_prep_op, (BasisState, ProductState)):
            return False
        try:
            hamiltonian = self._to_hamiltonian(observable)
            _, ops = hamiltonian.terms()
            for op in ops:
                _ = pauli_word_to_string(op, wire_map={w: i for i, w in enumerate(op.wires)})
        except Exception:
            return False
        return True
