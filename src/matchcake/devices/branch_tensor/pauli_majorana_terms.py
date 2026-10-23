from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from pennylane.operation import Operator

from ...utils import JordanWigner, live_pauli_sentence


def _as_complex(value: Union[complex, np.ndarray, torch.Tensor]) -> complex:
    """Return a scalar coefficient as a python ``complex``, unwrapping a one-element array or tensor.

    :param value: A python, numpy, pennylane.numpy or torch scalar, or a one-element array.
    :return: The value as a python ``complex``.
    :rtype: complex
    """
    if isinstance(value, (torch.Tensor, np.ndarray)):
        value = value.item()
    return complex(value)


class PauliMajoranaTerms:
    r"""Decomposition of a Pauli-sum observable into sorted Majorana monomials.

    Maps a PennyLane ``Operator`` to index-aligned ``(supports, coefficients)`` lists. Each support is
    a sorted, duplicate-free tuple of Majorana mode indices; each coefficient is the term's Pauli
    coefficient times the Jordan-Wigner phase :math:`\kappa`.

    The uniform Wick phase :math:`i^{-w/2}` is not folded into the coefficients: it depends only on
    the support size and is applied during the reduction.

    A Pauli word whose Majorana support has odd weight cannot be evaluated by the two-state Wick rule
    directly. In the parity-purified :math:`(2n+2)` frame, passing ``marker`` appends that index to the
    support and makes it even. Without a marker such a term vanishes by parity superselection and is
    dropped from the decomposition.
    """

    @staticmethod
    def marginal_projector_terms(
        target_bits: Sequence[int],
        measured_qubits: Sequence[int],
    ) -> Tuple[List[Tuple[int, ...]], List[complex]]:
        r"""Decompose a marginal computational-basis projector into Majorana monomials.

        .. math::
            P_y = \prod_i \frac{I + (1 - 2 y_i)\, Z_{q_i}}{2},
            \qquad Z_q = -i\, c_{2q} c_{2q+1},

        which expands into one monomial per subset of the measured qubits, :math:`2^k` in all. The
        overall :math:`2^{-k}` is not folded into the coefficients.

        Every monomial is parity-even, so this decomposition needs no parity marker.

        :param target_bits: Outcome bit of each measured qubit.
        :param measured_qubits: Qubit indices the bits refer to, index-aligned with ``target_bits``.
            They must be distinct; any order is accepted.
        :return: Index-aligned ``(supports, coefficients)``, ready for ``sum_monomials``.
        :rtype: Tuple[List[Tuple[int, ...]], List[complex]]
        :raises ValueError: if ``target_bits`` and ``measured_qubits`` have different lengths, or if a
            qubit is measured twice.
        """
        bits = [int(bit) for bit in target_bits]
        qubits = [int(qubit) for qubit in measured_qubits]
        if len(bits) != len(qubits):
            raise ValueError(
                f"marginal_projector_terms got {len(bits)} bits and {len(qubits)} measured qubits; "
                f"they must be index-aligned."
            )
        # Distinct qubits give disjoint, commuting bilinears; a repeat would need c_mu^2 = I
        if len(set(qubits)) != len(qubits):
            raise ValueError(f"marginal_projector_terms got a repeated measured qubit in {qubits}.")
        supports: List[Tuple[int, ...]] = [()]
        coefficients: List[complex] = [1.0 + 0.0j]
        for bit, qubit in zip(bits, qubits):
            factor = complex((1 - 2 * bit) * -1j)
            supports = supports + [tuple(sorted(support + (2 * qubit, 2 * qubit + 1))) for support in supports]
            coefficients = coefficients + [coefficient * factor for coefficient in coefficients]
        return supports, coefficients

    @staticmethod
    def from_observable(
        observable: Operator,
        wires: Sequence,
        marker: Optional[int] = None,
    ) -> Tuple[List[Tuple[int, ...]], List[complex]]:
        r"""Decompose a Pauli-sum observable into sorted Majorana monomials.

        Terms whose Majorana support has odd weight take the parity marker when ``marker`` is given,
        and are dropped otherwise.

        :param observable: A Pauli-decomposable observable, or a bare Pauli word.
        :param wires: Device wire labels in qubit order.
        :param marker: Parity-marker index ``D - 1`` on the lifted path, or ``None`` on the basis path.
        :return: Index-aligned ``(supports, coefficients)``, ready for ``sum_monomials``.
        :rtype: Tuple[List[Tuple[int, ...]], List[complex]]
        :raises TypeError: if part of ``observable`` has no Pauli representation, or is a non-integer
            power, or a negative power of anything but a single Pauli word.
        :raises ValueError: if ``observable`` acts on a wire that is not in ``wires``.
        """
        wire_list = list(wires)
        wire_positions = {wire: index for index, wire in enumerate(wire_list)}
        jordan_wigner = JordanWigner(len(wire_list))
        supports: List[Tuple[int, ...]] = []
        coefficients: List[complex] = []
        for pauli_word, value in live_pauli_sentence(observable).items():
            unknown_wires = [wire for wire in pauli_word.wires if wire not in wire_positions]
            if unknown_wires:
                raise ValueError(
                    f"the observable acts on wires {unknown_wires}, which are not in the device wires {wire_list}."
                )
            support, kappa = jordan_wigner.pauli_to_majorana(pauli_word, wire_list)
            modes = [int(mode) for mode in support]
            if len(modes) % 2 != 0:
                if marker is None:
                    continue  # parity-odd term vanishes on the parity-even basis path
                modes = modes + [int(marker)]
            supports.append(tuple(modes))
            coefficients.append(_as_complex(value) * complex(kappa))
        return supports, coefficients
