import torch
from pennylane.pauli import pauli_word_to_string

from ....typing import TensorLike
from ..expval_strategy import ExpvalStrategy
import pennylane as qml
import numpy as np

from ._pauli_map import _MAJORANA_COEFFS_MAP, _MAJORANA_INDICES_LAMBDAS
from ....utils.majorana import majorana_to_pauli
from ....utils.torch_utils import to_tensor


class CliffordExpvalStrategy(ExpvalStrategy):
    def __call__(
            self,
            global_sptm: TensorLike,
            state_prep_op: qml.StatePrep,
            observable: qml.Hamiltonian,
            **kwargs
    ) -> TensorLike:
        wires = state_prep_op.wires
        n_qubits = len(wires)
        global_sptm = to_tensor(global_sptm, dtype=torch.complex128)
        clifford_device = qml.device('default.clifford', wires=wires)

        @qml.qnode(clifford_device)
        def clifford_circuit():
            state_prep_op.queue()
            return [
                qml.expval(majorana_to_pauli(mu, n=n_qubits) @ majorana_to_pauli(nu, n=n_qubits))
                for mu, nu in np.ndindex(2 * n_qubits, 2 * n_qubits)
            ]

        expvals = to_tensor(
            qml.math.stack(clifford_circuit()).reshape(2 * n_qubits, 2 * n_qubits),
            dtype=global_sptm.dtype, device=global_sptm.device
        )

        pauli_kinds = [
            pauli_word_to_string(op, wire_map={w: i for i, w in enumerate(op.wires)})
            for op in observable.ops
        ]
        majorana_coeffs = np.asarray([_MAJORANA_COEFFS_MAP[p] for p in pauli_kinds])
        majorana_indices = np.asarray([
            _MAJORANA_INDICES_LAMBDAS[p](op.wires[0])
            for p, op in zip(pauli_kinds, observable.ops)
        ])

        transition_matrices = qml.math.einsum("...kij->i...kj", global_sptm[..., majorana_indices, :])
        transition_tensor = qml.math.einsum("...i,...j->...ij", *transition_matrices)
        result = qml.math.einsum(
            "k,k,...kij,ij->...",
            to_tensor(majorana_coeffs, dtype=global_sptm.dtype, device=global_sptm.device),
            to_tensor(observable.coeffs, dtype=global_sptm.dtype, device=global_sptm.device),
            transition_tensor,
            expvals,
        )
        return result


