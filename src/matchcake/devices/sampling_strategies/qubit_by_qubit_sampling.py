from typing import Callable

import numpy as np
import pennylane as qml
import tqdm
from pennylane.typing import TensorLike
from pennylane.wires import Wires

from ...utils.math import random_index
from ...utils.torch_utils import to_numpy
from .k_qubits_by_k_qubits_sampling import KQubitsByKQubitsSampling


class QubitByQubitSampling(KQubitsByKQubitsSampling):
    NAME = "QubitByQubitSampling"
    K: int = 1

    def generate_samples(
        self,
        device: qml.devices.QubitDevice,
        state_prob_func: Callable[[TensorLike, Wires], TensorLike],
        **kwargs,
    ) -> TensorLike:
        """
        Generate qubit-by-qubit samples.

        1. Sample x_0 from the probability distribution pi_0(x_0)
        2. for j = 1 to n-1 do
        3.     Sample x_j from the probability distribution pi_j(x_0, ..., x_{j-1}, x_j) / pi_{j-1}(x_0, ..., x_{j-1})
        4. end for
        5. return x_0 ... x_{n-1}

        :return: Samples with shape (shots, num_wires) of probabilities |<x|psi>|^2
        """
        # pi_0 = pi_0(x_0) = [p_0(0), p_0(1)]
        probs = to_numpy(qml.math.stack([state_prob_func([i], Wires([0])) for i in [0, 1]], axis=-1))
        probs = probs / probs.sum(axis=-1, keepdims=True)

        # Sample x_0 from the probability distribution pi_0(x_0)
        samples = device.sample_basis_states(2, to_numpy(probs))
        batch_shape = qml.math.shape(samples)

        # x_0
        binary = device.states_to_binary(samples, 1)
        binaries = [binary]
        for j in range(1, device.num_wires):
            binaries_array = qml.math.concatenate(binaries, -1)
            unique_states = np.unique(binaries_array, axis=0)
            unique_batch_shape = unique_states.shape[:-1]

            zeros_states = qml.math.concatenate([unique_states, qml.math.full((*unique_batch_shape, 1), 0)], -1)
            ones_states = qml.math.concatenate([unique_states, qml.math.full((*unique_batch_shape, 1), 1)], -1)
            all_states = qml.math.stack([zeros_states, ones_states], axis=0).reshape(-1, j + 1).astype(int)
            all_probs = qml.math.stack(
                [state_prob_func(state.astype(int), Wires(np.arange(j + 1))) for state in all_states]
            )
            # pi_j = pi_j(x_0, ..., x_{j-1}, x_j) / pi_{j-1}(x_0, ..., x_{j-1})

            probs = qml.math.full((*batch_shape, 2), 0.0)
            for i, state in enumerate(all_states):
                mask = np.isclose(binaries_array, state[:-1]).all(axis=-1)
                probs[..., state[-1]] = np.where(mask, all_probs[i], probs[..., state[-1]])

            # Sample x_j from the probability distribution pi_j
            samples = random_index(probs, axis=-1)

            # x_j
            binary = device.states_to_binary(samples, 1)
            binaries.append(binary)

        # return x_0 ... x_{n-1}
        return qml.math.concatenate(binaries, -1)
