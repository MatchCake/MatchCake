from .sampling_strategy import SamplingStrategy
from pennylane.typing import TensorLike
from pennylane.wires import Wires
from typing import Callable
import numpy as np
import pennylane as qml

from ...utils.torch_utils import to_numpy
from ...utils.math import random_index
from ..probability_strategies import ProbabilityStrategy


class QubitByQubitSampling(SamplingStrategy):
    NAME = "QubitByQubitSampling"

    def generate_samples(
            self,
            device: qml.QubitDevice,
            state_prob_func: Callable[[TensorLike, Wires], TensorLike],
            **kwargs
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

        # x_0
        binary = device.states_to_binary(samples, 1)
        binaries = [binary]
        for j in range(1, device.num_wires):
            binaries_array = qml.math.concatenate(binaries, -1)
            zeros_states = qml.math.concatenate([binaries_array, qml.math.zeros((device.shots, 1))], -1)
            ones_states = qml.math.concatenate([binaries_array, qml.math.ones((device.shots, 1))], -1)
            zeros_probs = qml.math.stack([
                state_prob_func(state.astype(int), Wires(np.arange(j + 1)))
                for state in zeros_states
            ])
            ones_probs = qml.math.stack([
                state_prob_func(state.astype(int), Wires(np.arange(j + 1)))
                for state in ones_states
            ])
            # pi_j = pi_j(x_0, ..., x_{j-1}, x_j) / pi_{j-1}(x_0, ..., x_{j-1})
            # probs = qml.math.stack([zeros_probs, ones_probs], -1) / probs
            probs = to_numpy(qml.math.stack([zeros_probs, ones_probs], -1))

            # Sample x_j from the probability distribution pi_j
            samples = qml.math.concatenate([np.random.choice([0, 1], 1, p=p / p.sum()) for p in probs])

            # x_j
            binary = device.states_to_binary(samples, 1)
            binaries.append(binary)

        # return x_0 ... x_{n-1}
        return qml.math.concatenate(binaries, -1)

    def batch_generate_samples(
            self,
            device: qml.QubitDevice,
            states_prob_func: Callable[[TensorLike, Wires], TensorLike],
            **kwargs
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
        probs = to_numpy(states_prob_func(np.asarray([[0], [1]]), [Wires([0]), Wires([0])]))
        # Sample x_0 from the probability distribution pi_0(x_0)
        samples = random_index(probs, n=device.shots, axis=-1)
        batch_shape = qml.math.shape(samples)
        ravel_batch_size = np.prod(batch_shape)

        # x_0
        binary = device.states_to_binary(samples, 1)
        binaries = [binary]
        for j in range(1, device.num_wires):
            binaries_array = qml.math.concatenate(binaries, -1)
            zeros_states = qml.math.concatenate([binaries_array, qml.math.zeros_like(binaries_array)], -1).astype(int)
            ones_states = qml.math.concatenate([binaries_array, qml.math.ones_like(binaries_array)], -1).astype(int)
            half_wires = np.arange(j + 1).reshape((1, -1)).repeat(ravel_batch_size, axis=0)
            # pi_j = pi_j(x_0, ..., x_{j-1}, x_j) / pi_{j-1}(x_0, ..., x_{j-1})
            zeros_probs = states_prob_func(zeros_states.reshape(ravel_batch_size, -1), half_wires).reshape(*batch_shape, -1)
            ones_probs = states_prob_func(ones_states.reshape(ravel_batch_size, -1), half_wires).reshape(*batch_shape, -1)
            probs = to_numpy(qml.math.stack([zeros_probs, ones_probs], -1))
            # states = qml.math.concatenate([zeros_states, ones_states], 0)
            # wires = np.arange(j + 1).reshape((1, -1)).repeat(states.shape[0], axis=0)
            # probs = to_numpy(states_prob_func(states, wires))
            # split the first dimension into two parts
            # Sample x_j from the probability distribution pi_j
            samples = random_index(probs, axis=-1)

            # x_j
            binary = device.states_to_binary(samples, 1)
            binaries.append(binary)

        # return x_0 ... x_{n-1}
        return qml.math.concatenate(binaries, -1)
