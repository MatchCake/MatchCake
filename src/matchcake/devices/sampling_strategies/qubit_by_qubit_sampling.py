from .sampling_strategy import SamplingStrategy
from pennylane.typing import TensorLike
from pennylane.wires import Wires
from typing import Callable
import numpy as np
import pennylane as qml

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
        probs = qml.math.stack([state_prob_func([i], Wires([0])) for i in [0, 1]])
        probs = probs / probs.sum()

        # Sample x_0 from the probability distribution pi_0(x_0)
        samples = device.sample_basis_states(2, probs)

        # x_0
        binary = device.states_to_binary(samples, 1)
        binaries = [binary]
        for j in range(1, device.num_wires):
            zeros_states = qml.math.concatenate(
                [qml.math.concatenate(binaries, -1), qml.math.zeros((device.shots, 1))], -1
            )
            ones_states = qml.math.concatenate(
                [qml.math.concatenate(binaries, -1), qml.math.ones((device.shots, 1))], -1
            )
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
            probs = qml.math.stack([zeros_probs, ones_probs], -1)

            # Sample x_j from the probability distribution pi_j
            samples = qml.math.concatenate([np.random.choice([0, 1], 1, p=p / p.sum()) for p in probs])

            # x_j
            binary = device.states_to_binary(samples, 1)
            binaries.append(binary)

        # return x_0 ... x_{n-1}
        return qml.math.concatenate(binaries, -1)

