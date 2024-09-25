import itertools

from .sampling_strategy import SamplingStrategy
from pennylane.typing import TensorLike
from pennylane.wires import Wires
from typing import Callable
import numpy as np
import pennylane as qml


class TwoQubitsByTwoQubitsSampling(SamplingStrategy):
    NAME = "2QubitBy2QubitSampling"

    def generate_samples(
            self,
            device: qml.QubitDevice,
            prob_func : Callable[[Wires, TensorLike], TensorLike],
            **kwargs
    ) -> TensorLike:
        n_per_subset = 2
        wires_batched = [Wires(wires) for wires in np.asarray(device.wires.tolist()).reshape(-1, n_per_subset)]
        wires_binary_states = np.array(list(itertools.product([0, 1], repeat=n_per_subset)))

        # wires_binary_states = np.array(list(itertools.product([0, 1], repeat=n_per_subset)))
        # prob_func = self.get_prob_strategy_func()
        # wires_batched = [Wires(wires) for wires in np.asarray(self.wires.tolist()).reshape(-1, n_per_subset)]
        #
        # probs_batched = qml.math.stack(
        #     [
        #         [
        #             prob_func(wires, wires_binary_state)
        #             for wires_binary_state in wires_binary_states
        #         ]
        #         for wires in wires_batched
        #     ]
        # )
        # samples_batched = [
        #     self.sample_basis_states(2**n_per_subset, probs / probs.sum())
        #     for probs in probs_batched
        # ]
        # binary_batched = [
        #     self.states_to_binary(samples, n_per_subset)
        #     for samples in samples_batched
        # ]

        # pi_0 = pi_0(x_0) = [p_0(0), p_0(1)]
        probs = qml.math.stack([prob_func(Wires([0]), [i]) for i in [0, 1]])
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
                prob_func(Wires(np.arange(j + 1)), state.astype(int))
                for state in zeros_states
            ])
            ones_probs = qml.math.stack([
                prob_func(Wires(np.arange(j + 1)), state.astype(int))
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

