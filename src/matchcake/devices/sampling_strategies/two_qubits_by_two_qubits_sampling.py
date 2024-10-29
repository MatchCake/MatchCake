import itertools

import tqdm

from .sampling_strategy import SamplingStrategy
from pennylane.typing import TensorLike
from pennylane.wires import Wires
from typing import Callable
import numpy as np
import pennylane as qml

from ...utils.math import random_index
from ...utils.torch_utils import to_numpy


class TwoQubitsByTwoQubitsSampling(SamplingStrategy):
    NAME = "2QubitBy2QubitSampling"

    def generate_samples(
            self,
            device: qml.QubitDevice,
            prob_func: Callable[[Wires, TensorLike], TensorLike],
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
        k = 2
        p_bar = tqdm.tqdm(
            total=device.num_wires // k,
            desc=f"[{self.NAME}] Generating Samples",
            disable=not kwargs.get("show_progress", False),
            unit=f"{k}set",
        )
        bk = int(2 ** k)
        added_states = device.states_to_binary(np.arange(bk), k)
        # pi_0 = pi_0(x_0) = [p_0(0), p_0(1)]
        if device.num_wires % k == 0:
            first_k = k
            first_added_states = added_states
        else:
            first_k = device.num_wires % k
            first_added_states = device.states_to_binary(np.arange(2 ** first_k), first_k)
        probs = to_numpy(states_prob_func(first_added_states, Wires(range(first_k)))).T
        # Sample x_0 from the probability distribution pi_0(x_0)
        samples = random_index(probs, n=device.shots, axis=-1)
        batch_shape = qml.math.shape(samples)
        p_bar.update(1)

        # x_0
        binary = device.states_to_binary(samples, k)
        binaries = [binary]
        for j in range(first_k, device.num_wires, k):
            binaries_array = qml.math.concatenate(binaries, -1)
            unique_states = np.unique(binaries_array.reshape(-1, binaries_array.shape[-1]), axis=0)
            unique_batch_shape = unique_states.shape[:-1]

            # build next states
            all_states_list = [
                qml.math.concatenate([unique_states, qml.math.full((*unique_batch_shape, k), state)], -1)
                for state in added_states
            ]
            all_states = qml.math.stack(all_states_list, axis=0)
            all_states = all_states.reshape(-1, all_states.shape[-1]).astype(int)

            # compute probs of unique states: pi_j = pi_j(x_0, ..., x_{j-1}, x_j) / pi_{j-1}(x_0, ..., x_{j-1})
            try:
                unique_states_probs = to_numpy(states_prob_func(all_states, np.arange(all_states.shape[-1])))
            except Exception as e:
                unique_states_probs = to_numpy(states_prob_func(all_states, np.arange(all_states.shape[-1])))

            # compute the probs tensor
            probs = qml.math.full((*batch_shape, bk), 0.0)
            for i, state in enumerate(all_states):
                mask = np.isclose(binaries_array, state[:-k]).all(axis=-1)
                state_idx = int(state[-k:].dot(2 ** np.arange(k)[::-1]))
                probs[..., state_idx] = np.where(mask, unique_states_probs[i, ...], probs[..., state_idx])

            # Sample x_j from the probability distribution pi_j
            samples = random_index(probs, axis=-1)

            # x_j
            binary = device.states_to_binary(samples, k)
            binaries.append(binary)
            p_bar.update(1)
        p_bar.close()

        # return x_0 ... x_{n-1}
        return qml.math.concatenate(binaries, -1)
