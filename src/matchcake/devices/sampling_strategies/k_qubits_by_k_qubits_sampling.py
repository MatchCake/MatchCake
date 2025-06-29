import itertools
from typing import Callable

import numpy as np
import pennylane as qml
import tqdm
from pennylane.typing import TensorLike
from pennylane.wires import Wires

from ...utils.math import random_index
from ...utils.torch_utils import to_numpy
from .sampling_strategy import SamplingStrategy


class KQubitsByKQubitsSampling(SamplingStrategy):
    NAME = "kQubitBykQubitSampling"
    K: int = None

    @classmethod
    def compute_extend_probs_to_all(
        cls,
        all_states: TensorLike,
        extended_states: TensorLike,
        extended_states_probs: TensorLike,
    ):
        batch_shape = all_states.shape[:-1]
        k = extended_states.shape[-1] - all_states.shape[-1]
        bk = int(2**k)
        probs = qml.math.full((*batch_shape, bk), 0.0)
        for i, state in enumerate(extended_states):
            # mask = np.isclose(all_states, state[:-k]).all(axis=-1)
            mask = (all_states == state[:-k]).all(axis=-1)
            state_idx = int(state[-k:].dot(2 ** np.arange(k)[::-1]))
            probs[..., state_idx] = np.where(mask, extended_states_probs[i, ...], probs[..., state_idx])
        return probs

    @classmethod
    def extend_states(cls, states: TensorLike, added_states: TensorLike, unique: bool = True) -> TensorLike:
        if unique:
            states = np.unique(states.reshape(-1, states.shape[-1]), axis=0)
        k = added_states.shape[-1]
        states_batch_shape = states.shape[:-1]
        all_states_list = [
            qml.math.concatenate([states, qml.math.full((*states_batch_shape, k), state)], -1) for state in added_states
        ]
        all_states = qml.math.stack(all_states_list, axis=0)
        all_states = all_states.reshape(-1, all_states.shape[-1]).astype(int)
        return all_states

    def batch_generate_samples_by_subsets_of_k(
        self,
        device: qml.devices.QubitDevice,
        states_prob_func: Callable[[TensorLike, Wires], TensorLike],
        k: int = 1,
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
        p_bar = tqdm.tqdm(
            total=device.num_wires,
            desc=f"[{self.NAME}] Generating Samples by Subsets of {k}",
            disable=not kwargs.get("show_progress", False),
            unit=f"wire",
        )
        added_states = device.states_to_binary(np.arange(int(2**k)), k)
        # pi_0 = pi_0(x_0) = [p_0(0), p_0(1)]
        probs = to_numpy(states_prob_func(added_states, Wires(range(k)))).T
        # Sample x_0 from the probability distribution pi_0(x_0)
        samples = random_index(probs, n=device.shots, axis=-1)
        p_bar.update(k)

        # x_0
        binary = device.states_to_binary(samples, k)
        binaries = [binary]
        n_wires_left = device.num_wires % k
        n_k = len(list(range(k, device.num_wires - n_wires_left, k)))
        k_list = [j for j in ([k] * n_k) + [n_wires_left] if j > 0]

        for k_j in k_list:
            added_states = device.states_to_binary(np.arange(2**k_j), k_j)
            binaries_array = qml.math.concatenate(binaries, -1)

            extended_states = self.extend_states(binaries_array, added_states, unique=True)
            # compute probs of unique states: pi_j = pi_j(x_0, ..., x_{j-1}, x_j) / pi_{j-1}(x_0, ..., x_{j-1})
            extended_states_probs = to_numpy(states_prob_func(extended_states, np.arange(extended_states.shape[-1])))
            probs = self.compute_extend_probs_to_all(binaries_array, extended_states, extended_states_probs)

            # Sample x_j from the probability distribution pi_j
            samples = random_index(probs, axis=-1)

            # x_j
            binary = device.states_to_binary(samples, k_j)
            binaries.append(binary)
            p_bar.update(k_j)

        p_bar.close()

        # return x_0 ... x_{n-1}
        return qml.math.concatenate(binaries, -1)

    def batch_generate_samples(
        self,
        device: qml.devices.QubitDevice,
        states_prob_func: Callable[[TensorLike, Wires], TensorLike],
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
        return self.batch_generate_samples_by_subsets_of_k(device, states_prob_func, k=self.K, **kwargs)

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

        def states_prob_func(states: TensorLike, wires: Wires) -> TensorLike:
            wires = np.asarray(wires)
            if wires.ndim == 1:
                wires = np.expand_dims(wires, axis=0).repeat(len(states), axis=0)
            return qml.math.stack([state_prob_func(s, w) for s, w in zip(states, wires)], axis=0)

        return self.batch_generate_samples_by_subsets_of_k(device, states_prob_func, k=self.K, **kwargs)
