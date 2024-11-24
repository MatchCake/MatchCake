from typing import Callable, Tuple

import tqdm
from pennylane.typing import TensorLike
from pennylane.wires import Wires
import pennylane as qml
import numpy as np

from ..sampling_strategies.k_qubits_by_k_qubits_sampling import KQubitsByKQubitsSampling
from .star_state_finding_strategy import StarStateFindingStrategy
from ...utils.torch_utils import to_numpy


class GreedyStrategy(StarStateFindingStrategy):
    NAME: str = "Greedy"

    def __call__(
            self,
            device: qml.QubitDevice,
            states_prob_func: Callable[[TensorLike, Wires], TensorLike],
            **kwargs
    ) -> Tuple[TensorLike, TensorLike]:
        p_bar = tqdm.tqdm(
            total=device.num_wires,
            desc=f"[{self.NAME}] Finding Star State",
            disable=not kwargs.get("show_progress", False),
            unit=f"wire",
        )
        k = 2
        added_states = device.states_to_binary(np.arange(int(2 ** k)), k)
        # pi_0 = pi_0(x_0) = [p_0(0), p_0(1)]
        probs = to_numpy(states_prob_func(added_states, Wires(range(k)))).T
        star_states = added_states[np.argmax(probs, axis=-1)]
        p_bar.update(k)

        # x_0
        n_wires_left = device.num_wires % k
        n_k = len(list(range(k, device.num_wires - n_wires_left, k)))
        k_list = [j for j in ([k] * n_k) + [n_wires_left] if j > 0]

        for k_j in k_list:
            added_states = device.states_to_binary(np.arange(2 ** k_j), k_j)

            extended_states = KQubitsByKQubitsSampling.extend_states(star_states, added_states, unique=True)
            # compute probs of unique states: pi_j = pi_j(x_0, ..., x_{j-1}, x_j) / pi_{j-1}(x_0, ..., x_{j-1})
            extended_states_probs = to_numpy(states_prob_func(extended_states, np.arange(extended_states.shape[-1])))
            probs = KQubitsByKQubitsSampling.compute_extend_probs_to_all(binaries_array, extended_states, extended_states_probs)

            # Sample x_j from the probability distribution pi_j
            samples = random_index(probs, axis=-1)

            # x_j
            binary = device.states_to_binary(samples, k_j)
            binaries.append(binary)
            p_bar.update(k_j)

        p_bar.close()

        # return x_0 ... x_{n-1}
        return qml.math.concatenate(binaries, -1)

