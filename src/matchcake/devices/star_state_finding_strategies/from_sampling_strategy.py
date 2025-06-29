from typing import Callable, Tuple

import numpy as np
import pennylane as qml
from pennylane.typing import TensorLike
from pennylane.wires import Wires

from ...utils.torch_utils import to_numpy
from .star_state_finding_strategy import StarStateFindingStrategy


class FromSamplingStrategy(StarStateFindingStrategy):
    NAME: str = "FromSampling"

    def __call__(
        self,
        device: qml.devices.QubitDevice,
        states_prob_func: Callable[[TensorLike, Wires], TensorLike],
        **kwargs,
    ) -> Tuple[TensorLike, TensorLike]:
        samples = device.samples
        if samples is None:
            samples = device.generate_samples()

        samples = to_numpy(samples).astype(int)
        #                                 (n_samples, batch_size, sample_size)
        samples_reshaped = samples.reshape(samples.shape[0], -1, samples.shape[-1])

        star_states, star_probs = [], []
        for bi in range(samples_reshaped.shape[1]):
            batch_samples = samples_reshaped[:, bi, :]

            unique_samples, unique_counts = np.unique(batch_samples, return_counts=True, axis=0)
            unique_probs = unique_counts / samples.shape[0]
            star_state = unique_samples[np.argmax(unique_counts)]
            star_prob = unique_probs[np.argmax(unique_counts)]
            star_states.append(star_state)
            star_probs.append(star_prob)

        star_states = np.stack(star_states, axis=0).reshape(samples.shape[1:])
        star_probs = np.stack(star_probs, axis=0).reshape(samples.shape[1:-1])
        return star_states, star_probs
