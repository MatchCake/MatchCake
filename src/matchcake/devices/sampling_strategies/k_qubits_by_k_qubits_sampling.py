from typing import Callable, List, Optional, Tuple

import numpy as np
import pennylane as qml
import tqdm
from pennylane.typing import TensorLike
from pennylane.wires import Wires

from ...utils._pfaffian_family import sample_outcomes
from ...utils.math import random_index
from ...utils.torch_utils import to_tensor
from .sampling_strategy import SamplingStrategy

_SAMPLER_CROSSOVER_K = 10


class KQubitsByKQubitsSampling(SamplingStrategy):
    r"""Autoregressive sampler that draws ``K`` qubits per step.

    Samples are generated block by block following the chain rule of
    probability:

    1. Sample :math:`x_0` from the marginal :math:`\pi_0(x_0)`.
    2. For :math:`j = 1` to :math:`n - 1`, sample :math:`x_j` from the
       conditional :math:`\pi_j(x_0, \dots, x_{j-1}, x_j) /
       \pi_{j-1}(x_0, \dots, x_{j-1})`.
    3. Return :math:`x_0 \dots x_{n-1}`.

    The implementation is backend agnostic. The probability callback may return
    NumPy arrays, Torch tensors, or any other ``qml.math`` backend; all
    probability arithmetic stays in that backend and only the discrete binary
    samples are materialised as NumPy integers. Arbitrary leading batch
    dimensions on the probability outputs are supported.
    """

    NAME = "kQubitBykQubitSampling"
    K: Optional[int] = None

    @staticmethod
    def build_subset_sizes(num_wires: int, k: int) -> List[int]:
        """Split ``num_wires`` into the per-step block sizes used by the sampler.

        The first ``k`` wires are sampled by the marginal step, so this returns
        the sizes of the remaining steps. Every step samples ``k`` wires except
        a final step covering ``num_wires % k`` leftover wires when ``k`` does
        not divide ``num_wires``.

        :param num_wires: Total number of wires to sample.
        :type num_wires: int
        :param k: Number of wires sampled per step.
        :type k: int
        :return: List of block sizes for the steps following the marginal step.
        :rtype: List[int]
        """
        n_wires_left = num_wires % k
        n_full_steps = len(range(k, num_wires - n_wires_left, k))
        return [size for size in ([k] * n_full_steps) + [n_wires_left] if size > 0]

    @staticmethod
    def _device_covariance(device: qml.devices.QubitDevice) -> Optional[TensorLike]:
        """Return the device's real Majorana covariance, or None when it is unavailable.

        The covariance exists only for a product-state preparation, whose evolved
        ``Lambda(t)`` is a valid Pfaffian probability family for :func:`sample_outcomes`. A
        complex-dtyped covariance is rejected: the sampler needs a real family and ``to_tensor``
        would silently discard the imaginary part.

        :param device: Device providing ``covariance_matrix``.
        :type device: qml.devices.QubitDevice
        :return: The real covariance matrix, or None.
        :rtype: Optional[TensorLike]
        """
        try:
            covariance = device.covariance_matrix
        except (ValueError, AttributeError):
            return None
        if "complex" in str(getattr(covariance, "dtype", "")):
            return None
        return covariance

    @staticmethod
    def _covariance_in_wire_label_order(covariance: TensorLike, device: qml.devices.QubitDevice) -> TensorLike:
        """Reorder the covariance Majorana blocks from device wire-position order to wire-label order.

        ``device.covariance_matrix`` is built in ``device.wires`` position order, but samples must be
        emitted in wire-label order (PennyLane's ``q_0, q_1, ...`` convention, which the per-step loop
        follows via ``all_wires.indices``). For identity labels (``wires = range(n)``) this is a no-op.

        :param covariance: Covariance ``(..., 2n, 2n)`` in device wire-position order.
        :type covariance: TensorLike
        :param device: Device providing the wire labels.
        :type device: qml.devices.QubitDevice
        :return: Covariance reordered so block ``i`` corresponds to wire label ``i``.
        :rtype: TensorLike
        """
        positions = np.asarray(device.wires.indices(Wires(range(device.num_wires))))  # position of label i
        majorana = np.stack([2 * positions, 2 * positions + 1], axis=1).ravel()  # (2n,)
        return covariance[..., majorana[:, None], majorana[None, :]]

    @staticmethod
    def _sample_from_covariance(covariance: TensorLike, shots: int) -> np.ndarray:
        """Draw all-wire samples directly from the covariance family.

        Uses the closed-form autoregressive conditionals (``O(k^3)`` per shot, no full-matrix
        Pfaffian), which also avoids the additive-epsilon normalization bias of the per-step
        ``random_index`` sampler at large wire counts.

        :param covariance: Real Majorana covariance ``(..., 2k, 2k)``.
        :type covariance: TensorLike
        :param shots: Number of samples.
        :type shots: int
        :return: Samples ``(shots, ..., k)`` (big-endian per wire).
        :rtype: np.ndarray
        """
        shots = int(getattr(shots, "total_shots", shots))
        return np.asarray(sample_outcomes(to_tensor(covariance), shots))

    @classmethod
    def extend_states(cls, states: TensorLike, added_states: TensorLike, unique: bool = True) -> TensorLike:
        """Append every candidate block to the (unique) prefixes seen so far.

        :param states: Already sampled prefixes, shape ``(..., prefix_len)``.
        :type states: TensorLike
        :param added_states: Candidate blocks to append, shape ``(2 ** k, k)``.
        :type added_states: TensorLike
        :param unique: Whether to deduplicate the prefixes before extending.
        :type unique: bool
        :return: Extended states, shape ``(2 ** k * n_prefixes, prefix_len + k)``.
            Rows are block-major: row ``block_idx * n_prefixes + prefix_idx``.
        :rtype: TensorLike
        """
        extended_states, _, _ = cls._extend_with_inverse(states, added_states, unique=unique)
        return extended_states

    @classmethod
    def scatter_extended_probs(
        cls,
        extended_states_probs: TensorLike,
        prefix_inverse: np.ndarray,
        n_prefixes: int,
        n_blocks: int,
        batch_shape: Tuple[int, ...],
    ) -> TensorLike:
        """Scatter unique extended-state probabilities back onto every sample.

        Given the probabilities of the ``n_blocks * n_prefixes`` unique extended
        states, this gathers, for each sample, the probability of every
        candidate block conditioned on that sample's prefix.

        :param extended_states_probs: Probabilities of the extended states,
            shape ``(n_blocks * n_prefixes, *extra)`` where ``extra`` are the
            optional circuit-batch dimensions.
        :type extended_states_probs: TensorLike
        :param prefix_inverse: Mapping from each flattened sample to its prefix
            index, shape ``(prod(batch_shape),)``.
        :type prefix_inverse: np.ndarray
        :param n_prefixes: Number of unique prefixes.
        :type n_prefixes: int
        :param n_blocks: Number of candidate blocks (``2 ** k``).
        :type n_blocks: int
        :param batch_shape: Shape of the sample batch, ``(shots, *extra)``.
        :type batch_shape: Tuple[int, ...]
        :return: Per-sample block probabilities, shape ``(*batch_shape, n_blocks)``.
        :rtype: TensorLike
        """
        extra_dims = tuple(qml.math.shape(extended_states_probs)[1:])
        # grid[block_idx, prefix_idx, *extra] = P(prefix extended by block | *extra)
        grid = qml.math.reshape(extended_states_probs, (n_blocks, n_prefixes) + extra_dims)

        n_extra = len(extra_dims)
        inverse = np.asarray(prefix_inverse, dtype=int).reshape(batch_shape)

        block_index = np.arange(n_blocks).reshape((1,) * (1 + n_extra) + (n_blocks,))
        prefix_index = inverse.reshape(batch_shape + (1,))
        gather_indices = [block_index, prefix_index]
        for extra_axis, extra_size in enumerate(extra_dims):
            index_shape = [1] * (2 + n_extra)
            index_shape[1 + extra_axis] = extra_size
            gather_indices.append(np.arange(extra_size).reshape(index_shape))

        # Fancy indexing broadcasts to (*batch_shape, n_blocks) and preserves the backend.
        return grid[tuple(gather_indices)]

    def batch_generate_samples_by_subsets_of_k(
        self,
        device: qml.devices.QubitDevice,
        states_prob_func: Callable[[TensorLike, Wires], TensorLike],
        k: int = 1,
        **kwargs,
    ) -> TensorLike:
        """Generate samples ``k`` wires at a time using the chain rule.

        :param device: Device providing ``num_wires``, ``shots`` and binary helpers.
        :type device: qml.devices.QubitDevice
        :param states_prob_func: Callback returning the probabilities of a batch
            of binary states on the given wires.
        :type states_prob_func: Callable[[TensorLike, Wires], TensorLike]
        :param k: Number of wires sampled per step.
        :type k: int
        :return: Samples of shape ``(shots, *batch, num_wires)`` drawn from
            :math:`|\\langle x | \\psi \\rangle|^2`.
        :rtype: TensorLike
        """
        # Covariance fast path: for a product-state family above the size crossover, draw all wires
        # at once with the O(k^3)/shot autoregressive sampler instead of the per-step batched-prefix
        # loop. Same distribution, no full-matrix Pfaffian, and no random_index normalization bias.
        if device.num_wires >= _SAMPLER_CROSSOVER_K:
            covariance = self._device_covariance(device)
            if covariance is not None:
                covariance = self._covariance_in_wire_label_order(covariance, device)
                return self._sample_from_covariance(covariance, device.shots)

        p_bar = tqdm.tqdm(
            total=device.num_wires,
            desc=f"[{self.NAME}] Generating Samples by Subsets of {k}",
            disable=not kwargs.get("show_progress", False),
            unit="wire",
        )

        # Marginal step: pi_0(x_0) over the first k wires.
        added_states = device.states_to_binary(np.arange(int(2**k)), k)
        marginal_probs = qml.math.transpose(states_prob_func(added_states, Wires(range(k))))  # (*batch, 2 ** k)
        samples = random_index(marginal_probs, n=device.shots, axis=-1)  # (shots, *batch)
        binaries = [device.states_to_binary(samples, k)]
        p_bar.update(k)

        for k_j in self.build_subset_sizes(device.num_wires, k):
            added_states = device.states_to_binary(np.arange(2**k_j), k_j)
            prefixes = qml.math.concatenate(binaries, -1)  # (shots, *batch, prefix_len)

            extended_states, prefix_inverse, n_prefixes = self._extend_with_inverse(prefixes, added_states)
            # Conditional masses pi_j(prefix, block); random_index normalises over
            # the block axis, dividing by the prefix marginal pi_{j-1}(prefix).
            extended_states_probs = states_prob_func(extended_states, np.arange(extended_states.shape[-1]))
            probs = self.scatter_extended_probs(
                extended_states_probs,
                prefix_inverse,
                n_prefixes,
                int(2**k_j),
                qml.math.shape(prefixes)[:-1],
            )

            samples = random_index(probs, axis=-1)  # (shots, *batch)
            binaries.append(device.states_to_binary(samples, k_j))
            p_bar.update(k_j)

        p_bar.close()
        return qml.math.concatenate(binaries, -1)

    def batch_generate_samples(
        self,
        device: qml.devices.QubitDevice,
        states_prob_func: Callable[[TensorLike, Wires], TensorLike],
        **kwargs,
    ) -> TensorLike:
        """Generate batched samples using ``K`` (or 1) wires per step.

        :param device: Device to sample from.
        :type device: qml.devices.QubitDevice
        :param states_prob_func: Batched probability callback.
        :type states_prob_func: Callable[[TensorLike, Wires], TensorLike]
        :return: Samples of shape ``(shots, *batch, num_wires)``.
        :rtype: TensorLike
        """
        return self.batch_generate_samples_by_subsets_of_k(device, states_prob_func, k=self.K or 1, **kwargs)

    def generate_samples(
        self,
        device: qml.devices.QubitDevice,
        state_prob_func: Callable[[TensorLike, Wires], TensorLike],
        **kwargs,
    ) -> TensorLike:
        """Generate samples from a single-state probability callback.

        Wraps a callback that scores one binary state at a time into the batched
        interface used by :meth:`batch_generate_samples_by_subsets_of_k`.

        :param device: Device to sample from.
        :type device: qml.devices.QubitDevice
        :param state_prob_func: Callback scoring a single binary state on the
            given wires.
        :type state_prob_func: Callable[[TensorLike, Wires], TensorLike]
        :return: Samples of shape ``(shots, *batch, num_wires)``.
        :rtype: TensorLike
        """

        def states_prob_func(states: TensorLike, wires: Wires) -> TensorLike:
            wires = np.asarray(wires)
            if wires.ndim == 1:
                wires = np.expand_dims(wires, axis=0).repeat(len(states), axis=0)
            return qml.math.stack([state_prob_func(s, w) for s, w in zip(states, wires)], axis=0)

        return self.batch_generate_samples_by_subsets_of_k(device, states_prob_func, k=self.K or 1, **kwargs)

    @classmethod
    def _extend_with_inverse(
        cls,
        states: TensorLike,
        added_states: TensorLike,
        unique: bool = True,
    ) -> Tuple[TensorLike, np.ndarray, int]:
        """Deduplicate prefixes and append every candidate block to each.

        :param states: Sampled prefixes, shape ``(..., prefix_len)``.
        :type states: TensorLike
        :param added_states: Candidate blocks, shape ``(2 ** k, k)``.
        :type added_states: TensorLike
        :param unique: Whether to deduplicate prefixes. When ``False`` every
            input row is treated as its own prefix.
        :type unique: bool
        :return: Tuple of the extended states (block-major, shape
            ``(2 ** k * n_prefixes, prefix_len + k)``), the inverse mapping from
            each flattened input prefix to its prefix index, and the number of
            unique prefixes.
        :rtype: Tuple[TensorLike, np.ndarray, int]
        """
        prefix_len = qml.math.shape(states)[-1]
        flat_prefixes = np.asarray(qml.math.reshape(states, (-1, prefix_len))).astype(int)
        if unique:
            prefixes, inverse = np.unique(flat_prefixes, axis=0, return_inverse=True)
        else:
            prefixes = flat_prefixes
            inverse = np.arange(flat_prefixes.shape[0])
        inverse = np.asarray(inverse, dtype=int).reshape(-1)

        n_prefixes = prefixes.shape[0]
        k = added_states.shape[-1]
        added_states = np.asarray(added_states).astype(int)
        blocks = [np.broadcast_to(block, (n_prefixes, k)) for block in added_states]
        extended_states = np.stack([np.concatenate([prefixes, block], axis=-1) for block in blocks], axis=0).reshape(
            -1, prefix_len + k
        )
        return extended_states, inverse, n_prefixes
