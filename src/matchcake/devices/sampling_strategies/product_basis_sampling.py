from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union, cast

import numpy as np
import pennylane as qml
import torch
import tqdm
from pennylane.typing import TensorLike
from pennylane.wires import Wires

from ...utils.math import convert_and_cast_like
from ...utils.torch_utils import detach, to_tensor
from .sampling_strategy import SamplingStrategy


class ProductBasisSampling(SamplingStrategy):
    r"""Sampler for measurements in an arbitrary single-qubit product basis.

    Draws samples of a matchgate circuit applied to a product state, with each qubit :math:`k`
    measured along its own Bloch direction :math:`\hat{n}_k`, i.e. measuring the observable
    :math:`\hat{n}_k \cdot \vec{\sigma}_k`. Outcome bit ``0`` stands for the eigenvalue
    :math:`+1` and bit ``1`` for :math:`-1`, so ``(0, 0, 1)`` is the computational basis and
    ``(1, 0, 0)`` the X basis. Such measurements leave the matchgate set, since rotating a
    qubit to the X basis takes a Hadamard, but they can still be sampled efficiently
    [Brod2016]_ (Theorem 3 and Sec. III.1).

    The sampler follows Brod's construction. An ancilla in :math:`|+\rangle` is placed before
    the first qubit, with Majorana operators :math:`a_0 = X_A` and :math:`a_1 = Y_A`. On the
    states it meets the ancilla is always in :math:`|+\rangle`, so :math:`X_A = 1`. Its
    Majorana :math:`a_1` is the parity index of the extended covariance matrix
    :math:`\tilde\Lambda` (see
    :func:`~matchcake.devices.expval_strategies.m_pfaffian._extended_covariance.extended_covariance_matrix`),
    which therefore describes the product state with the ancilla as a Gaussian state. The
    circuit acts on it through the lifted single-particle transition matrix.

    The qubits are then measured one at a time, from the first to the last. When qubit
    :math:`k` is reached, the qubits :math:`l < k` are in computational basis states with
    known values of :math:`Z_l`, and on the current state

    .. math::

        X_k = -i\,\sigma_k\, a_1 c_{2k}, \qquad
        Y_k = -i\,\sigma_k\, a_1 c_{2k+1}, \qquad
        Z_k = -i\, c_{2k} c_{2k+1},

    where :math:`\sigma_k = \prod_{l<k} Z_l = \pm 1`. All three are quadratic in the Majorana
    operators :math:`(a_1, c_{2k}, c_{2k+1})`, and a single-qubit rotation of qubit
    :math:`k` acts as a rotation in :math:`SO(3)` of these three operators, which keeps the
    state Gaussian. Each step therefore:

    1. Rotates the triple :math:`(a_1, c_{2k}, c_{2k+1})` so that
       :math:`\hat{n}_k \cdot \vec{\sigma}_k` becomes :math:`-i\, c_{2k} c_{2k+1}`. The new
       :math:`a_1` is the direction :math:`w = (n_z, -\sigma_k n_y, \sigma_k n_x)` of the old
       triple.
    2. Measures :math:`Z_k` on the rotated state. The outcome :math:`\lambda = \pm 1` has
       probability :math:`(1 - \lambda \tilde\Lambda_{2k, 2k+1}) / 2`.
    3. Projects onto the outcome with the rank-2 update

       .. math::

           \tilde\Lambda_{ij} \leftarrow \tilde\Lambda_{ij}
           + \lambda \frac{\tilde\Lambda_{2k, i} \tilde\Lambda_{2k+1, j}
           - \tilde\Lambda_{2k, j} \tilde\Lambda_{2k+1, i}}{1 - \lambda \tilde\Lambda_{2k, 2k+1}},

       and updates :math:`\sigma_{k+1} = \lambda\, \sigma_k`.

    After the rotation qubit :math:`k` is left in a computational basis state, which is what
    makes :math:`\sigma_{k+1}` a number for the next step. The rotation and the update only
    change the probabilities of the qubits already measured, not those of the others, so the
    outcomes follow the distribution of the original measurement. When all the bases are the
    computational basis, the rotations are the identity and the sampler draws ordinary
    computational basis samples.

    The probability of a complete outcome is the product of the conditional probabilities
    and is computed exactly by :meth:`outcome_probabilities`. Marginal probabilities on a
    subset of the qubits are not provided, since whether they can be computed efficiently is
    open [Brod2016]_ (Sec. IV).

    **Cost and implementation.** Step :math:`k` touches a matrix of size
    :math:`d = 2(n - k) + 1`: :math:`O(d)` for the rotation and :math:`O(d^2)` for the rank-2
    update, so a shot costs :math:`\sum_k d^2 \approx \tfrac{4}{3} n^3 = O(n^3)`, with no
    Pfaffian. The work is organized to be efficient on both CPU and GPU:

    * The ancilla is kept as the last index, so the matrix of the unmeasured qubits is always a
      trailing block of a single buffer, updated in place. The update uses ``baddbmm_`` on CPU
      and two ``addcmul_`` on CUDA, the fastest kernels on each.
    * The two rotations of each qubit (one per sign :math:`\sigma_k`) are computed once, and the
      random numbers are drawn on the compute device, so the loop over the qubits runs without
      host synchronization once every shot has its own matrix.
    * Shots with the same outcomes so far share a single matrix, together with the number of
      shots that follow it (prefix deduplication). The split of these shots between the two
      outcomes of the next qubit is drawn from a binomial distribution, which is the same as
      drawing each shot independently. This saves most of the work for states with few likely
      outcomes. Once the number of distinct prefixes exceeds :attr:`DEDUPLICATION_THRESHOLD`
      times the number of shots, every shot gets its own matrix.
    * The shots are processed in chunks that fit a memory budget, so the matrices of a chunk
      stay within memory. See :meth:`shots_per_chunk`.

    :param shots_chunk_size: Number of shots processed together. Defaults to None, in which case
        it is computed from ``memory_budget`` by :meth:`shots_per_chunk`.
    :type shots_chunk_size: Optional[int]
    :param memory_budget: Memory, in bytes, that the matrices of a chunk may use. Defaults to
        None: :attr:`DEFAULT_CPU_MEMORY_BUDGET` on CPU, and :attr:`DEFAULT_CUDA_MEMORY_FRACTION`
        of the free memory on CUDA.
    :type memory_budget: Optional[int]
    :param device: Torch device the samples are computed on, e.g. ``"cuda"``. Defaults to None,
        the device of the extended covariance matrix.
    :type device: Optional[Union[str, torch.device]]
    :param dtype: Floating point dtype of the computation. Defaults to None, the dtype of the
        extended covariance matrix when it is ``float32`` or ``float64``, and ``float64``
        otherwise. ``float32`` is faster, in particular on consumer GPUs, at the cost of
        precision on large circuits.
    :type dtype: Optional[torch.dtype]
    :param deduplicate: Whether shots with the same outcomes so far share their matrix.
        Defaults to True.
    :type deduplicate: bool

    .. [Brod2016] D. J. Brod, "Efficient classical simulation of matchgate circuits with
        generalized inputs and measurements," Phys. Rev. A 93, 062332 (2016),
        arXiv:1602.03539.
    """

    NAME = "ProductBasisSampling"
    DEFAULT_CPU_MEMORY_BUDGET: int = 256 * 2**20
    DEFAULT_CUDA_MEMORY_FRACTION: float = 0.5
    DEDUPLICATION_THRESHOLD: float = 0.25
    # A chunk holds about twice its matrices when the deduplicated prefixes are expanded.
    MEMORY_OVERHEAD_FACTOR: float = 2.0
    PAULI_BLOCH_VECTORS: Dict[str, Tuple[float, float, float]] = {
        "X": (1.0, 0.0, 0.0),
        "Y": (0.0, 1.0, 0.0),
        "Z": (0.0, 0.0, 1.0),
    }

    @staticmethod
    def measurement_frames(bloch_vectors: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
        r"""Build the rotations that turn the measured observable into :math:`Z_k`.

        The rows of each rotation are the new :math:`(a_1, c_{2k}, c_{2k+1})` written in the
        old ones: :math:`w = (n_z, -\sigma n_y, \sigma n_x)`, then two unit vectors
        :math:`u, v` orthogonal to :math:`w` with :math:`u \times v = w`. The rotation is
        proper, and :math:`-i\,(u \cdot c)(v \cdot c) = \hat{n} \cdot \vec{\sigma}`.

        :param bloch_vectors: Unit Bloch vectors, shape (..., 3).
        :type bloch_vectors: torch.Tensor
        :param signs: The signs :math:`\sigma = \pm 1`, shape (...).
        :type signs: torch.Tensor
        :return: The rotations, shape (..., 3, 3).
        :rtype: torch.Tensor
        """
        bloch_vectors, signs = torch.broadcast_tensors(bloch_vectors, signs[..., None])
        signs = signs[..., 0]
        w = torch.stack(
            [bloch_vectors[..., 2], -signs * bloch_vectors[..., 1], signs * bloch_vectors[..., 0]], dim=-1
        )  # (..., 3)
        # The reference vector only needs to be far from parallel to w.
        reference = torch.zeros_like(w)
        use_second_axis = torch.abs(w[..., 1]) < 0.9
        reference[..., 1] = use_second_axis.to(w.dtype)
        reference[..., 2] = (~use_second_axis).to(w.dtype)
        u = reference - torch.sum(reference * w, dim=-1, keepdim=True) * w
        u = u / torch.linalg.norm(u, dim=-1, keepdim=True)
        v = torch.linalg.cross(w, u, dim=-1)
        return torch.stack([w, u, v], dim=-2)

    @classmethod
    def pauli_bloch_vectors(cls, bases: Sequence[str]) -> np.ndarray:
        """Return the Bloch vectors of the given Pauli bases.

        :param bases: One Pauli letter (``"X"``, ``"Y"`` or ``"Z"``) per qubit.
        :type bases: Sequence[str]
        :return: The Bloch vectors, shape (n, 3).
        :rtype: np.ndarray
        :raises ValueError: If a letter is not ``"X"``, ``"Y"`` or ``"Z"``.
        """
        unknown = [basis for basis in bases if basis not in cls.PAULI_BLOCH_VECTORS]
        if unknown:
            raise ValueError(f"Unknown Pauli bases {unknown}. Expected 'X', 'Y' or 'Z'.")
        return np.array([cls.PAULI_BLOCH_VECTORS[basis] for basis in bases], dtype=float).reshape(-1, 3)

    def __init__(
        self,
        shots_chunk_size: Optional[int] = None,
        memory_budget: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
        deduplicate: bool = True,
    ):
        if shots_chunk_size is not None and shots_chunk_size < 1:
            raise ValueError(f"shots_chunk_size must be positive. Got {shots_chunk_size}.")
        if memory_budget is not None and memory_budget < 1:
            raise ValueError(f"memory_budget must be positive. Got {memory_budget}.")
        if dtype is not None and dtype not in (torch.float32, torch.float64):
            raise ValueError(f"dtype must be torch.float32 or torch.float64. Got {dtype}.")
        self.shots_chunk_size = shots_chunk_size
        self.memory_budget = memory_budget
        self.device = None if device is None else torch.device(device)
        self.dtype = dtype
        self.deduplicate = deduplicate

    def batch_generate_samples(
        self,
        device: qml.devices.QubitDevice,
        states_prob_func: Optional[Callable[[TensorLike, Wires], TensorLike]] = None,
        **kwargs,
    ) -> TensorLike:
        """Generate samples from the extended covariance matrix of the device.

        The probability callback is not used, since the probabilities come from the extended
        covariance matrix.

        :param device: Device providing ``num_wires``, ``shots`` and ``extended_covariance_matrix``.
        :type device: qml.devices.QubitDevice
        :param states_prob_func: Unused, kept for the :class:`SamplingStrategy` interface.
        :type states_prob_func: Optional[Callable[[TensorLike, Wires], TensorLike]]

        :keyword bloch_vectors: The Bloch vector of the measurement basis of each wire, in the
            order of the device wires, shape (num_wires, 3). Defaults to the computational basis.
        :keyword show_progress: Whether to display a progress bar. Defaults to False.

        :return: Samples of shape (shots, ..., num_wires).
        :rtype: TensorLike
        """
        bloch_vectors = kwargs.get("bloch_vectors", None)
        if bloch_vectors is None:
            bloch_vectors = self.pauli_bloch_vectors(["Z"] * device.num_wires)
        return self.sample(
            device.extended_covariance_matrix,
            bloch_vectors,
            device.shots,
            show_progress=kwargs.get("show_progress", False),
        )

    def generate_samples(
        self,
        device: qml.devices.QubitDevice,
        state_prob_func: Optional[Callable[[TensorLike, Wires], TensorLike]] = None,
        **kwargs,
    ) -> TensorLike:
        """Generate samples from the extended covariance matrix of the device.

        Same as :meth:`batch_generate_samples`, which takes the same keyword arguments.

        :param device: Device providing ``num_wires``, ``shots`` and ``extended_covariance_matrix``.
        :type device: qml.devices.QubitDevice
        :param state_prob_func: Unused, kept for the :class:`SamplingStrategy` interface.
        :type state_prob_func: Optional[Callable[[TensorLike, Wires], TensorLike]]
        :return: Samples of shape (shots, ..., num_wires).
        :rtype: TensorLike
        """
        return self.batch_generate_samples(device, state_prob_func, **kwargs)

    def outcome_probabilities(
        self,
        extended_covariance_matrix: TensorLike,
        bloch_vectors: TensorLike,
        outcomes: TensorLike,
    ) -> TensorLike:
        """Compute the exact probabilities of complete measurement outcomes.

        :param extended_covariance_matrix: The evolved extended covariance matrix,
            shape (..., 2n+1, 2n+1).
        :type extended_covariance_matrix: TensorLike
        :param bloch_vectors: The Bloch vector of the measurement basis of each qubit, shape (n, 3).
        :type bloch_vectors: TensorLike
        :param outcomes: Outcome bits of all the qubits, shape (m, n).
        :type outcomes: TensorLike
        :return: The probability of each outcome, shape (m, ...).
        :rtype: TensorLike
        """
        gamma, bloch_vectors_t = self._prepare_inputs(extended_covariance_matrix, bloch_vectors)
        n_qubits = bloch_vectors_t.shape[0]
        outcomes_t = torch.as_tensor(np.asarray(qml.math.unwrap(outcomes)), dtype=torch.long, device=gamma.device)
        outcomes_t = outcomes_t.reshape(-1, outcomes_t.shape[-1])  # (m, n)
        if outcomes_t.shape[-1] != n_qubits:
            raise ValueError(f"Expected outcomes on {n_qubits} qubits. Got {outcomes_t.shape[-1]} instead.")
        batch_shape = tuple(gamma.shape[:-2])
        roots = gamma.reshape((-1,) + gamma.shape[-2:])  # (b, 2n+1, 2n+1)
        n_roots, n_outcomes = roots.shape[0], outcomes_t.shape[0]
        frames = self._frames_by_sign(bloch_vectors_t)
        chunk = self.shots_per_chunk(n_qubits, n_roots, gamma.dtype, gamma.device)
        probabilities = []
        for start in range(0, n_outcomes, chunk):
            chunk_outcomes = outcomes_t[start : start + chunk]  # (c, n)
            n_chunk = chunk_outcomes.shape[0]
            # Rows ordered as (outcome, root), so that the result reshapes to (c, b).
            buffer = roots.repeat(n_chunk, 1, 1)  # (c * b, 2n+1, 2n+1)
            forced_bits = chunk_outcomes.repeat_interleave(n_roots, dim=0)  # (c * b, n)
            chunk_probabilities = torch.ones(n_chunk * n_roots, dtype=gamma.dtype, device=gamma.device)
            signs = torch.ones_like(chunk_probabilities)
            for k in range(n_qubits):
                probability_of_zero, block, rest = self._rotate(buffer, 2 * k, frames[k], signs)
                bit = forced_bits[:, k]
                probability, eigenvalue = self._project(
                    buffer[:, 2 * k + 2 :, 2 * k + 2 :], block, rest, bit, probability_of_zero
                )
                chunk_probabilities = chunk_probabilities * probability
                signs = signs * eigenvalue
            probabilities.append(chunk_probabilities.reshape((n_chunk,) + batch_shape))
        if not probabilities:
            result = torch.zeros((0,) + batch_shape, dtype=gamma.dtype, device=gamma.device)
        else:
            result = torch.cat(probabilities, dim=0)
        return convert_and_cast_like(result, extended_covariance_matrix)

    def sample(
        self,
        extended_covariance_matrix: TensorLike,
        bloch_vectors: TensorLike,
        shots: int,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Draw samples in the given product basis.

        The random numbers are drawn from the torch generator of the compute device.

        :param extended_covariance_matrix: The evolved extended covariance matrix,
            shape (..., 2n+1, 2n+1).
        :type extended_covariance_matrix: TensorLike
        :param bloch_vectors: The Bloch vector of the measurement basis of each qubit, shape (n, 3).
        :type bloch_vectors: TensorLike
        :param shots: The number of samples to draw.
        :type shots: int
        :param show_progress: Whether to display a progress bar over the shot chunks.
        :type show_progress: bool
        :return: Outcome bits, shape (shots, ..., n).
        :rtype: np.ndarray
        """
        gamma, bloch_vectors_t = self._prepare_inputs(extended_covariance_matrix, bloch_vectors)
        n_qubits = bloch_vectors_t.shape[0]
        batch_shape = tuple(gamma.shape[:-2])
        roots = gamma.reshape((-1,) + gamma.shape[-2:])  # (b, 2n+1, 2n+1)
        frames = self._frames_by_sign(bloch_vectors_t)
        chunk = self.shots_per_chunk(n_qubits, roots.shape[0], gamma.dtype, gamma.device)
        chunks = []
        for start in tqdm.trange(
            0, shots, chunk, desc=f"[{self.NAME}] Generating Samples", disable=not show_progress, unit="chunk"
        ):
            bits = self._sample_chunk(roots, frames, min(chunk, shots - start))  # (b, c, n)
            chunks.append(bits.transpose(0, 1).cpu().numpy())  # (c, b, n)
        if not chunks:
            return np.zeros((0,) + batch_shape + (n_qubits,), dtype=np.int64)
        samples = np.concatenate(chunks, axis=0)  # (shots, b, n)
        return samples.reshape((shots,) + batch_shape + (n_qubits,))

    def shots_per_chunk(self, n_qubits: int, batch_size: int, dtype: torch.dtype, device: torch.device) -> int:
        """Return the number of shots processed together.

        This is :attr:`shots_chunk_size` when it is set. Otherwise, the number of shots whose
        matrices, one :math:`(2n+1) \\times (2n+1)` matrix per shot and circuit batch element,
        fit in the memory budget, counting :attr:`MEMORY_OVERHEAD_FACTOR` for the temporaries.

        :param n_qubits: Number of qubits.
        :type n_qubits: int
        :param batch_size: Number of circuit batch elements.
        :type batch_size: int
        :param dtype: Floating point dtype of the computation.
        :type dtype: torch.dtype
        :param device: Torch device of the computation.
        :type device: torch.device
        :return: The number of shots per chunk, at least 1.
        :rtype: int
        """
        if self.shots_chunk_size is not None:
            return self.shots_chunk_size
        budget = self.memory_budget
        if budget is None:
            budget = self.DEFAULT_CPU_MEMORY_BUDGET
            if device.type == "cuda":  # pragma: no cover
                free_memory, _ = torch.cuda.mem_get_info(device)
                budget = int(self.DEFAULT_CUDA_MEMORY_FRACTION * free_memory)
        dim = 2 * n_qubits + 1
        bytes_per_shot = self.MEMORY_OVERHEAD_FACTOR * max(batch_size, 1) * dim * dim * torch.finfo(dtype).bits // 8
        return max(1, int(budget // bytes_per_shot))

    def _frames_by_sign(self, bloch_vectors: torch.Tensor) -> List[Optional[torch.Tensor]]:
        """Precompute the rotations of every qubit for both signs.

        :param bloch_vectors: Unit Bloch vectors, shape (n, 3).
        :type bloch_vectors: torch.Tensor
        :return: For each qubit, the rotations for the signs ``+1`` and ``-1``, shape (2, 3, 3),
            or None when the basis is the computational basis and the rotation is the identity.
        :rtype: List[Optional[torch.Tensor]]
        """
        signs = torch.tensor([1.0, -1.0], dtype=bloch_vectors.dtype, device=bloch_vectors.device)
        frames = self.measurement_frames(bloch_vectors[:, None, :], signs[None, :])  # (n, 2, 3, 3)
        is_computational = (torch.abs(bloch_vectors[:, 2] - 1.0) < 1e-12).tolist()
        return [None if is_identity else frames[k] for k, is_identity in enumerate(is_computational)]

    def _project(
        self,
        next_block: torch.Tensor,
        block: torch.Tensor,
        rest: torch.Tensor,
        bit: torch.Tensor,
        probability_of_zero: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project the matrices onto the measured outcomes, in place.

        :param next_block: The matrices of the unmeasured qubits and the ancilla, the ancilla
            last, shape (m, d - 2, d - 2). Updated in place.
        :type next_block: torch.Tensor
        :param block: The rotated triple block, in the order (ancilla, even, odd), shape (m, 3, 3).
        :type block: torch.Tensor
        :param rest: The rotated triple rows on the unmeasured qubits, shape (m, 3, d - 3).
        :type rest: torch.Tensor
        :param bit: The outcome bits, shape (m,).
        :type bit: torch.Tensor
        :param probability_of_zero: The probability of the outcome 0, shape (m,).
        :type probability_of_zero: torch.Tensor
        :return: The probability of each outcome and its eigenvalue, both of shape (m,).
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        eigenvalue = 1.0 - 2.0 * bit.to(next_block.dtype)
        probability = torch.where(bit == 0, probability_of_zero, 1.0 - probability_of_zero)
        # Impossible imposed outcomes have a zero probability already. Their matrices are left
        # without the update instead of dividing by zero.
        denominator = 2.0 * probability
        scale = torch.where(denominator > 0, eigenvalue / torch.where(denominator > 0, denominator, 1.0), 0.0)

        ancilla_row = rest[:, 0, :]  # (m, d - 3)
        next_block[:, -1, :-1] = ancilla_row
        next_block[:, :-1, -1] = -ancilla_row
        row_even = torch.cat([rest[:, 1, :], block[:, 1, :1]], dim=-1)  # (m, d - 2), ancilla last
        row_odd = torch.cat([rest[:, 2, :], block[:, 2, :1]], dim=-1)  # (m, d - 2)
        # Rank-2 update scale * (row_even row_odd^T - row_odd row_even^T).
        if next_block.is_cuda:  # pragma: no cover
            scaled_even = (scale[:, None] * row_even)[:, :, None]
            scaled_odd = (scale[:, None] * row_odd)[:, :, None]
            next_block.addcmul_(scaled_even, row_odd[:, None, :])
            next_block.addcmul_(scaled_odd, row_even[:, None, :], value=-1.0)
        else:
            left = scale[:, None, None] * torch.stack([row_even, row_odd], dim=-1)  # (m, d - 2, 2)
            right = torch.stack([row_odd, -row_even], dim=-2)  # (m, 2, d - 2)
            next_block.baddbmm_(left, right)
        return probability, eigenvalue

    def _rotate(
        self,
        buffer: torch.Tensor,
        start: int,
        frames: Optional[torch.Tensor],
        signs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Rotate the triple of the next qubit and read the probability of its outcome 0.

        :param buffer: The matrices, with the unmeasured block ``buffer[:, start:, start:]``,
            whose first two indices are the next qubit and last index is the ancilla.
        :type buffer: torch.Tensor
        :param start: Index of the first Majorana of the next qubit in the buffer.
        :type start: int
        :param frames: The rotations of the qubit for the signs ``+1`` and ``-1``, shape (2, 3, 3),
            or None for the identity.
        :type frames: Optional[torch.Tensor]
        :param signs: The signs of each matrix, shape (m,).
        :type signs: torch.Tensor
        :return: The probability of the outcome 0, shape (m,), the rotated triple block in the
            order (ancilla, even, odd), shape (m, 3, 3), and the rotated triple rows on the
            unmeasured qubits, shape (m, 3, d - 3).
        :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        """
        live = buffer[:, start:, start:]  # (m, d, d)
        dim = live.shape[-1]
        triple = [dim - 1, 0, 1]
        rows = live[:, triple, :]  # (m, 3, d), a copy
        block = rows[:, :, triple]  # (m, 3, 3)
        rest = rows[:, :, 2 : dim - 1]  # (m, 3, d - 3)
        if frames is not None:
            signed_frames = frames[(signs < 0).long()]  # (m, 3, 3)
            block = signed_frames @ block @ signed_frames.transpose(-1, -2)
            rest = signed_frames @ rest
        probability_of_zero = torch.clamp((1.0 - block[:, 1, 2]) / 2.0, 0.0, 1.0)
        return probability_of_zero, block, rest

    def _sample_chunk(self, roots: torch.Tensor, frames: List[Optional[torch.Tensor]], shots: int) -> torch.Tensor:
        """Draw one chunk of shots for every circuit batch element.

        :param roots: The extended covariance matrices, shape (b, 2n+1, 2n+1).
        :type roots: torch.Tensor
        :param frames: The rotations of each qubit, from :meth:`_frames_by_sign`.
        :type frames: List[Optional[torch.Tensor]]
        :param shots: The number of shots per batch element.
        :type shots: int
        :return: The outcome bits, shape (b, shots, n).
        :rtype: torch.Tensor
        """
        n_roots, n_qubits = roots.shape[0], len(frames)
        dtype, device = roots.dtype, roots.device
        total = n_roots * shots
        groups = torch.arange(n_roots, device=device)
        deduplicated = self.deduplicate and shots > 1
        if deduplicated:
            # One path per batch element, followed by all its shots.
            buffer = roots.clone()
            counts = torch.full((n_roots,), float(shots), dtype=dtype, device=device)
        else:
            buffer = roots.repeat_interleave(shots, dim=0)
            groups = groups.repeat_interleave(shots)
            counts = torch.ones(total, dtype=dtype, device=device)
        start = 0
        signs = torch.ones(buffer.shape[0], dtype=dtype, device=device)
        bits = torch.zeros(buffer.shape[0], n_qubits, dtype=torch.long, device=device)
        for k in range(n_qubits):
            probability_of_zero, block, rest = self._rotate(buffer, start, frames[k], signs)
            if not deduplicated:
                bit = (torch.rand(buffer.shape[0], dtype=dtype, device=device) >= probability_of_zero).long()
                _, eigenvalue = self._project(
                    buffer[:, start + 2 :, start + 2 :], block, rest, bit, probability_of_zero
                )
                bits[:, k] = bit
                signs = signs * eigenvalue
                start += 2
                continue
            # Split the shots of each path between the outcomes 0 and 1.
            zeros_count = torch.binomial(counts, probability_of_zero)
            ones_count = counts - zeros_count
            zero_paths = torch.nonzero(zeros_count > 0).flatten()
            one_paths = torch.nonzero(ones_count > 0).flatten()
            parents = torch.cat([zero_paths, one_paths])
            bit = torch.cat([torch.zeros_like(zero_paths), torch.ones_like(one_paths)])
            counts = torch.cat([zeros_count[zero_paths], ones_count[one_paths]])
            buffer = buffer[:, start + 2 :, start + 2 :][parents]  # (m', d - 2, d - 2), a copy
            start = 0
            _, eigenvalue = self._project(buffer, block[parents], rest[parents], bit, probability_of_zero[parents])
            signs = signs[parents] * eigenvalue
            groups = groups[parents]
            bits = bits[parents]
            bits[:, k] = bit
            if parents.shape[0] > self.DEDUPLICATION_THRESHOLD * total and k + 1 < n_qubits:
                # Too many distinct prefixes: every shot gets its own matrix.
                repeats = counts.long()
                buffer = buffer.repeat_interleave(repeats, dim=0, output_size=total)
                signs = signs.repeat_interleave(repeats, dim=0, output_size=total)
                groups = groups.repeat_interleave(repeats, dim=0, output_size=total)
                bits = bits.repeat_interleave(repeats, dim=0, output_size=total)
                counts = torch.ones(total, dtype=dtype, device=device)
                deduplicated = False
        if deduplicated:
            repeats = counts.long()
            groups = groups.repeat_interleave(repeats, dim=0, output_size=total)
            bits = bits.repeat_interleave(repeats, dim=0, output_size=total)
        if self.deduplicate and shots > 1:
            # The deduplicated shots come grouped by outcome. Shuffle them within each batch element.
            keys = groups.to(dtype) + torch.rand(total, dtype=dtype, device=device)
            bits = bits[torch.argsort(keys)]
        return bits.reshape(n_roots, shots, n_qubits)

    def _prepare_inputs(
        self,
        extended_covariance_matrix: TensorLike,
        bloch_vectors: TensorLike,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert the inputs to torch tensors on the compute device and check their shapes.

        :param extended_covariance_matrix: The extended covariance matrix, shape (..., 2n+1, 2n+1).
        :type extended_covariance_matrix: TensorLike
        :param bloch_vectors: The Bloch vectors, shape (n, 3).
        :type bloch_vectors: TensorLike
        :return: The covariance matrix and the normalized Bloch vectors, with the compute dtype
            and device.
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        :raises ValueError: If the shapes do not match or a Bloch vector is zero.
        """
        gamma = cast(torch.Tensor, to_tensor(detach(qml.math.real(extended_covariance_matrix)), dtype=None))
        dtype = self.dtype
        if dtype is None:
            dtype = gamma.dtype if gamma.dtype in (torch.float32, torch.float64) else torch.float64
        device = self.device if self.device is not None else gamma.device
        gamma = gamma.to(dtype=dtype, device=device)
        bloch_vectors_t = cast(torch.Tensor, to_tensor(detach(qml.math.real(bloch_vectors)), dtype=None))
        bloch_vectors_t = bloch_vectors_t.to(dtype=dtype, device=device)
        if bloch_vectors_t.ndim != 2 or bloch_vectors_t.shape[-1] != 3:
            raise ValueError(f"The Bloch vectors must have shape (n, 3). Got {tuple(bloch_vectors_t.shape)} instead.")
        expected_dim = 2 * bloch_vectors_t.shape[0] + 1
        if gamma.ndim < 2 or tuple(gamma.shape[-2:]) != (expected_dim, expected_dim):
            raise ValueError(
                f"The extended covariance matrix must have shape (..., {expected_dim}, {expected_dim}) for "
                f"{bloch_vectors_t.shape[0]} qubits. Got {tuple(gamma.shape)} instead."
            )
        norms = torch.linalg.norm(bloch_vectors_t, dim=-1, keepdim=True)
        if bool(torch.any(norms < 1e-12)):
            raise ValueError("The Bloch vectors must be non-zero.")
        return gamma, bloch_vectors_t / norms
