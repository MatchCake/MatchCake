import itertools
import numbers
import warnings
from typing import Iterable, List, Optional, Sized, Tuple, Union

import numpy as np
import pennylane as qml
import tqdm
from pennylane import numpy as pnp
from pennylane.typing import TensorLike
from scipy import sparse

from .. import utils


class NonInteractingFermionicLookupTable:
    """
    Lookup table for the non-interacting fermionic device.

    :param transition_matrix: The transition matrix of the device.
    :type transition_matrix: TensorLike
    :param show_progress: Whether to show progress bars.
    :type show_progress: bool

    The lookup table is a 3x3 matrix where the rows and columns are labeled by the following:
    - 0: c_d_alpha
    - 1: c_e_alpha
    - 2: c_2p_alpha_m1

    # TODO: Add more documentation.
    # TODO: Tips for optimization: Maybe there is a way to use the sparsity of the block diagonal matrix to reduce
    # TODO: the number of operations in the lookup table.
    """

    DEFAULT_CACHE_OBSERVABLES = False
    ALL_2D_INDEXES = np.asarray(list(itertools.product(range(3), repeat=2)))
    ALL_1D_INDEXES = utils.math.convert_2d_to_1d_indexes(ALL_2D_INDEXES, n_rows=3)

    def __init__(
        self,
        transition_matrix: TensorLike,
        *,
        cache_observables: bool = DEFAULT_CACHE_OBSERVABLES,
        **kwargs,
    ):
        self._transition_matrix = transition_matrix
        self._block_diagonal_matrix = utils.get_block_diagonal_matrix(self.n_particles)
        self.cache_observables = cache_observables

        # Entries of the lookup table
        self._c_d_alpha__c_d_beta = None
        self._c_d_alpha__c_e_beta = None
        self._c_d_alpha__c_2p_beta_m1 = None
        self._c_e_alpha__c_d_beta = None
        self._c_e_alpha__c_e_beta = None
        self._c_e_alpha__c_2p_beta_m1 = None
        self._c_2p_alpha_m1__c_d_beta = None
        self._c_2p_alpha_m1__c_e_beta = None
        self._c_2p_alpha_m1__c_2p_beta_m1 = None

        self._block_bm_transition_transpose_matrix = None
        self._block_bm_transition_dagger_matrix = None
        self._transition_bm_block_matrix = None
        self._stacked_items = None

        self._observables = {}
        self.p_bar = None
        self.show_progress = kwargs.get("show_progress", False)

    @property
    def memory_usage(self):
        """
        Compute the memory usage of the lookup table in bytes.
        """
        size = qml.math.prod(qml.math.shape(self.transition_matrix))
        mem = size * self.transition_matrix.dtype.itemsize
        tensors = [
            self._c_d_alpha__c_d_beta,
            self._c_d_alpha__c_e_beta,
            self._c_d_alpha__c_2p_beta_m1,
            self._c_e_alpha__c_d_beta,
            self._c_e_alpha__c_e_beta,
            self._c_e_alpha__c_2p_beta_m1,
            self._c_2p_alpha_m1__c_d_beta,
            self._c_2p_alpha_m1__c_e_beta,
            self._c_2p_alpha_m1__c_2p_beta_m1,
        ]
        tensors = [t for t in tensors if t is not None]
        mem += sum([qml.math.prod(qml.math.shape(t)) * t.dtype.itemsize for t in tensors])
        return mem

    @property
    def memory_usage_in_gb(self):
        return self.memory_usage / 1e9

    @property
    def transition_matrix(self):
        return self._transition_matrix

    @property
    def n_particles(self):
        return qml.math.shape(self.transition_matrix)[-2]

    @property
    def batch_size(self):
        if qml.math.ndim(self.transition_matrix) < 3:
            return 0
        return qml.math.shape(self.transition_matrix)[0]

    @property
    def block_diagonal_matrix(self):
        if self._block_diagonal_matrix is None:
            self._block_diagonal_matrix = utils.get_block_diagonal_matrix(self.n_particles)
        return self._block_diagonal_matrix

    @property
    def shape(self) -> Tuple[int, int]:
        return 3, 3

    @property
    def c_d_alpha__c_d_beta(self):
        if self._c_d_alpha__c_d_beta is None:
            self._c_d_alpha__c_d_beta = self._compute_c_d_alpha__c_d_beta()
        return self._c_d_alpha__c_d_beta

    @property
    def c_d_alpha__c_e_beta(self):
        if self._c_d_alpha__c_e_beta is None:
            self._c_d_alpha__c_e_beta = self._compute_c_d_alpha__c_e_beta()
        return self._c_d_alpha__c_e_beta

    @property
    def c_d_alpha__c_2p_beta_m1(self):
        if self._c_d_alpha__c_2p_beta_m1 is None:
            self._c_d_alpha__c_2p_beta_m1 = self._compute_c_d_alpha__c_2p_beta_m1()
        return self._c_d_alpha__c_2p_beta_m1

    @property
    def c_e_alpha__c_d_beta(self):
        if self._c_e_alpha__c_d_beta is None:
            self._c_e_alpha__c_d_beta = self._compute_c_e_alpha__c_d_beta()
        return self._c_e_alpha__c_d_beta

    @property
    def c_e_alpha__c_e_beta(self):
        if self._c_e_alpha__c_e_beta is None:
            self._c_e_alpha__c_e_beta = self._compute_c_e_alpha__c_e_beta()
        return self._c_e_alpha__c_e_beta

    @property
    def c_e_alpha__c_2p_beta_m1(self):
        if self._c_e_alpha__c_2p_beta_m1 is None:
            self._c_e_alpha__c_2p_beta_m1 = self._compute_c_e_alpha__c_2p_beta_m1()
        return self._c_e_alpha__c_2p_beta_m1

    @property
    def c_2p_alpha_m1__c_d_beta(self):
        if self._c_2p_alpha_m1__c_d_beta is None:
            self._c_2p_alpha_m1__c_d_beta = self._compute_c_2p_alpha_m1__c_d_beta()
        return self._c_2p_alpha_m1__c_d_beta

    @property
    def c_2p_alpha_m1__c_e_beta(self):
        if self._c_2p_alpha_m1__c_e_beta is None:
            self._c_2p_alpha_m1__c_e_beta = self._compute_c_2p_alpha_m1__c_e_beta()
        return self._c_2p_alpha_m1__c_e_beta

    @property
    def c_2p_alpha_m1__c_2p_beta_m1(self):
        if self._c_2p_alpha_m1__c_2p_beta_m1 is None:
            self._c_2p_alpha_m1__c_2p_beta_m1 = self._compute_c_2p_alpha_m1__c_2p_beta_m1()
        return self._c_2p_alpha_m1__c_2p_beta_m1

    @property
    def block_bm_transition_transpose_matrix(self):
        if self._block_bm_transition_transpose_matrix is None:
            self._block_bm_transition_transpose_matrix = self._compute_block_bm_transition_transpose_matrix_()
        return self._block_bm_transition_transpose_matrix

    @property
    def block_bm_transition_dagger_matrix(self):
        if self._block_bm_transition_dagger_matrix is None:
            self._block_bm_transition_dagger_matrix = self._compute_block_bm_transition_dagger_matrix_()
        return self._block_bm_transition_dagger_matrix

    @property
    def transition_bm_block_matrix(self):
        if self._transition_bm_block_matrix is None:
            self._transition_bm_block_matrix = self._compute_transition_bm_block_matrix_()
        return self._transition_bm_block_matrix

    @property
    def getter_table(self) -> List[List[callable]]:
        return [
            [
                self.get_c_d_alpha__c_d_beta,
                self.get_c_d_alpha__c_e_beta,
                self.get_c_d_alpha__c_2p_beta_m1,
            ],
            [
                self.get_c_e_alpha__c_d_beta,
                self.get_c_e_alpha__c_e_beta,
                self.get_c_e_alpha__c_2p_beta_m1,
            ],
            [
                self.get_c_2p_alpha_m1__c_d_beta,
                self.get_c_2p_alpha_m1__c_e_beta,
                self.get_c_2p_alpha_m1__c_2p_beta_m1,
            ],
        ]

    @property
    def stacked_items(self):
        if self._stacked_items is None:
            self._stacked_items = self.compute_stack_and_pad_items(self.ALL_2D_INDEXES, close_p_bar=False)
        return self._stacked_items

    def _compute_block_bm_transition_transpose_matrix_(self):
        self.p_bar_set_postfix_str("Computing BT^T matrix.")
        return qml.math.einsum(f"ij,...kj->...ik", self.block_diagonal_matrix, self.transition_matrix)

    def _compute_block_bm_transition_dagger_matrix_(self):
        self.p_bar_set_postfix_str("Computing BT^dagger matrix.")
        return qml.math.einsum(
            f"ij,...kj->...ik",
            self.block_diagonal_matrix,
            qml.math.conjugate(self.transition_matrix),
        )

    def _compute_transition_bm_block_matrix_(self):
        self.p_bar_set_postfix_str("Computing TB matrix.")
        return qml.math.einsum(f"...ij,jk->...ik", self.transition_matrix, self.block_diagonal_matrix)

    def _compute_c_d_alpha__c_d_beta(self):
        self.p_bar_set_postfix_str("Computing c_d_alpha__c_d_beta.")
        return qml.math.einsum(
            f"...pj,...kj->...pk",
            self.transition_bm_block_matrix,
            self.transition_matrix,
        )

    def _compute_c_d_alpha__c_e_beta(self):
        self.p_bar_set_postfix_str("Computing c_d_alpha__c_e_beta.")
        return qml.math.einsum(
            f"...pj,...kj->...pk",
            self.transition_bm_block_matrix,
            qml.math.conjugate(self.transition_matrix),
        )

    def _compute_c_d_alpha__c_2p_beta_m1(self):
        self.p_bar_set_postfix_str("Computing c_d_alpha__c_2p_beta_m1.")
        return self.transition_bm_block_matrix

    def _compute_c_e_alpha__c_d_beta(self):
        self.p_bar_set_postfix_str("Computing c_e_alpha__c_d_beta.")
        return qml.math.einsum(
            f"...pi,...ik->...pk",
            qml.math.conjugate(self._transition_matrix),
            self.block_bm_transition_transpose_matrix,
        )

    def _compute_c_e_alpha__c_e_beta(self):
        self.p_bar_set_postfix_str("Computing c_e_alpha__c_e_beta.")
        return qml.math.einsum(
            f"...pi,...ik->...pk",
            qml.math.conjugate(self._transition_matrix),
            self.block_bm_transition_dagger_matrix,
        )

    def _compute_c_e_alpha__c_2p_beta_m1(self):
        self.p_bar_set_postfix_str("Computing c_e_alpha__c_2p_beta_m1.")
        return qml.math.einsum(
            f"...pi,ij->...pj",
            qml.math.conjugate(self._transition_matrix),
            self.block_diagonal_matrix,
        )

    def _compute_c_2p_alpha_m1__c_d_beta(self):
        self.p_bar_set_postfix_str("Computing c_2p_alpha_m1__c_d_beta.")
        return self.block_bm_transition_transpose_matrix

    def _compute_c_2p_alpha_m1__c_e_beta(self):
        self.p_bar_set_postfix_str("Computing c_2p_alpha_m1__c_e_beta.")
        return self.block_bm_transition_dagger_matrix

    def _compute_c_2p_alpha_m1__c_2p_beta_m1(self):
        self.p_bar_set_postfix_str("Computing c_2p_alpha_m1__c_2p_beta_m1.")
        if self.batch_size > 0:
            size = qml.math.shape(self.transition_matrix)[-1]
            shape = ([self.batch_size] if self.batch_size else []) + [size, size]
            matrix = pnp.zeros(shape, dtype=complex)
            matrix[..., :, :] = qml.math.eye(size, dtype=complex)
        else:
            matrix = np.eye(self.transition_matrix.shape[-1])
        matrix = qml.math.convert_like(matrix, self.transition_matrix)
        return matrix

    def get_c_d_alpha__c_d_beta(self):
        return self.c_d_alpha__c_d_beta

    def get_c_d_alpha__c_e_beta(self):
        return self.c_d_alpha__c_e_beta

    def get_c_d_alpha__c_2p_beta_m1(self):
        return self.c_d_alpha__c_2p_beta_m1

    def get_c_e_alpha__c_d_beta(self):
        return self.c_e_alpha__c_d_beta

    def get_c_e_alpha__c_e_beta(self):
        return self.c_e_alpha__c_e_beta

    def get_c_e_alpha__c_2p_beta_m1(self):
        return self.c_e_alpha__c_2p_beta_m1

    def get_c_2p_alpha_m1__c_d_beta(self):
        return self.c_2p_alpha_m1__c_d_beta

    def get_c_2p_alpha_m1__c_e_beta(self):
        return self.c_2p_alpha_m1__c_e_beta

    def get_c_2p_alpha_m1__c_2p_beta_m1(self):
        return self.c_2p_alpha_m1__c_2p_beta_m1

    def __getitem__(self, item: Union[Tuple[int, int], int]):
        if isinstance(item, int):
            item = utils.math.convert_1d_to_2d_indexes([item], n_rows=3)[0]
        i, j = item
        getter = self.getter_table[i][j]
        return getter()

    def compute_items(self, indexes: Iterable[Tuple[int, int]], close_p_bar: bool = True) -> List[TensorLike]:
        """
        Compute the items of the lookup table corresponding to the indexes.

        :param indexes: Indexes of the items to compute.
        :param close_p_bar: Whether to close the progress bar.
        :return: The items of the lookup table corresponding to the indexes.
        """
        self.initialize_p_bar(total=len(indexes), initial=0, desc="Computing Lookup Table Items")
        items = []
        indexes = np.asarray(indexes)
        indexes = indexes.reshape(-1, 2)
        for i, j in indexes:
            items.append(self[i, j])
            self.update_p_bar()
        if close_p_bar:
            self.close_p_bar()
        return items

    def compute_stack_and_pad_items(
        self,
        indexes: Iterable[Tuple[int, int]],
        pad_value: numbers.Number = 0.0,
        close_p_bar: bool = True,
    ) -> TensorLike:
        items = self.compute_items(indexes, close_p_bar=close_p_bar)
        items_shapes = [qml.math.shape(i) for i in items]
        items_has_same_shape = all([i == items_shapes[0] for i in items_shapes])

        if items_has_same_shape:
            items = qml.math.stack(items)
        else:
            # need to pad the items to max dim in each dimension and stack them
            max_dim_0, max_dim_1 = max([i[-2] for i in items_shapes]), max([i[-1] for i in items_shapes])
            for i, (item, item_shape) in enumerate(zip(items, items_shapes)):
                new_shape = list(item_shape)
                new_shape[-2] = max_dim_0
                new_shape[-1] = max_dim_1
                new_item = qml.math.convert_like(np.full(new_shape, fill_value=pad_value, dtype=complex), item)
                new_item[..., : item_shape[-2], : item_shape[-1]] = item
                items[i] = new_item
            items = qml.math.stack(items)
        return items

    def get_observable(self, k: int, system_state: np.ndarray) -> np.ndarray:
        r"""
        TODO: change k to y* or wires
        Get the observable corresponding to the index k and the state.

        :param k: Index of the observable
        :type k: int
        :param system_state: State of the system
        :type system_state: np.ndarray
        :return: The observable of shape (2(h + k), 2(h + k)) where h is the hamming weight of the state.
        :rtype: np.ndarray
        """
        warnings.warn(
            "This method is deprecated. Use get_observable_of_target_state instead.",
            DeprecationWarning,
        )
        key = (k, utils.state_to_binary_string(system_state, n=self.n_particles))
        if key not in self._observables:
            self._observables[key] = self._compute_observable(k, system_state)
        return self._observables[key]

    def get_observable_of_target_state(
        self,
        system_state: Union[int, np.ndarray, sparse.sparray],
        target_binary_state: Optional[np.ndarray] = None,
        indexes_of_target_state: Optional[np.ndarray] = None,
        **kwargs,
    ) -> TensorLike:
        r"""
        Get the observable corresponding to target_binary_state and the system_state.

        :param system_state: State of the system
        :type system_state: Union[int, np.ndarray, sparse.sparray]
        :param target_binary_state: Target state of the system
        :type target_binary_state: Optional[np.ndarray]
        :param indexes_of_target_state: Indexes of the target state of the system
        :type indexes_of_target_state: Optional[np.ndarray]
        :return: The observable of shape (2(h + k), 2(h + k)) where h is the hamming weight of the system state.
        :rtype: np.ndarray
        """
        self.show_progress = kwargs.get("show_progress", self.show_progress)
        if not self.cache_observables:
            return self.compute_observable_of_target_state(
                system_state, target_binary_state, indexes_of_target_state, **kwargs
            )

        key = (
            utils.state_to_binary_string(system_state, n=self.n_particles),
            "".join([str(i) for i in target_binary_state]),
            ",".join([str(i) for i in indexes_of_target_state]),
        )
        if key not in self._observables:
            self._observables[key] = self.compute_observable_of_target_state(
                system_state, target_binary_state, indexes_of_target_state, **kwargs
            )
        return self._observables[key]

    def get_observables_of_target_states(
        self,
        system_state: Union[int, np.ndarray, sparse.sparray],
        target_binary_states: Optional[np.ndarray] = None,
        indexes_of_target_states: Optional[np.ndarray] = None,
        **kwargs,
    ) -> TensorLike:
        r"""
        Get the observable corresponding to target_binary_state and the system_state.

        :param system_state: State of the system
        :type system_state: Union[int, np.ndarray, sparse.sparray]
        :param target_binary_states: Target state of the system
        :type target_binary_states: Optional[np.ndarray]
        :param indexes_of_target_states: Indexes of the target state of the system
        :type indexes_of_target_states: Optional[np.ndarray]
        :return: The observable of shape (2(h + k), 2(h + k)) where h is the hamming weight of the system state.
        :rtype: np.ndarray
        """
        self.show_progress = kwargs.get("show_progress", self.show_progress)
        if not self.cache_observables:
            return self.compute_observables_of_target_states(
                system_state, target_binary_states, indexes_of_target_states, **kwargs
            )

        all_keys = []
        keys_to_compute = []
        target_binary_states_to_compute, indexes_of_target_states_to_compute = [], []
        for target_binary_state, indexes_of_target_state in zip(target_binary_states, indexes_of_target_states):
            key = (
                utils.state_to_binary_string(system_state, n=self.n_particles),
                "".join([str(i) for i in target_binary_state]),
                ",".join([str(i) for i in indexes_of_target_state]),
            )
            all_keys.append(key)
            if key not in self._observables:
                keys_to_compute.append(key)
                target_binary_states_to_compute.append(target_binary_state)
                indexes_of_target_states_to_compute.append(indexes_of_target_state)

        raise NotImplementedError("This setting is not implemented yet.")

        all_obs = []
        for key in all_keys:
            all_obs.append(self._observables[key])
        return qml.math.stack(all_obs)

    def _compute_observable(self, k: int, system_state: Union[int, np.ndarray, sparse.sparray]) -> np.ndarray:
        warnings.warn(
            "This method is deprecated. Use compute_observable_of_target_states instead.",
            DeprecationWarning,
        )
        ket_majorana_indexes = utils.decompose_binary_state_into_majorana_indexes(system_state)
        bra_majorana_indexes = list(reversed(ket_majorana_indexes))

        unmeasured_cls_indexes = [2 for _ in range(len(ket_majorana_indexes))]
        measure_cls_indexes = np.array([[1, 0] for _ in range(k + 1)]).flatten().tolist()
        lt_indexes = unmeasured_cls_indexes + measure_cls_indexes + unmeasured_cls_indexes

        # measure_indexes = np.array([[i, i] for i in range(k+1)]).flatten().tolist()
        measure_indexes = [k, k]
        majorana_indexes = list(bra_majorana_indexes) + measure_indexes + list(ket_majorana_indexes)

        obs_size = len(majorana_indexes)
        obs_shape = ([self.batch_size] if self.batch_size else []) + [
            obs_size,
            obs_size,
        ]
        obs = np.zeros(obs_shape, dtype=complex)
        for i, j in zip(*np.triu_indices(obs_size, k=1)):
            i_k, j_k = majorana_indexes[i], majorana_indexes[j]
            row, col = lt_indexes[i], lt_indexes[j]
            obs[..., i, j] = self[row, col][..., i_k, j_k]
        obs = obs - qml.math.swapaxes(obs, -2, -1)
        return obs

    def compute_observable_of_target_state(
        self,
        system_state: Union[int, np.ndarray, sparse.sparray],
        target_binary_state: Optional[np.ndarray] = None,
        indexes_of_target_state: Optional[np.ndarray] = None,
        **kwargs,
    ) -> TensorLike:
        observables = self.compute_observables_of_target_states(
            system_state=system_state,
            target_binary_states=target_binary_state,
            indexes_of_target_states=indexes_of_target_state,
            **kwargs,
        )
        if len(qml.math.shape(observables)) > 2 and not self.batch_size:
            observables = qml.math.reshape(observables, (-1, *qml.math.shape(observables)[-2:]))[0]
        elif len(qml.math.shape(observables)) > 3:
            observables = qml.math.reshape(observables, (-1, *qml.math.shape(observables)[-3:]))[0]
        return observables

    def assert_binary(self, binary_state: np.ndarray) -> bool:
        r"""
        Check if the binary state contains only zeros or ones. If not, a value error will be raised.

        :param binary_state: Input binary state.
        :type binary_state: np.ndarray

        :return: Whether the input state is binary or not.
        :rtype: bool

        :raises: ValueError
        """
        unique_values = np.unique(binary_state)
        if len(unique_values) > 2 or not np.all(np.isin(unique_values, [0, 1])):
            raise ValueError(f"The binary state must contain only zeros and ones. Currently contains: {unique_values}")
        return True

    def _setup_inputs(
        self,
        system_state: Union[int, np.ndarray, sparse.sparray],
        target_binary_states: Optional[np.ndarray] = None,
        indexes_of_target_states: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        r"""
        Setup the inputs for the compute_observables_of_target_states method.
        This includes reshaping the inputs and checking their validity.

        :param system_state: The state of the system.
        :type system_state: Union[int, np.ndarray, sparse.sparray]
        :param target_binary_states: The target states to compute the probability from.
        :type target_binary_states: Optional[np.ndarray]
        :param indexes_of_target_states: The index of the particles of the target states.
        :type indexes_of_target_states: Optional[np.ndarray]
        :param kwargs: Additional keywords arguments

        :return: The processed inputs: (system_state, target_binary_states, indexes_of_target_states, initial_ndim)
        :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray, int]
        """
        initial_ndim = None
        if target_binary_states is not None:
            target_binary_states = np.asarray(target_binary_states)
            initial_ndim = len(qml.math.shape(target_binary_states))
            target_binary_states = target_binary_states.reshape(-1, target_binary_states.shape[-1])

        if indexes_of_target_states is not None:
            indexes_of_target_states = np.asarray(indexes_of_target_states)
            initial_ndim = len(qml.math.shape(indexes_of_target_states))
            indexes_of_target_states = indexes_of_target_states.reshape(-1, indexes_of_target_states.shape[-1])

        if target_binary_states is None and indexes_of_target_states is None:
            target_binary_states = np.array(
                [
                    [
                        1,
                    ]
                ]
            )
            indexes_of_target_states = np.array(
                [
                    [
                        0,
                    ]
                ]
            )
        elif target_binary_states is not None and indexes_of_target_states is None:
            indexes_of_target_states = np.arange(target_binary_states.shape[-1], dtype=int)[np.newaxis, :].repeat(
                target_binary_states.shape[0], axis=0
            )
        elif target_binary_states is None and indexes_of_target_states is not None:
            target_binary_states = np.ones(indexes_of_target_states.shape[-1], dtype=int)[np.newaxis, :].repeat(
                indexes_of_target_states.shape[0], axis=0
            )

        self.assert_binary(system_state)
        self.assert_binary(target_binary_states)
        return (
            system_state,
            target_binary_states,
            indexes_of_target_states,
            initial_ndim,
        )

    def _get_bra_ket_indexes(self, system_state: np.ndarray, target_binary_states: np.ndarray, **kwargs):
        target_batch_size = target_binary_states.shape[0]
        ket_majorana_indexes = utils.decompose_binary_state_into_majorana_indexes(system_state)
        bra_majorana_indexes = list(reversed(ket_majorana_indexes))

        ket_majorana_indexes = np.asarray(ket_majorana_indexes)[np.newaxis, :].repeat(target_batch_size, axis=0)
        bra_majorana_indexes = np.asarray(bra_majorana_indexes)[np.newaxis, :].repeat(target_batch_size, axis=0)
        return bra_majorana_indexes, ket_majorana_indexes

    def _get_lt_indexes(
        self,
        target_binary_states: np.ndarray,
        ket_majorana_indexes: np.ndarray,
    ):
        target_batch_size = target_binary_states.shape[0]
        unmeasured_cls_indexes = [2 for _ in range(ket_majorana_indexes.shape[-1])]
        unmeasured_cls_indexes = np.asarray(unmeasured_cls_indexes)[np.newaxis, :].repeat(target_batch_size, axis=0)
        measure_cls_indexes = np.asarray([np.array([[b, 1 - b] for b in t]).flatten() for t in target_binary_states])
        lt_indexes = np.concatenate(
            [unmeasured_cls_indexes, measure_cls_indexes, unmeasured_cls_indexes],
            axis=-1,
        )
        return lt_indexes.astype(int)

    def _get_majorana_indexes(
        self,
        indexes_of_target_states: np.ndarray,
        bra_majorana_indexes: np.ndarray,
        ket_majorana_indexes: np.ndarray,
    ):
        measure_indexes = np.asarray([np.array([[i, i] for i in t]).flatten() for t in indexes_of_target_states])
        majorana_indexes = np.concatenate(
            [bra_majorana_indexes, measure_indexes, ket_majorana_indexes], axis=-1
        ).astype(int)
        return majorana_indexes

    def convert_2d_indexes_to_1d_indexes(
        self,
        all_indexes,
        unique_indexes,
    ):
        new_all_indexes = np.empty(all_indexes.shape[:-1], dtype=int)
        for i, (r, c) in enumerate(unique_indexes):
            mask = np.isclose(all_indexes, [r, c]).all(axis=-1)
            new_all_indexes = np.where(mask, i, new_all_indexes)
        return new_all_indexes

    def extend_unique_indexes_to_all_indexes(
        self,
        all_indexes,
        unique_indexes,
    ):
        new_all_indexes = np.empty(all_indexes.shape, dtype=int)
        for i, j in enumerate(unique_indexes):
            new_all_indexes = np.where(all_indexes == j, i, new_all_indexes)
        return new_all_indexes

    def compute_observables_of_target_states(
        self,
        system_state: Union[int, np.ndarray, sparse.sparray],
        target_binary_states: Optional[np.ndarray] = None,
        indexes_of_target_states: Optional[np.ndarray] = None,
        **kwargs,
    ) -> TensorLike:
        system_state, target_binary_states, indexes_of_target_states, initial_ndim = self._setup_inputs(
            system_state, target_binary_states, indexes_of_target_states, **kwargs
        )
        target_batch_size = target_binary_states.shape[0]
        bra_majorana_indexes, ket_majorana_indexes = self._get_bra_ket_indexes(system_state, target_binary_states)
        lt_indexes = self._get_lt_indexes(target_binary_states, ket_majorana_indexes)
        majorana_indexes = self._get_majorana_indexes(
            indexes_of_target_states, bra_majorana_indexes, ket_majorana_indexes
        )

        obs_size = majorana_indexes.shape[-1]
        obs_indices = np.stack(np.triu_indices(obs_size, k=1))
        lt_item_rows, lt_item_cols = (
            majorana_indexes[..., obs_indices[0]],
            majorana_indexes[..., obs_indices[1]],
        )

        # compute items needed for the observable
        all_lt_indexes = np.stack((lt_indexes[..., obs_indices[0]], lt_indexes[..., obs_indices[1]]), axis=-1)
        all_lt_indexes_raveled = all_lt_indexes.reshape(-1, 2)

        all_lt_indexes_raveled_1d = utils.math.convert_2d_to_1d_indexes(all_lt_indexes_raveled, n_rows=3)
        all_lt_indexes_1d = all_lt_indexes_raveled_1d.reshape(*all_lt_indexes.shape[:-1])
        unique_lt_indexes_raveled_1d = np.fromiter(set(all_lt_indexes_raveled_1d), dtype=int)
        unique_lt_indexes_raveled_2d = utils.math.convert_1d_to_2d_indexes(unique_lt_indexes_raveled_1d, n_rows=3)

        lt_items = self.compute_stack_and_pad_items(unique_lt_indexes_raveled_2d, close_p_bar=False)
        new_all_lt_indexes = self.extend_unique_indexes_to_all_indexes(all_lt_indexes_1d, unique_lt_indexes_raveled_1d)
        lt_items = lt_items.reshape(lt_items.shape[0], -1, *lt_items.shape[-2:])

        batch_size = [self.batch_size] if self.batch_size else [1]
        obs_shape = [target_batch_size] + batch_size + [obs_size, obs_size]
        obs = qml.math.convert_like(np.zeros(obs_shape, dtype=complex), self.transition_matrix)
        obs[..., obs_indices[0], obs_indices[1]] = qml.math.transpose(
            lt_items[new_all_lt_indexes, ..., lt_item_rows, lt_item_cols], (0, -1, -2)
        )

        self.p_bar_set_postfix_str("Finishing the computation of the observable.")
        obs = obs - qml.math.einsum("...ij->...ji", obs)
        if self.batch_size is None or self.batch_size == 0:
            obs = obs[:, 0, ...]
        if initial_ndim == 1:
            obs = obs[0, ...]
        self.p_bar_set_postfix_str("Finished the computation of the observable.")
        self.close_p_bar()
        return obs

    def update_p_bar(self, *args, **kwargs):
        if self.p_bar is None:
            return
        self.p_bar.update(*args, **kwargs)
        self.p_bar.refresh()

    def p_bar_set_n(self, n: int):
        if self.p_bar is not None:
            self.p_bar.n = n
            self.p_bar.refresh()

    def initialize_p_bar(self, *args, **kwargs):
        kwargs.setdefault("disable", not self.show_progress)
        if self.p_bar is None and not self.show_progress:
            return
        self.p_bar = tqdm.tqdm(*args, **kwargs)
        return self.p_bar

    def p_bar_set_postfix(self, *args, **kwargs):
        if self.p_bar is not None:
            self.p_bar.set_postfix(*args, **kwargs)
            self.p_bar.refresh()

    def p_bar_set_postfix_str(self, *args, **kwargs):
        if self.p_bar is not None:
            self.p_bar.set_postfix_str(*args, **kwargs)
            self.p_bar.refresh()

    def close_p_bar(self):
        if self.p_bar is not None:
            self.p_bar.close()

    def compute_pfaffian_of_target_states(
        self,
        system_state: Union[int, np.ndarray, sparse.sparray],
        target_binary_states: Optional[np.ndarray] = None,
        indexes_of_target_states: Optional[np.ndarray] = None,
        **kwargs,
    ) -> TensorLike:
        system_state, target_binary_states, indexes_of_target_states, initial_ndim = self._setup_inputs(
            system_state, target_binary_states, indexes_of_target_states, **kwargs
        )
        target_batch_size = target_binary_states.shape[0]
        bra_majorana_indexes, ket_majorana_indexes = self._get_bra_ket_indexes(system_state, target_binary_states)
        lt_indexes = self._get_lt_indexes(target_binary_states, ket_majorana_indexes)
        majorana_indexes = self._get_majorana_indexes(
            indexes_of_target_states, bra_majorana_indexes, ket_majorana_indexes
        )

        obs_size = majorana_indexes.shape[-1]
        obs_indices = np.stack(np.triu_indices(obs_size, k=1))
        lt_item_rows, lt_item_cols = (
            majorana_indexes[..., obs_indices[0]],
            majorana_indexes[..., obs_indices[1]],
        )

        # compute items needed for the observable
        all_lt_indexes = np.stack((lt_indexes[..., obs_indices[0]], lt_indexes[..., obs_indices[1]]), axis=-1)
        all_lt_indexes_raveled = all_lt_indexes.reshape(-1, 2)

        all_lt_indexes_raveled_1d = utils.math.convert_2d_to_1d_indexes(all_lt_indexes_raveled, n_rows=3)
        all_lt_indexes_1d = all_lt_indexes_raveled_1d.reshape(*all_lt_indexes.shape[:-1])
        unique_lt_indexes_raveled_1d = np.fromiter(set(all_lt_indexes_raveled_1d), dtype=int)
        unique_lt_indexes_raveled_2d = utils.math.convert_1d_to_2d_indexes(unique_lt_indexes_raveled_1d, n_rows=3)
        lt_items = self.compute_stack_and_pad_items(unique_lt_indexes_raveled_2d, close_p_bar=False)
        new_all_lt_indexes = self.extend_unique_indexes_to_all_indexes(all_lt_indexes_1d, unique_lt_indexes_raveled_1d)
        lt_items = lt_items.reshape(lt_items.shape[0], -1, *lt_items.shape[-2:])

        batch_size = [self.batch_size] if self.batch_size else [1]
        obs_shape = [target_batch_size] + batch_size + [obs_size, obs_size]
        obs = qml.math.convert_like(np.zeros(obs_shape, dtype=complex), self.transition_matrix)
        obs[..., obs_indices[0], obs_indices[1]] = qml.math.transpose(
            lt_items[new_all_lt_indexes, ..., lt_item_rows, lt_item_cols], (0, -1, -2)
        )

        self.p_bar_set_postfix_str("Finishing the computation of the observable.")
        obs = obs - qml.math.einsum("...ij->...ji", obs)
        if self.batch_size is None or self.batch_size == 0:
            obs = obs[:, 0, ...]
        if initial_ndim == 1:
            obs = obs[0, ...]
        self.p_bar_set_postfix_str("Finished the computation of the observable.")
        self.close_p_bar()
        return qml.math.real(utils.pfaffian(obs, method="PfaffianFDBPf", show_progress=self.show_progress))
