import numbers
import warnings
from typing import Optional, Tuple, Union, Iterable, List, Sized

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from scipy import sparse
import tqdm

from .. import utils
from ..templates import TensorLike


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
    def __init__(
            self,
            transition_matrix: TensorLike,
            **kwargs
    ):
        self._transition_matrix = transition_matrix
        self._block_diagonal_matrix = utils.get_block_diagonal_matrix(self.n_particles)
        
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
        
        self._observables = {}
        self.p_bar = None
        self.show_progress = kwargs.get("show_progress", False)
    
    @property
    def memory_usage(self):
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
            [self.get_c_d_alpha__c_d_beta, self.get_c_d_alpha__c_e_beta, self.get_c_d_alpha__c_2p_beta_m1],
            [self.get_c_e_alpha__c_d_beta, self.get_c_e_alpha__c_e_beta, self.get_c_e_alpha__c_2p_beta_m1],
            [self.get_c_2p_alpha_m1__c_d_beta, self.get_c_2p_alpha_m1__c_e_beta, self.get_c_2p_alpha_m1__c_2p_beta_m1],
        ]

    def _compute_block_bm_transition_transpose_matrix_(self):
        self.p_bar_set_postfix_str("Computing BT^T matrix.")
        return qml.math.einsum(
            f"ij,...kj->...ik",
            self.block_diagonal_matrix,
            self.transition_matrix
        )

    def _compute_block_bm_transition_dagger_matrix_(self):
        self.p_bar_set_postfix_str("Computing BT^dagger matrix.")
        return qml.math.einsum(
            f"ij,...kj->...ik",
            self.block_diagonal_matrix,
            qml.math.conjugate(self.transition_matrix)
        )

    def _compute_transition_bm_block_matrix_(self):
        self.p_bar_set_postfix_str("Computing TB matrix.")
        return qml.math.einsum(
            f"...ij,jk->...ik",
            self.transition_matrix,
            self.block_diagonal_matrix
        )
    
    def _compute_c_d_alpha__c_d_beta(self):
        self.p_bar_set_postfix_str("Computing c_d_alpha__c_d_beta.")
        return qml.math.einsum(
            f"...pj,...kj->...pk",
            self.transition_bm_block_matrix,
            self.transition_matrix
        )

    def _compute_c_d_alpha__c_e_beta(self):
        self.p_bar_set_postfix_str("Computing c_d_alpha__c_e_beta.")
        return qml.math.einsum(
            f"...pj,...kj->...pk",
            self.transition_bm_block_matrix,
            qml.math.conjugate(self.transition_matrix)
        )
    
    def _compute_c_d_alpha__c_2p_beta_m1(self):
        self.p_bar_set_postfix_str("Computing c_d_alpha__c_2p_beta_m1.")
        return self.transition_bm_block_matrix

    def _compute_c_e_alpha__c_d_beta(self):
        self.p_bar_set_postfix_str("Computing c_e_alpha__c_d_beta.")
        return qml.math.einsum(
            f"...pi,...ik->...pk",
            qml.math.conjugate(self._transition_matrix),
            self.block_bm_transition_transpose_matrix
        )
    
    def _compute_c_e_alpha__c_e_beta(self):
        self.p_bar_set_postfix_str("Computing c_e_alpha__c_e_beta.")
        return qml.math.einsum(
            f"...pi,...ik->...pk",
            qml.math.conjugate(self._transition_matrix),
            self.block_bm_transition_dagger_matrix
        )
    
    def _compute_c_e_alpha__c_2p_beta_m1(self):
        self.p_bar_set_postfix_str("Computing c_e_alpha__c_2p_beta_m1.")
        return qml.math.einsum(
            f"...pi,ij->...pj",
            qml.math.conjugate(self._transition_matrix),
            self.block_diagonal_matrix
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

    def __getitem__(self, item: Tuple[int, int]):
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
        for (i, j) in indexes:
            items.append(self[i, j])
            self.update_p_bar()
        if close_p_bar:
            self.close_p_bar()
        return items

    def compute_stack_and_pad_items(
            self,
            indexes: Iterable[Tuple[int, int]],
            pad_value: numbers.Number = 0.0,
            close_p_bar: bool = True
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
                new_item[..., :item_shape[-2], :item_shape[-1]] = item
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
        warnings.warn("This method is deprecated. Use get_observable_of_target_state instead.", DeprecationWarning)
        key = (k, utils.state_to_binary_string(system_state, n=self.n_particles))
        if key not in self._observables:
            self._observables[key] = self._compute_observable(k, system_state)
        return self._observables[key]

    def get_observable_of_target_state(
            self,
            system_state: Union[int, np.ndarray, sparse.sparray],
            target_binary_state: Optional[np.ndarray] = None,
            indexes_of_target_state: Optional[np.ndarray] = None,
            **kwargs
    ) -> np.ndarray:
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
        key = (
            utils.state_to_binary_string(system_state, n=self.n_particles),
            ''.join([str(i) for i in target_binary_state]),
            ','.join([str(i) for i in indexes_of_target_state]),
        )
        if key not in self._observables:
            self._observables[key] = self.compute_observable_of_target_state(
                system_state, target_binary_state, indexes_of_target_state, **kwargs
            )
        return self._observables[key]
    
    def _compute_observable(self, k: int, system_state: Union[int, np.ndarray, sparse.sparray]) -> np.ndarray:
        warnings.warn("This method is deprecated. Use compute_observable_of_target_state instead.", DeprecationWarning)
        ket_majorana_indexes = utils.decompose_state_into_majorana_indexes(system_state, n=self.n_particles)
        bra_majorana_indexes = list(reversed(ket_majorana_indexes))

        unmeasured_cls_indexes = [2 for _ in range(len(ket_majorana_indexes))]
        measure_cls_indexes = np.array([[1, 0] for _ in range(k + 1)]).flatten().tolist()
        lt_indexes = unmeasured_cls_indexes + measure_cls_indexes + unmeasured_cls_indexes

        # measure_indexes = np.array([[i, i] for i in range(k+1)]).flatten().tolist()
        measure_indexes = [k, k]
        majorana_indexes = list(bra_majorana_indexes) + measure_indexes + list(ket_majorana_indexes)

        obs_size = len(majorana_indexes)
        obs_shape = ([self.batch_size] if self.batch_size else []) + [obs_size, obs_size]
        obs = np.zeros(obs_shape, dtype=complex)
        for (i, j) in zip(*np.triu_indices(obs_size, k=1)):
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
            **kwargs
    ) -> TensorLike:
        if target_binary_state is None and indexes_of_target_state is None:
            target_binary_state = np.array([1, ])
            indexes_of_target_state = np.array([0, ])
        elif target_binary_state is not None and indexes_of_target_state is None:
            indexes_of_target_state = np.arange(len(target_binary_state), dtype=int)
        elif target_binary_state is None and indexes_of_target_state is not None:
            target_binary_state = np.ones(len(indexes_of_target_state), dtype=int)

        ket_majorana_indexes = utils.decompose_state_into_majorana_indexes(system_state, n=self.n_particles)
        bra_majorana_indexes = list(reversed(ket_majorana_indexes))

        unmeasured_cls_indexes = [2 for _ in range(len(ket_majorana_indexes))]
        measure_cls_indexes = np.array([[b, 1 - b] for b in target_binary_state]).flatten().tolist()
        lt_indexes = np.asarray(unmeasured_cls_indexes + measure_cls_indexes + unmeasured_cls_indexes)

        measure_indexes = np.array([[i, i] for i in indexes_of_target_state]).flatten().tolist()
        majorana_indexes = np.asarray(list(bra_majorana_indexes) + measure_indexes + list(ket_majorana_indexes))

        obs_size = len(majorana_indexes)
        obs_shape = ([self.batch_size] if self.batch_size else []) + [obs_size, obs_size]
        obs = qml.math.convert_like(np.zeros(obs_shape, dtype=complex), self.transition_matrix)

        obs_indices = np.stack(np.triu_indices(obs_size, k=1))
        lt_item_cols, lt_item_rows = majorana_indexes[obs_indices[0]], majorana_indexes[obs_indices[1]]

        # compute items needed for the observable
        all_lt_indexes = np.stack((lt_indexes[obs_indices[0]], lt_indexes[obs_indices[1]]), axis=-1)
        unique_lt_indexes = np.unique(all_lt_indexes, axis=0)
        lt_items = self.compute_stack_and_pad_items(unique_lt_indexes, close_p_bar=False)

        # find the new indexes of the items in the stack
        new_all_lt_indexes = np.empty(len(obs_indices[0]), dtype=int)
        for i, (r, c) in enumerate(unique_lt_indexes):
            new_all_lt_indexes = np.where(np.all(np.isclose(all_lt_indexes, [r, c]), axis=-1), i, new_all_lt_indexes)

        # insert the elements in obs
        obs[..., obs_indices[0], obs_indices[1]] = lt_items[new_all_lt_indexes, ..., lt_item_cols, lt_item_rows].T

        self.p_bar_set_postfix_str("Finishing the computation of the observable.")
        obs = obs - qml.math.einsum("...ij->...ji", obs)
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
