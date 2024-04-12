import warnings
from typing import Optional, Tuple, Union

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from scipy import sparse

from .. import utils


class NonInteractingFermionicLookupTable:
    def __init__(
            self,
            transition_matrix: pnp.ndarray,
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
        
        self._observables = {}
    
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
    
    def _compute_c_d_alpha__c_d_beta(self):
        return qml.math.einsum(
            f"...pi,ij,...kj->...pk",
            self._transition_matrix,
            self.block_diagonal_matrix,
            self.transition_matrix
        )

    def _compute_c_d_alpha__c_e_beta(self):
        return qml.math.einsum(
            f"...pi,ij,...kj->...pk",
            self._transition_matrix,
            self.block_diagonal_matrix,
            qml.math.conjugate(self.transition_matrix)
        )
    
    def _compute_c_d_alpha__c_2p_beta_m1(self):
        return qml.math.einsum(
            f"...pi,ij->...pj",
            self._transition_matrix,
            self.block_diagonal_matrix
        )

    def _compute_c_e_alpha__c_d_beta(self):
        return qml.math.einsum(
            f"...pi,ij,...kj->...pk",
            qml.math.conjugate(self._transition_matrix),
            self.block_diagonal_matrix,
            self.transition_matrix
        )
    
    def _compute_c_e_alpha__c_e_beta(self):
        return qml.math.einsum(
            f"...pi,ij,...kj->...pk",
            qml.math.conjugate(self._transition_matrix),
            self.block_diagonal_matrix,
            qml.math.conjugate(self.transition_matrix)
        )
    
    def _compute_c_e_alpha__c_2p_beta_m1(self):
        return qml.math.einsum(
            f"...pi,ij->...pj",
            qml.math.conjugate(self._transition_matrix),
            self.block_diagonal_matrix
        )

    def _compute_c_2p_alpha_m1__c_d_beta(self):
        return qml.math.einsum(
            f"ij,...kj->...ik",
            self.block_diagonal_matrix,
            self.transition_matrix
        )

    def _compute_c_2p_alpha_m1__c_e_beta(self):
        return qml.math.einsum(
            f"ij,...kj->...ik",
            self.block_diagonal_matrix,
            qml.math.conjugate(self.transition_matrix)
        )

    def _compute_c_2p_alpha_m1__c_2p_beta_m1(self):
        if self.batch_size > 0:
            size = qml.math.shape(self.transition_matrix)[-1]
            shape = ([self.batch_size] if self.batch_size else []) + [size, size]
            matrix = pnp.zeros(shape, dtype=complex)
            matrix[..., :, :] = qml.math.eye(size, dtype=complex)
            return matrix
        return np.eye(self.transition_matrix.shape[-1])

    def __getitem__(self, item: Tuple[int, int]):
        i, j = item
        if i == 0 and j == 0:
            return self.c_d_alpha__c_d_beta
        elif i == 0 and j == 1:
            return self.c_d_alpha__c_e_beta
        elif i == 0 and j == 2:
            return self.c_d_alpha__c_2p_beta_m1
        elif i == 1 and j == 0:
            return self.c_e_alpha__c_d_beta
        elif i == 1 and j == 1:
            return self.c_e_alpha__c_e_beta
        elif i == 1 and j == 2:
            return self.c_e_alpha__c_2p_beta_m1
        elif i == 2 and j == 0:
            return self.c_2p_alpha_m1__c_d_beta
        elif i == 2 and j == 1:
            return self.c_2p_alpha_m1__c_e_beta
        elif i == 2 and j == 2:
            return self.c_2p_alpha_m1__c_2p_beta_m1
        else:
            raise IndexError(f"Index ({i}, {j}) out of bounds for lookup table of shape {self.shape}")

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
        key = (
            utils.state_to_binary_string(system_state, n=self.n_particles),
            ''.join([str(i) for i in target_binary_state]),
            ','.join([str(i) for i in indexes_of_target_state]),
        )
        if key not in self._observables:
            self._observables[key] = self.compute_observable_of_target_state(
                system_state, target_binary_state, indexes_of_target_state
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
    ) -> np.ndarray:
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
        lt_indexes = unmeasured_cls_indexes + measure_cls_indexes + unmeasured_cls_indexes

        measure_indexes = np.array([[i, i] for i in indexes_of_target_state]).flatten().tolist()
        majorana_indexes = list(bra_majorana_indexes) + measure_indexes + list(ket_majorana_indexes)

        obs_size = len(majorana_indexes)
        obs_shape = ([self.batch_size] if self.batch_size else []) + [obs_size, obs_size]
        obs = qml.math.convert_like(pnp.zeros(obs_shape, dtype=complex), self.transition_matrix)
        for (i, j) in zip(*np.triu_indices(obs_size, k=1)):
            i_k, j_k = majorana_indexes[i], majorana_indexes[j]
            row, col = lt_indexes[i], lt_indexes[j]
            obs[..., i, j] = self[row, col][..., i_k, j_k]
        obs = obs - qml.math.einsum("...ij->...ji", obs)
        return obs
