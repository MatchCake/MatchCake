from typing import Optional, Tuple

import numpy as np
from pennylane import numpy as pnp
from . import utils


class NonInteractingFermionicLookupTable:
    def __init__(
            self,
            transition_matrix: pnp.ndarray,
            block_diagonal_matrix: Optional[pnp.ndarray] = None,
    ):
        self._transition_matrix = transition_matrix
        self._block_diagonal_matrix = block_diagonal_matrix
        
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
    def transition_matrix(self):
        return self._transition_matrix
    
    @property
    def block_diagonal_matrix(self):
        if self._block_diagonal_matrix is None:
            self._block_diagonal_matrix = utils.get_block_diagonal_matrix(self.transition_matrix.shape[0])
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
        return self._transition_matrix @ self.block_diagonal_matrix @ self.transition_matrix.T
    
    def _compute_c_d_alpha__c_e_beta(self):
        return self._transition_matrix @ self.block_diagonal_matrix @ pnp.conjugate(self.transition_matrix.T)
    
    def _compute_c_d_alpha__c_2p_beta_m1(self):
        return self._transition_matrix @ self.block_diagonal_matrix
    
    def _compute_c_e_alpha__c_d_beta(self):
        return pnp.conjugate(self._transition_matrix) @ self.block_diagonal_matrix @ self.transition_matrix.T
    
    def _compute_c_e_alpha__c_e_beta(self):
        return (
                pnp.conjugate(self._transition_matrix)
                @ self.block_diagonal_matrix
                @ pnp.conjugate(self.transition_matrix.T)
        )
    
    def _compute_c_e_alpha__c_2p_beta_m1(self):
        return pnp.conjugate(self._transition_matrix) @ self.block_diagonal_matrix
    
    def _compute_c_2p_alpha_m1__c_d_beta(self):
        return self.block_diagonal_matrix @ self.transition_matrix.T
    
    def _compute_c_2p_alpha_m1__c_e_beta(self):
        return self.block_diagonal_matrix @ pnp.conjugate(self.transition_matrix.T)
    
    def _compute_c_2p_alpha_m1__c_2p_beta_m1(self):
        return self.block_diagonal_matrix  # TODO: delta_alpha_beta

    def __getitem__(self, i: int, j: int):
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

    def get_observable(self, k: int, hamming_weight: int) -> np.ndarray:
        r"""
        Get the observable corresponding to the index k and the hamming weight of the system state.
        
        :param k: Index of the observable
        :type k: int
        :param hamming_weight: Hamming weight of the system state
        :type hamming_weight: int
        :return: The observable of shape (2h + 2, 2h + 2) where h is the hamming weight.
        :rtype: np.ndarray
        """
        if k not in self._observables:
            self._observables[k] = self._compute_observable(k, hamming_weight)
        return self._observables[k]
    
    def _compute_observable(self, k: int, hamming_weight: int) -> np.ndarray:
        raise NotImplementedError()
        
