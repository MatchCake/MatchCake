from typing import Union, Optional, Any

import pennylane as qml
from pennylane.operation import Operation
from pennylane import numpy as pnp
from pennylane.math import TensorLike
from pennylane.wires import Wires, WiresLike

from ..base.matchgate import Matchgate
from .. import matchgate_parameter_sets as mps, utils
from .single_particle_transition_matrices.single_particle_transition_matrix import (
    SingleParticleTransitionMatrixOperation
)
from ..utils import make_wires_continuous
from ..utils.math import fermionic_operator_matmul


class MatchgateOperation(Matchgate, Operation):
    num_params = mps.MatchgatePolarParams.N_PARAMS
    num_wires = 2
    par_domain = "A"

    grad_method = "A"
    grad_recipe = None

    generator = None

    casting_priorities = ["numpy", "autograd", "jax", "tf", "torch"]  # greater index means higher priority

    @classmethod
    def random_params(cls, batch_size=None, **kwargs):
        seed = kwargs.pop("seed", None)
        return mps.MatchgatePolarParams.random_batch_numpy(batch_size=batch_size, seed=seed)

    @classmethod
    def random(cls, wires: Wires, batch_size=None, **kwargs) -> "MatchgateOperation":
        return cls(cls.random_params(batch_size=batch_size, wires=wires, **kwargs), wires=wires, **kwargs)

    @staticmethod
    def _matrix(*params):
        # TODO: maybe remove this method to use only compute_matrix
        polar_params = mps.MatchgatePolarParams(*params)
        std_params = mps.MatchgateStandardParams.parse_from_params(polar_params)
        matrix = std_params.to_matrix()
        if qml.math.get_interface(matrix) == "torch":
            matrix = matrix.resolve_conj()
        return matrix
    
    @staticmethod
    def compute_matrix(*params, **hyperparams):
        return MatchgateOperation._matrix(*params)

    @staticmethod
    def compute_decomposition(
            *params: TensorLike,
            wires: Optional[WiresLike] = None,
            **hyperparameters: dict[str, Any],
    ):
        return [qml.QubitUnitary(MatchgateOperation.compute_matrix(*params, **hyperparameters), wires=wires)]

    
    def __init__(
            self,
            params: Union[mps.MatchgateParams, pnp.ndarray, list, tuple],
            wires=None,
            id=None,
            **kwargs
    ):
        if wires is not None:
            wires = Wires(wires)
            assert len(wires) == 2, f"MatchgateOperation requires exactly 2 wires, got {len(wires)}."
            assert wires[-1] - wires[0] == 1, f"MatchgateOperation requires consecutive wires, got {wires}."
        in_param_type = kwargs.get("in_param_type", mps.MatchgatePolarParams)
        in_params = in_param_type.parse_from_any(params)
        Matchgate.__init__(self, in_params, **kwargs)
        np_params = self.polar_params.to_vector()
        self.num_params = len(np_params)
        self.draw_label_params = kwargs.get("draw_label_params", None)
        Operation.__init__(self, *np_params, wires=wires, id=id)

    @property
    def batch_size(self):
        not_none_params = [
            p for p in
            self.get_all_params_set(make_params=False)
            if p is not None
        ]
        if len(not_none_params) == 0:
            raise ValueError("No params set. Cannot make standard params.")
        batch_size = not_none_params[0].batch_size
        if batch_size in [0, ]:
            return None
        return batch_size

    @property
    def sorted_wires(self):
        return Wires(sorted(self.wires.tolist()))

    @property
    def cs_wires(self):
        return Wires(make_wires_continuous(self.wires))

    def get_padded_single_particle_transition_matrix(self, wires=None):
        r"""
        Return the padded single particle transition matrix in order to have the block diagonal form where
        the block is the single particle transition matrix at the corresponding wires.
        
        :param wires: The wires of the whole system.
        :return: padded single particle transition matrix
        """
        return self.to_sptm_operation().pad(wires=wires).matrix()
    
    def adjoint(self):
        new_params = self.standard_params.adjoint()
        return MatchgateOperation(
            new_params,
            wires=self.wires,
            in_param_type=new_params.__class__,
        )
    
    def __matmul__(self, other):
        if isinstance(other, SingleParticleTransitionMatrixOperation):
            return fermionic_operator_matmul(self.to_sptm_operation(), other)

        if not isinstance(other, MatchgateOperation):
            raise ValueError(f"Cannot multiply MatchgateOperation with {type(other)}")
        
        if self.wires != other.wires:
            return fermionic_operator_matmul(self.to_sptm_operation(), other.to_sptm_operation())

        new_params = self.standard_params @ other.standard_params
        return MatchgateOperation(
            new_params,
            wires=self.wires,
            in_param_type=new_params.__class__,
        )
    
    def label(self, decimals=None, base_label=None, cache=None):
        if self.draw_label_params is None:
            return super().label(decimals=decimals, base_label=base_label, cache=cache)

        op_label = base_label or self.__class__.__name__
        return f"{op_label}({self.draw_label_params})"

    def __repr__(self):
        return Operation.__repr__(self)

    def __str__(self):
        return Operation.__str__(self)

    def __copy__(self):
        return Operation.__copy__(self)

    def to_sptm_operation(self):
        return SingleParticleTransitionMatrixOperation(
            self.single_particle_transition_matrix,
            wires=self.wires,
            **getattr(self, "_hyperparameters", {})
        )



