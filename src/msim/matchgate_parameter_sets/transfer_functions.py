from typing import Type, Dict, Callable

from . import (
    MatchgateParams,
    MatchgatePolarParams,
    MatchgateStandardParams,
    MatchgateHamiltonianCoefficientsParams,
    MatchgateComposedHamiltonianParams,
    MatchgateStandardHamiltonianParams,
)

_transfer_funcs_by_type: Dict[Type[MatchgateParams], Dict[Type[MatchgateParams], Callable]] = {
    # from              : to
    MatchgatePolarParams: {
        MatchgateStandardParams: polar_to_standard,
        MatchgateHamiltonianCoefficientsParams: polar_to_hamiltonian_coefficients,
        MatchgateComposedHamiltonianParams: polar_to_composed_hamiltonian,
        MatchgateStandardHamiltonianParams: polar_to_standard_hamiltonian,
    },
    MatchgateStandardParams: {
        MatchgatePolarParams: standard_to_polar,
        MatchgateHamiltonianCoefficientsParams: standard_to_hamiltonian_coefficients,
        MatchgateComposedHamiltonianParams: standard_to_composed_hamiltonian,
        MatchgateStandardHamiltonianParams: standard_to_standard_hamiltonian,
    },
    MatchgateHamiltonianCoefficientsParams: {
        MatchgatePolarParams: hamiltonian_coefficients_to_polar,
        MatchgateStandardParams: hamiltonian_coefficients_to_standard,
        MatchgateComposedHamiltonianParams: hamiltonian_coefficients_to_composed_hamiltonian,
        MatchgateStandardHamiltonianParams: hamiltonian_coefficients_to_standard_hamiltonian,
    },
    MatchgateComposedHamiltonianParams: {
        MatchgatePolarParams: composed_hamiltonian_to_polar,
        MatchgateStandardParams: composed_hamiltonian_to_standard,
        MatchgateHamiltonianCoefficientsParams: composed_hamiltonian_to_hamiltonian_coefficients,
        MatchgateStandardHamiltonianParams: composed_hamiltonian_to_standard_hamiltonian,
    },
    MatchgateStandardHamiltonianParams: {
        MatchgatePolarParams: standard_hamiltonian_to_polar,
        MatchgateStandardParams: standard_hamiltonian_to_standard,
        MatchgateHamiltonianCoefficientsParams: standard_hamiltonian_to_hamiltonian_coefficients,
        MatchgateComposedHamiltonianParams: standard_hamiltonian_to_composed_hamiltonian,
    },
}


def params_to(params, __cls: Type[MatchgateParams]) -> MatchgateParams:
    if isinstance(params, __cls):
        return params
    elif not isinstance(params, MatchgateParams):
        return __cls(*params)
    else:
        return _transfer_funcs_by_type[type(params)][__cls](params)








