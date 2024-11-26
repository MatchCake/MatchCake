import itertools

import numpy as np
import pytest
import pennylane as qml

from matchcake import utils
from matchcake.base.lookup_table import NonInteractingFermionicLookupTable
from matchcake.devices import NIFDevice
from matchcake.devices.contraction_strategies import contraction_strategy_map
from matchcake.circuits import RandomSptmOperationsGenerator, RandomSptmHaarOperationsGenerator
from .. import get_slow_test_mark
from ..test_nif_device import devices_init
from ..configs import (
    N_RANDOM_TESTS_PER_CASE,
    ATOL_MATRIX_COMPARISON,
    RTOL_MATRIX_COMPARISON,
    TEST_SEED,
    set_seed
)

set_seed(TEST_SEED)

@get_slow_test_mark()
@pytest.mark.slow
@pytest.mark.parametrize(
    "operations_generator, contraction_strategy",
    [
        (
                gen_cls(
                    wires=np.arange(num_wires),
                    n_ops=num_gates,
                    batch_size=batch_size,
                    seed=TEST_SEED + i
                ),
                contraction_strategy
        )
        for i in range(N_RANDOM_TESTS_PER_CASE)
        for num_wires in range(2, 10)
        for num_gates in [0, 1, 10 * num_wires]
        for batch_size in [None, 16]
        for contraction_strategy in contraction_strategy_map.keys()
        for gen_cls in [RandomSptmOperationsGenerator, RandomSptmHaarOperationsGenerator]
    ]
)
def test_global_sptm_unitary(operations_generator: RandomSptmOperationsGenerator, contraction_strategy):
    nif_device = NIFDevice(wires=operations_generator.wires, contraction_strategy=contraction_strategy)
    nif_device.execute_generator(
        operations_generator, n_ops=operations_generator.n_ops, apply=True, reset=True, cache_global_sptm=True
    )
    global_sptm = nif_device.apply_metadata["global_sptm"]
    global_sptm_dagger = np.einsum("...ij->...ji", global_sptm).conj()
    expected_eye = np.einsum("...ij,...jk->...ik", global_sptm, global_sptm_dagger)
    eye = np.zeros_like(expected_eye)
    eye[..., np.arange(2 * operations_generator.n_wires), np.arange(2 * operations_generator.n_wires)] = 1
    np.testing.assert_allclose(
        expected_eye, eye,
        atol=ATOL_MATRIX_COMPARISON,
        rtol=RTOL_MATRIX_COMPARISON,
    )
