import itertools
import random

import numpy as np
import pennylane as qml
import pytest
from pennylane.wires import Wires

from matchcake import utils
from matchcake.base.lookup_table import NonInteractingFermionicLookupTable
from matchcake.circuits import (RandomSptmHaarOperationsGenerator,
                                RandomSptmOperationsGenerator)
from matchcake.devices import NIFDevice
from matchcake.devices.contraction_strategies import contraction_strategy_map
from matchcake.operations import (SptmFermionicSuperposition, SptmFHH,
                                  SptmfRxRx, SptmFSwap, SptmFSwapRzRz,
                                  SptmIdentity, SptmRyRy, SptmRzRz)
from matchcake.utils.math import dagger, det

from .. import get_slow_test_mark
from ..configs import (ATOL_MATRIX_COMPARISON, N_RANDOM_TESTS_PER_CASE,
                       RTOL_MATRIX_COMPARISON, TEST_SEED, set_seed)
from ..test_nif_device import devices_init

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
                seed=TEST_SEED + i,
            ),
            contraction_strategy,
        )
        for i in range(N_RANDOM_TESTS_PER_CASE)
        for num_wires in [2, 3, 6]
        for num_gates in [0, 1, 10 * num_wires]
        for batch_size in [None, 16]
        for contraction_strategy in contraction_strategy_map.keys()
        for gen_cls in [
            RandomSptmOperationsGenerator,
            RandomSptmHaarOperationsGenerator,
        ]
    ],
)
def test_global_sptm_unitary(operations_generator: RandomSptmOperationsGenerator, contraction_strategy):
    nif_device = NIFDevice(wires=operations_generator.wires, contraction_strategy=contraction_strategy)
    nif_device.execute_generator(
        operations_generator,
        n_ops=operations_generator.n_ops,
        apply=True,
        reset=True,
        cache_global_sptm=True,
    )
    global_sptm = nif_device.global_sptm.matrix()
    expected_eye = np.einsum("...ij,...jk->...ik", global_sptm, dagger(global_sptm))
    eye = np.zeros_like(expected_eye)
    eye[
        ...,
        np.arange(2 * operations_generator.n_wires),
        np.arange(2 * operations_generator.n_wires),
    ] = 1
    np.testing.assert_allclose(
        expected_eye,
        eye,
        atol=ATOL_MATRIX_COMPARISON,
        rtol=RTOL_MATRIX_COMPARISON,
    )


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
                seed=TEST_SEED + i,
            ),
            contraction_strategy,
        )
        for i in range(N_RANDOM_TESTS_PER_CASE)
        for num_wires in [2, 3, 6]
        for num_gates in [0, 1, 10 * num_wires]
        for batch_size in [None, 16]
        for contraction_strategy in contraction_strategy_map.keys()
        for gen_cls in [
            RandomSptmOperationsGenerator,
            RandomSptmHaarOperationsGenerator,
        ]
    ],
)
def test_global_sptm_det(operations_generator: RandomSptmOperationsGenerator, contraction_strategy):
    nif_device = NIFDevice(wires=operations_generator.wires, contraction_strategy=contraction_strategy)
    nif_device.execute_generator(
        operations_generator,
        n_ops=operations_generator.n_ops,
        apply=True,
        reset=True,
        cache_global_sptm=True,
    )
    global_sptm = nif_device.apply_metadata["global_sptm"]
    sptm_det = det(global_sptm)
    np.testing.assert_allclose(
        sptm_det,
        1.0,
        atol=ATOL_MATRIX_COMPARISON,
        rtol=RTOL_MATRIX_COMPARISON,
    )


@get_slow_test_mark()
@pytest.mark.slow
@pytest.mark.parametrize(
    "operations, contraction_strategy",
    [
        (operations, contraction_strategy)
        for contraction_strategy in contraction_strategy_map.keys()
        for operations in [
            [SptmfRxRx.random(wires=[0, 1]), SptmFSwap(wires=[0, 1])],
            [SptmfRxRx.random(wires=[0, 1]), SptmFSwap(wires=[1, 2])],
            [SptmFSwap(wires=[1, 2]), SptmfRxRx.random(wires=[0, 1])],
            [SptmFSwap(wires=[0, 1]), SptmfRxRx.random(wires=[0, 1])],
            [
                SptmFSwap(wires=[0, 1]),
                SptmfRxRx.random(wires=[0, 1]),
                SptmFSwap(wires=[0, 1]),
            ],
            [
                SptmFSwap(wires=[0, 1]),
                SptmfRxRx.random(wires=[1, 2]),
                SptmFSwap(wires=[0, 1]),
            ],
            [
                SptmFSwap(wires=[0, 1]),
                SptmfRxRx.random(wires=[1, 2]),
                SptmFSwap(wires=[2, 3]),
            ],
            sum(
                [[SptmfRxRx.random(wires=[0, 1]), SptmFSwap(wires=[0, 1])] for _ in range(10)],
                start=[],
            ),
            sum(
                [[SptmfRxRx.random(wires=[i, i + 1]), SptmFSwap(wires=[0, i + 1])] for i in range(10)],
                start=[],
            ),
            sum(
                [
                    [
                        SptmfRxRx.random(wires=[2 * i, 2 * i + 1]),
                        SptmFSwap(wires=[2 * i + 2, 2 * i + 3]),
                    ]
                    for i in range(10)
                ],
                start=[],
            ),
            [
                random.choice([SptmfRxRx, SptmFSwap]).random(wires=w)
                for w in [
                    [0, 1],
                    [0, 1],
                    [2, 3],
                    [2, 3],
                    [2, 3],
                    [2, 3],
                    [3, 4],
                    [2, 3],
                    [0, 1],
                    [3, 4],
                    [1, 2],
                    [2, 3],
                    [1, 2],
                    [0, 1],
                    [3, 4],
                    [3, 4],
                    [1, 2],
                    [0, 1],
                    [2, 3],
                    [0, 1],
                    [1, 2],
                    [2, 3],
                    [3, 4],
                    [1, 2],
                    [1, 2],
                    [2, 3],
                    [2, 3],
                    [3, 4],
                    [1, 2],
                    [1, 2],
                    [3, 4],
                    [0, 1],
                    [3, 4],
                    [1, 2],
                    [2, 3],
                    [1, 2],
                    [2, 3],
                    [3, 4],
                    [0, 1],
                    [2, 3],
                    [0, 1],
                    [3, 4],
                    [1, 2],
                    [3, 4],
                    [3, 4],
                    [0, 1],
                    [2, 3],
                    [3, 4],
                    [3, 4],
                    [3, 4],
                ]
            ],
        ]
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ],
)
def test_global_sptm_det_specific_circuit(operations, contraction_strategy):
    wires = Wires.all_wires([op.cs_wires for op in operations])
    nif_device = NIFDevice(wires=wires, contraction_strategy=contraction_strategy)
    nif_device.execute_generator(
        iter(operations),
        n_ops=len(operations),
        apply=True,
        reset=True,
        cache_global_sptm=True,
    )
    global_sptm = nif_device.apply_metadata["global_sptm"]
    sptm_det = det(global_sptm)
    np.testing.assert_allclose(
        sptm_det,
        1.0,
        atol=ATOL_MATRIX_COMPARISON,
        rtol=RTOL_MATRIX_COMPARISON,
    )
