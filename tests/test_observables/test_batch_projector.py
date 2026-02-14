import numpy as np
from pennylane.wires import Wires

from matchcake.observables import BatchProjector


class TestBatchProjector:
    def test_batch_projector(self):
        wires = [0, 1]
        states = [[1, 0]]
        projector = BatchProjector(states, wires)
        assert [Wires(wires)] == projector.get_batch_wires()
        np.testing.assert_array_equal(projector.get_states(), states)
