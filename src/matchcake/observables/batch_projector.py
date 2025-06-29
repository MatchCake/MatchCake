import pennylane as qml
from pennylane.operation import Observable
from pennylane.wires import Wires


class BatchProjector(Observable):
    def __init__(self, states, wires, **kwargs):
        wires = Wires(wires, _override=True)
        wires_array = wires.toarray()
        wires_shape = wires_array.shape
        if len(wires_shape) == 1:
            wires = [wires for _ in range(len(states))]
        else:
            wires = Wires([Wires(w) for w in wires])
        super().__init__(states, wires=Wires.all_wires(wires), **kwargs)
        self.projectors = [qml.Projector(state, wires=s_wires) for state, s_wires in zip(states, wires)]

    def get_states(self):
        return [p.parameters[0] for p in self.projectors]

    def get_batch_wires(self):
        return [p.wires for p in self.projectors]
