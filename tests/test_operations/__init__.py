import numpy as np
import pennylane as qml
from pennylane.ops.qubit.observables import BasisStateProjector


def specific_ops_circuit(cls_params_wires_list, initial_state=None, **kwargs):
    all_wires = kwargs.get("all_wires", None)
    if all_wires is None:
        all_wires = set(
            sum([list(wires) for _, __, wires in cls_params_wires_list], [])
        )
    all_wires = np.sort(np.asarray(all_wires))
    if initial_state is None:
        initial_state = np.zeros(len(all_wires), dtype=int)
    qml.BasisState(initial_state, wires=all_wires)
    if kwargs.get("adjoint", False):
        adjoint_func = qml.adjoint
    else:
        adjoint_func = lambda x: x
    for cpw in cls_params_wires_list:
        if len(cpw) == 2:
            cls, wires = cpw
            cls = adjoint_func(cls)
            if cls.num_params == 0:
                cls(wires=wires)
            else:
                cls(np.random.uniform(0.0, 1.0, size=cls.num_params), wires=wires)
        elif len(cpw) == 3:
            cls, params, wires = cpw
            cls = adjoint_func(cls)
            cls(params, wires=wires)
        elif len(cpw) == 4:
            cls, params, wires, cls_kwargs = cpw
            cls = adjoint_func(cls)
            cls(params, wires=wires, **cls_kwargs)
        else:
            raise ValueError(f"Unknown cls_params_wires_list: {cpw}.")

    out_op = kwargs.get("out_op", "expval")
    if out_op == "state":
        return qml.state()
    elif out_op == "probs":
        return qml.probs(wires=kwargs.get("out_wires", None))
    elif out_op == "expval":
        projector: BasisStateProjector = qml.Projector(initial_state, wires=all_wires)
        return qml.expval(projector)
    else:
        raise ValueError(f"Unknown out_op: {out_op}.")
