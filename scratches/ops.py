import matchcake as mc
import pennylane as qml
import numpy as np

from matchcake.operations.fermionic_paulis import fXX


def recursive_expand(op):
    if isinstance(op, list):
        return [recursive_expand(sub_op) for sub_op in op]
    if op.has_decomposition:
        return [recursive_expand(sub_op) for sub_op in op.expand().operations]
    else:
        return op


def recursive_expand_to_matrices(op):
    if isinstance(op, list):
        return [recursive_expand_to_matrices(sub_op) for sub_op in op]
    if op.has_decomposition:
        return [recursive_expand_to_matrices(sub_op) for sub_op in op.expand().operations]
    else:
        return op.gate_data


def flatten_list(lst):
    out = []
    for item in lst:
        if isinstance(item, list):
            out.extend(flatten_list(item))
        else:
            out.append(item)
    return out


wires = [0, 1]
# fCNOT = mc.operations.fCNOT(wires=[0, 1])
# ops_list = flatten_list(recursive_expand_to_matrices(fCNOT))
ops_list = [
    mc.operations.fH(wires=wires),
    mc.operations.fH(wires=wires),
    mc.operations.fSWAP(wires=wires),
    fXX(wires=wires),
    mc.operations.fH(wires=wires),
    mc.operations.fH(wires=wires),
]
ops_list = [op.gate_data for op in ops_list]
ops_list_shapes = [mc.utils.math.shape(op) for op in ops_list]
print(ops_list_shapes)
op_product = qml.math.squeeze(mc.utils.recursive_2in_operator(qml.math.matmul, ops_list)).numpy().real

# print(ops_list)

print(op_product)

