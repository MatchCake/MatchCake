from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pennylane as qml
import torch
from pennylane.operation import Operation
from pennylane.typing import TensorLike

from ..sptm_circuit_records import SptmCircuitRecords
from .contraction_container import _ContractionMatchgatesContainer
from .contraction_strategy import ContractionStrategy


class BulkContractionStrategy(ContractionStrategy):
    r"""
    Leave every operation as it is while the circuit is applied, and compile the whole circuit when
    the global single-particle transition matrix is read.

    The other strategies multiply matrices together as the operations come. This one multiplies
    nothing: the device records each operation, and :meth:`build_plan` turns the records into a few
    batched products. The number of tensor operations then depends on the number of layers and of
    gate classes of the circuit, and not on its number of gates, which is what an accelerator needs.
    """

    NAME: str = "Bulk"

    @staticmethod
    def _schedule(starts: np.ndarray, sizes: np.ndarray, n_columns: int) -> np.ndarray:
        """Give each block the earliest layer that comes after every block recorded before it on its columns.

        :param starts: The first column of each block, in the order the blocks are applied, shape (n_blocks,).
        :type starts: np.ndarray
        :param sizes: The number of columns of each block, shape (n_blocks,).
        :type sizes: np.ndarray
        :param n_columns: The number of columns of the matrix.
        :type n_columns: int
        :return: The layer of each block, shape (n_blocks,).
        :rtype: np.ndarray
        """
        depth = [0] * n_columns
        layers = np.zeros(len(starts), dtype=np.int64)
        for index, (start, size) in enumerate(zip(starts.tolist(), sizes.tolist())):
            layer = max(depth[start : start + size])
            layers[index] = layer
            depth[start : start + size] = [layer + 1] * size
        return layers

    def __call__(self, operations: Sequence[Operation], **kwargs) -> Sequence[Operation]:
        return operations

    def get_container(self) -> _ContractionMatchgatesContainer:
        return _ContractionMatchgatesContainer()

    def get_next_operations(self, operation) -> Sequence[Optional[Operation]]:
        return [operation]

    def get_reminding(self):
        return None

    def build_plan(
        self, records: SptmCircuitRecords, n_columns: int, dtype_name: str
    ) -> Optional[List[Tuple[Any, TensorLike]]]:
        """Compile the recorded operations into stacks of blocks acting on disjoint columns.

        The matrices of all the gates of one class are computed in one call, every block is scheduled
        in the earliest layer that comes after all the blocks recorded before it on the same columns,
        and the blocks of each layer are stacked, one stack per block size.

        :param records: The operations applied to the device since its global matrix was last read.
        :type records: SptmCircuitRecords
        :param n_columns: The number of columns of the matrix the blocks multiply.
        :type n_columns: int
        :param dtype_name: The name of the real dtype of the blocks, such as ``"float64"``.
        :type dtype_name: str
        :return: For each layer and block size, in the order they are applied, the columns each block
            acts on, shape (n_blocks, 2k), and the blocks, shape (n_blocks, 2k, 2k) or
            (batch, n_blocks, 2k, 2k), as torch tensors if any recorded tensor is one and as numpy
            arrays otherwise. None if a recorded tensor is of another backend.
        :rtype: Optional[List[Tuple[Any, TensorLike]]]
        """
        entries = records.entries
        if len(entries) == 0:
            return []
        tensors = [tensor for _, _, _, tensor in entries]
        if not {qml.math.get_interface(tensor) for tensor in tensors} <= {"numpy", "torch"}:
            return None
        torch_tensors = [tensor for tensor in tensors if isinstance(tensor, torch.Tensor)]
        use_torch = len(torch_tensors) > 0
        torch_device = None
        if use_torch:
            # a tensor on an accelerator decides where the product lives
            torch_device = next((tensor.device for tensor in torch_tensors if tensor.device.type != "cpu"), "cpu")
        batch_size = max(
            (
                qml.math.shape(tensor)[0]
                for _, _, gate_class, tensor in entries
                if qml.math.ndim(tensor) > (1 if gate_class is not None else 2)
            ),
            default=0,
        )

        # tensors of the same kind are converted and stacked together: (kind) -> entry indices
        kinds: Dict[Tuple[Any, int, bool], List[int]] = {}
        for index, (_, size, gate_class, tensor) in enumerate(entries):
            is_batched = qml.math.ndim(tensor) > (1 if gate_class is not None else 2)
            kinds.setdefault((gate_class, size, is_batched), []).append(index)
        # Blocks with and without batch are kept in separate tables, one per block size, so that a
        # batched gate does not make every other block as large as the batch.
        tables: Dict[Tuple[int, bool], List[TensorLike]] = {}
        table_entries: Dict[Tuple[int, bool], List[int]] = {}
        for (gate_class, size, is_batched), indices in kinds.items():
            stacked: Any
            kind_tensors = [entries[index][3] for index in indices]
            if is_batched and len({qml.math.shape(tensor)[0] for tensor in kind_tensors}) > 1:
                # a batch of one is broadcast to the batch of the others before they are stacked
                kind_tensors = [
                    qml.math.broadcast_to(tensor, (batch_size, *qml.math.shape(tensor)[1:])) for tensor in kind_tensors
                ]
            if use_torch:
                if not all(isinstance(tensor, torch.Tensor) for tensor in kind_tensors):
                    kind_tensors = [torch.as_tensor(tensor) for tensor in kind_tensors]
                if len({(tensor.dtype, tensor.device) for tensor in kind_tensors}) > 1:
                    kind_tensors = [tensor.to(device=torch_device, dtype=torch.float64) for tensor in kind_tensors]
                stacked = torch.stack(kind_tensors).to(device=torch_device)
                if not stacked.is_floating_point():
                    stacked = stacked.to(torch.float64)
                # The formula is evaluated in the precision of the parameters, and its result is cast:
                # casting large angles first would lose their low bits before the cosine and sine.
                blocks = stacked if gate_class is None else gate_class.compute_sptm(stacked)
                blocks = blocks.to(getattr(torch, dtype_name))
            else:
                stacked = np.stack([np.asarray(tensor) for tensor in kind_tensors])
                if not np.issubdtype(stacked.dtype, np.floating):
                    stacked = stacked.astype(np.float64)
                blocks = stacked if gate_class is None else gate_class.compute_sptm(stacked)
                blocks = np.asarray(blocks, dtype=dtype_name)
            if is_batched and blocks.shape[1] != batch_size:
                target_shape = (blocks.shape[0], batch_size, size, size)
                blocks = blocks.expand(target_shape) if use_torch else np.broadcast_to(blocks, target_shape)
            tables.setdefault((size, is_batched), []).append(blocks)
            table_entries.setdefault((size, is_batched), []).extend(indices)

        starts = np.array([start for start, _, _, _ in entries], dtype=np.int64)
        sizes = np.array([size for _, size, _, _ in entries], dtype=np.int64)
        layers = self._schedule(starts, sizes, n_columns)
        keyed_groups = []
        for (size, is_batched), chunks in tables.items():
            entry_indices = np.array(table_entries[(size, is_batched)], dtype=np.int64)
            # The table is reordered by layer with one gather and cut into the stacks of the layers
            # with one split: indexing it once per layer would cost a pass over the whole table per
            # layer in the backward pass.
            order = np.argsort(layers[entry_indices], kind="stable")
            sorted_entries = entry_indices[order]
            sorted_layers = layers[sorted_entries]
            boundaries = np.flatnonzero(sorted_layers[1:] != sorted_layers[:-1]) + 1
            counts = np.diff(np.concatenate([[0], boundaries, [len(sorted_entries)]])).tolist()
            table: Any
            stacks: Any
            if use_torch:
                table = torch.cat(chunks).index_select(0, torch.as_tensor(order, device=torch_device))
                stacks = torch.split(table, counts)
            else:
                table = np.concatenate(chunks)[order]
                stacks = np.split(table, boundaries)
            stack: Any
            for run, stack in zip(np.split(sorted_entries, boundaries), stacks):
                columns: Any = starts[run][:, None] + np.arange(size)[None, :]
                if use_torch:
                    columns = torch.as_tensor(columns, device=torch_device)
                if is_batched:
                    stack = stack.transpose(0, 1) if use_torch else np.swapaxes(stack, 0, 1)
                keyed_groups.append((int(layers[run[0]]), columns, stack))
        # the stacks of a layer act on disjoint columns, so only the order of the layers matters
        keyed_groups.sort(key=lambda group: group[0])
        return [(columns, stack) for _, columns, stack in keyed_groups]
