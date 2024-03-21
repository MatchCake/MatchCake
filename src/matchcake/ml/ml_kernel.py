import numbers
import warnings
from collections import defaultdict
from typing import Optional, Tuple, Type, Union
import time
import datetime
import os
import numpy as np
from matplotlib import pyplot as plt
from pennylane import numpy as pnp
from pennylane import AngleEmbedding
from pennylane.templates.broadcast import wires_pyramid, PATTERN_TO_NUM_PARAMS, PATTERN_TO_WIRES
from pennylane.ops.qubit.observables import BasisStateProjector
from sklearn import svm
from sklearn.utils.estimator_checks import check_estimator
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.ensemble import VotingClassifier
import pennylane as qml
from pennylane.wires import Wires
import pythonbasictools as pbt
from tqdm import tqdm
import tables as tb

from ..devices.nif_device import NonInteractingFermionicDevice
from ..operations import MAngleEmbedding, MAngleEmbeddings, fRZZ, fCNOT, fSWAP, fH
from ..operations.fermionic_controlled_not import FastfCNOT


class GramMatrixKernel:
    def __init__(
            self,
            shape: Tuple[int, int],
            dtype: Optional[Union[str, np.dtype]] = np.float64,
            array_type: str = "table",
            **kwargs
    ):
        self._shape = shape
        self._dtype = dtype
        self._array_type = array_type
        self.kwargs = kwargs
        self.h5file = None
        self.data = None
        self._initiate_data_()

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def is_square(self):
        return self.shape[0] == self.shape[1]

    @property
    def size(self):
        return self.shape[0] * self.shape[1]

    def _initiate_data_(self):
        if self._array_type == "table":
            self.h5file = tb.open_file("gram_matrix.h5", mode="w")
            self.data = self.h5file.create_carray(
                self.h5file.root,
                "data",
                tb.Float64Atom(),
                shape=self.shape,
            )
        else:
            self.data = np.zeros(self.shape, dtype=self.dtype)

    def __array__(self):
        return self.data[:]

    def __getitem__(self, item):
        return self.data[item]

    def __setitem__(self, key, value):
        self.data[key] = value

    def symmetrize(self):
        if not self.is_square:
            warnings.warn("Cannot symmetrize a non-square matrix.")
            return
        if self._array_type == "table":
            self.data[:] = (self.data + self.data.T) / 2
        elif self._array_type == "ndarray":
            self.data = (self.data + self.data.T) / 2
        else:
            raise ValueError(f"Unknown array type: {self._array_type}.")

    def mirror(self):
        if not self.is_square:
            warnings.warn("Cannot mirror a non-square matrix.")
            return
        if self._array_type == "table":
            self.data[:] = self.data.T
        elif self._array_type == "ndarray":
            self.data = self.data.T
        else:
            raise ValueError(f"Unknown array type: {self._array_type}.")

    def triu_reflect(self):
        if not self.is_square:
            warnings.warn("Cannot triu_reflect a non-square matrix.")
            return
        if self._array_type == "table":
            self.data[:] = self.data + self.data.T
        elif self._array_type == "ndarray":
            self.data = self.data + self.data.T
        else:
            raise ValueError(f"Unknown array type: {self._array_type}.")

    def tril_reflect(self):
        if not self.is_square:
            warnings.warn("Cannot tril_reflect a non-square matrix.")
            return
        if self._array_type == "table":
            self.data[:] = self.data + self.data.T
        elif self._array_type == "ndarray":
            self.data = self.data + self.data.T
        else:
            raise ValueError(f"Unknown array type: {self._array_type}.")

    def fill_diagonal(self, value: float):
        if not self.is_square:
            warnings.warn("Cannot fill_diagonal a non-square matrix.")
            return
        if self._array_type == "table":
            self.data[np.diag_indices(self.shape[0])] = value
        elif self._array_type == "ndarray":
            np.fill_diagonal(self.data, value)
        else:
            raise ValueError(f"Unknown array type: {self._array_type}.")

    def close(self):
        if self.h5file is not None:
            self.h5file.close()
            self.h5file = None
            self.data = None

    def make_batches_indexes_generator(self, batch_size: int) -> Tuple[iter, int]:
        if self.is_square:
            # number of elements in the upper triangle without the diagonal elements
            length = self.size * (self.size - 1) // 2
        else:
            length = self.size - self.shape[0]
        n_batches = int(np.ceil(length / batch_size))

        def _gen():
            for i in range(n_batches):
                b_start_idx = i * batch_size
                b_end_idx = (i + 1) * batch_size
                # generate the coordinates of the matrix elements in that batch
                yield np.unravel_index(
                    np.arange(b_start_idx, b_end_idx),
                    self.shape
                )

        return _gen(), n_batches


def mrot_zz_template(param0, param1, wires):
    fRZZ([param0, param1], wires=wires)


class StdEstimator(BaseEstimator):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.kwargs = kwargs
        self.X_, self.y_, self.classes_ = None, None, None

    @property
    def is_fitted(self):
        attrs = ["X_", "y_", "classes_"]
        attrs_values = [getattr(self, attr, None) for attr in attrs]
        return all([attr is not None for attr in attrs_values])

    def check_is_fitted(self):
        check_is_fitted(self)
        if not self.is_fitted:
            raise ValueError(f"{self.__class__.__name__} is not fitted.")
        
    def fit(self, X, y=None, **kwargs):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y
        return self


class MLKernel(StdEstimator):
    def __init__(
            self,
            size: Optional[int] = None,
            **kwargs
    ):
        self._size = size
        self.nb_workers = kwargs.get("nb_workers", 0)
        self.batch_size = kwargs.get("batch_size", 32)
        self._assume_symmetric = kwargs.get("assume_symmetric", True)
        self._assume_diag_one = kwargs.get("assume_diag_one", True)
        self._batch_size_try_counter = 0
        self._gram_type = kwargs.get("gram_type", "ndarray")
        if self._gram_type not in {"ndarray", "hdf5"}:
            raise ValueError(f"Unknown gram type: {self._gram_type}.")
        super().__init__(**kwargs)

    def _compute_default_size(self):
        return self.X_.shape[-1]

    def pre_initialize(self):
        pass

    def initialize_parameters(self):
        pass

    def fit(self, X, y=None, **kwargs):
        super().fit(X, y)

        if self._size is None:
            self._size = self._compute_default_size()
        self.pre_initialize()
        self.initialize_parameters()
        return self

    def transform(self, x):
        check_is_fitted(self)
        x = check_array(x)
        return x

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def single_distance(self, x0, x1, **kwargs):
        raise NotImplementedError(f"This method is not implemented for {self.__class__.__name__}.")

    def batch_distance_in_sequence(self, x0, x1, **kwargs):
        assert qml.math.ndim(x0) > 1, f"Expected x0 to be a batch of vectors, got {qml.math.shape(x0)}."
        if qml.math.ndim(x1) == qml.math.ndim(x0):
            distances = [self.single_distance(_x0, _x1, **kwargs) for _x0, _x1 in zip(x0, x1)]
        else:
            distances = [self.single_distance(x, x1, **kwargs) for x in x0]
        return qml.math.asarray(distances)

    def batch_distance(self, x0, x1, **kwargs):
        return self.batch_distance_in_sequence(x0, x1, **kwargs)

    def make_batch(self, x0, x1):
        x0 = check_array(x0)
        x1 = check_array(x1)
        b_x0 = qml.math.repeat(x0, qml.math.shape(x1)[0], axis=0)
        b_x1 = qml.math.tile(x1, (qml.math.shape(x0)[0], 1))
        return b_x0, b_x1

    def get_batch_size_for(self, length: int):
        if self.batch_size == "try":
            return length // 2 ** self._batch_size_try_counter
        elif self.batch_size == "sqrt":
            return int(np.sqrt(length))
        elif self.batch_size == 0:
            return 1
        elif self.batch_size < 0:
            return length
        elif self.batch_size > length:
            return length
        return self.batch_size

    def make_batches_generator_(self, x0, x1, **kwargs):
        x0 = check_array(x0)
        x1 = check_array(x1)
        x0_indexes = np.arange(qml.math.shape(x0)[0])
        x1_indexes = np.arange(qml.math.shape(x1)[0])
        b_x0_indexes = qml.math.repeat(x0_indexes, qml.math.shape(x1_indexes)[0], axis=0)
        b_x1_indexes = qml.math.tile(x1_indexes, (qml.math.shape(x0_indexes)[0],))
        batch_size = self.get_batch_size_for(qml.math.shape(b_x0_indexes)[0])
        n_batches = int(np.ceil(b_x0_indexes.shape[0] / batch_size))

        verbose = kwargs.pop("verbose", False)
        desc = kwargs.pop(
            "desc",
            f"{self.__class__.__name__}: "
            f"pairwise_distances_in_batch("
            f"x0:{qml.math.shape(x0)}, "
            f"x1:{qml.math.shape(x1)}, "
            f"batch_size={self.batch_size})"
        )
        p_bar = tqdm(total=n_batches, desc=desc, disable=not verbose)
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            ib_x0_indexes = b_x0_indexes[start_idx:end_idx]
            ib_x1_indexes = b_x1_indexes[start_idx:end_idx]
            yield x0[ib_x0_indexes], x1[ib_x1_indexes]
            p_bar.update(1)
        p_bar.close()

    def make_batches_generator(self, x, **kwargs) -> Tuple[iter, int]:
        length = qml.math.shape(x)[0]
        batch_size = self.get_batch_size_for(length)
        n_batches = int(np.ceil(length / batch_size))

        def _gen():
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = (i + 1) * batch_size
                yield x[start_idx:end_idx]

        return _gen(), n_batches

    def pairwise_distances_in_sequence(self, x0, x1, **kwargs):
        x0 = check_array(x0)
        x1 = check_array(x1)
        self.check_is_fitted()
        verbose = kwargs.pop("verbose", False)
        desc = kwargs.pop(
            "desc",
            f"{self.__class__.__name__}: pairwise_distances(x0:{qml.math.shape(x0)}, x1:{qml.math.shape(x1)})"
        )
        _list_results = pbt.apply_func_multiprocess(
            func=self.batch_distance,
            iterable_of_args=[(x0, b) for b in x1],
            iterable_of_kwargs=[kwargs for _ in range(len(x1))],
            nb_workers=self.nb_workers,
            verbose=verbose,
            desc=desc,
        )
        _result = np.stack(_list_results, axis=-1)
        return _result

    def pairwise_distances_in_batch_(self, x0, x1, **kwargs):
        x0 = check_array(x0)
        x1 = check_array(x1)
        self.check_is_fitted()
        p_bar: Optional[tqdm] = kwargs.pop("p_bar", None)
        p_bar_postfix_str = p_bar.postfix if p_bar is not None else ""
        if p_bar is not None:
            p_bar.set_postfix_str(f"{p_bar_postfix_str} (eta: ?, ?/?)")
        batched_distances = []
        start_time = time.perf_counter()
        n_data = qml.math.shape(x0)[0] * qml.math.shape(x1)[0]
        n_done = 0
        for i, (b_x0, b_x1) in enumerate(self.make_batches_generator(x0, x1, **kwargs)):
            batched_distances.append(self.batch_distance(b_x0, b_x1, **kwargs))
            if p_bar is not None:
                n_done += qml.math.shape(b_x0)[0]
                curr_time = time.perf_counter()
                eta = (curr_time - start_time) / n_done * (n_data - n_done)
                eta_fmt = datetime.timedelta(seconds=eta)
                p_bar.set_postfix_str(f"{p_bar_postfix_str} (eta: {eta_fmt}, {n_done}/{n_data})")
        distances = qml.math.concatenate(batched_distances, axis=0)
        return qml.math.reshape(distances, (qml.math.shape(x0)[0], qml.math.shape(x1)[0]))

    def pairwise_distances_in_batch(self, x0, x1, **kwargs):
        x0 = check_array(x0)
        x1 = check_array(x1)
        self.check_is_fitted()
        p_bar: Optional[tqdm] = kwargs.pop("p_bar", None)
        p_bar_postfix_str = p_bar.postfix if p_bar is not None else ""
        if p_bar is not None:
            p_bar.set_postfix_str(f"{p_bar_postfix_str} (eta: ?, ?%)")
        gram = np.zeros((qml.math.shape(x0)[0], qml.math.shape(x1)[0]))
        is_square = gram.shape[0] == gram.shape[1]
        triu_indices = np.stack(np.triu_indices(n=gram.shape[0], m=gram.shape[1], k=1), axis=-1)
        if is_square:
            indices = triu_indices
        else:
            tril_indices = np.stack(np.tril_indices(n=gram.shape[0], m=gram.shape[1], k=-1), axis=-1)
            indices = np.concatenate([triu_indices, tril_indices], axis=0)
        start_time = time.perf_counter()
        n_data = qml.math.shape(indices)[0]
        n_done = 0
        batch_gen, n_batches = self.make_batches_generator(indices, **kwargs)
        for i, b_idx in enumerate(batch_gen):
            b_x0, b_x1 = x0[b_idx[:, 0]], x1[b_idx[:, 1]]
            batched_distances = self.batch_distance(b_x0, b_x1, **kwargs)
            gram[b_idx[:, 0], b_idx[:, 1]] = batched_distances
            if p_bar is not None:
                n_done += qml.math.shape(b_x0)[0]
                curr_time = time.perf_counter()
                eta = (curr_time - start_time) / n_done * (n_data - n_done)
                eta_fmt = datetime.timedelta(seconds=eta)
                p_bar.set_postfix_str(f"{p_bar_postfix_str} (eta: {eta_fmt}, {100 * n_done / n_data:.2f}%)")
        if is_square:
            # gram[tril_indices[:, 0], tril_indices[:, 1]] = gram[tril_indices[:, 1], tril_indices[:, 0]]
            gram = gram + gram.T
        np.fill_diagonal(gram, 1.0)
        if p_bar is not None:
            p_bar.set_postfix_str(p_bar_postfix_str)
        return qml.math.array(gram)

    def pairwise_distances(self, x0, x1, **kwargs):
        x0 = check_array(x0)
        x1 = check_array(x1)
        self.check_is_fitted()
        if self.batch_size == 0:
            return self.pairwise_distances_in_sequence(x0, x1, **kwargs)
        # TODO: add support for batch_size = try
        return self.pairwise_distances_in_batch(x0, x1, **kwargs)
        try:
            return self.pairwise_distances_in_batch(x0, x1, **kwargs)
        except Exception as e:
            warnings.warn(
                f"Failed to compute pairwise distances in batch of size {self.batch_size}."
                f" Got err: ({e})."
                f"Computing pairwise distances in sequence.",
                RuntimeWarning
            )
            if kwargs.get("throw_errors", False):
                raise e
            return self.pairwise_distances_in_sequence(x0, x1, **kwargs)

    def compute_gram_matrix(self, x, **kwargs):
        kwargs.setdefault(
            "desc", f"{self.__class__.__name__}: "
                    f"compute_gram_matrix("
                    f"x:{qml.math.shape(x)}"
                    f", batch_size={self.batch_size}"
                    f")"
        )
        return self.pairwise_distances(x, x, **kwargs)


class NIFKernel(MLKernel):
    UNPICKLABLE_ATTRIBUTES = ['_device', "_qnode"]

    def __init__(
            self,
            size: Optional[int] = None,
            **kwargs
    ):
        super().__init__(size=size, **kwargs)
        self._qnode = None
        self._device = None
        self._parameters = self.kwargs.get("parameters", None)
        self.use_cuda = self.kwargs.get("use_cuda", False)
        self.simpify_qnode = self.kwargs.get("simplify_qnode", False)
        self.qnode_kwargs = dict(
            interface="torch" if self.use_cuda else "auto",
            diff_method=None,
            cache=False,
        )
        self.qnode_kwargs.update(self.kwargs.get("qnode_kwargs", {}))

    @property
    def size(self):
        return self._size

    @property
    def wires(self):
        return Wires(list(range(self.size)))

    @property
    def parameters(self):
        if getattr(self, "use_cuda", False):
            return self.cast_tensor_to_interface(self._parameters)
        return self._parameters

    @parameters.setter
    def parameters(self, parameters):
        self._parameters = pnp.asarray(parameters)

    @property
    def n_ops(self):
        return self.get_n_ops()

    @property
    def n_params(self):
        return self.get_n_params()

    @property
    def qnode(self):
        if self._qnode is None and self.is_fitted:
            self.pre_initialize()
        return self._qnode

    @qnode.setter
    def qnode(self, qnode):
        self._qnode = qnode

    @property
    def tape(self):
        qnode = self.qnode
        if qnode is None:
            return None
        if getattr(qnode, "tape", None) is None and self.is_fitted:
            self.compile_qnode()
        return getattr(qnode, "tape", None)

    def __getstate__(self):
        state = {
            k: v
            for k, v in self.__dict__.items()
            if k not in self.UNPICKLABLE_ATTRIBUTES
        }
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.pre_initialize()

    def cast_tensor_to_interface(self, tensor):
        if self.use_cuda:
            import torch
            return torch.tensor(tensor).cuda()
        return tensor

    def initialize_parameters(self):
        if self._parameters is None:
            n_parameters = self.kwargs.get("n_parameters", PATTERN_TO_NUM_PARAMS["pyramid"](self.wires))
            self._parameters = [pnp.random.uniform(0, 2 * np.pi, size=2) for _ in range(n_parameters)]

    def pre_initialize(self):
        self._device = NonInteractingFermionicDevice(wires=self.size)
        self._qnode = qml.QNode(self.circuit, self._device, **self.qnode_kwargs)
        if self.simpify_qnode:
            self._qnode = qml.simplify(self.qnode)

    def compile_qnode(self):
        self.batch_distance(self.X_[:2], self.X_[:2])

    def fit(self, X, y=None, **kwargs):
        super().fit(X, y)
        # TODO: optimize parameters with the given dataset
        # TODO: add kernel alignment optimization
        return self

    def circuit(self, x0, x1):
        MAngleEmbedding(x0, wires=self.wires)
        qml.broadcast(unitary=mrot_zz_template, pattern="pyramid", wires=self.wires, parameters=self.parameters)
        # TODO: ajouter des MROT avec des paramètres aléatoires en forme de pyramids
        # TODO: ajouter une fonction qui génère une séquence de wires en forme pyramidale.
        qml.adjoint(MAngleEmbedding)(x1, wires=self.wires)
        qml.adjoint(qml.broadcast)(unitary=mrot_zz_template, pattern="pyramid", wires=self.wires,
                                   parameters=self.parameters)
        projector: BasisStateProjector = qml.Projector(np.zeros(self.size), wires=self.wires)
        return qml.expval(projector)

    def single_distance(self, x0, x1, **kwargs):
        x0, x1 = self.cast_tensor_to_interface(x0), self.cast_tensor_to_interface(x1)
        return self.qnode(x0, x1)

    def batch_distance(self, x0, x1, **kwargs):
        x0, x1 = self.cast_tensor_to_interface(x0), self.cast_tensor_to_interface(x1)
        return self.qnode(x0, x1)

    def get_n_ops(self):
        if self.tape is None:
            return None
        return len(self.tape.operations)

    def get_n_params(self):
        if self.tape is None:
            return None
        return len(self.tape.get_parameters())

    def draw(self, **kwargs):
        logging_func = kwargs.pop("logging_func", print)
        name = kwargs.pop("name", self.__class__.__name__)
        if getattr(self, "qnode", None) is None or getattr(self.qnode, "tape", None) is None:
            _str = f"{name}: "
        else:
            n_ops = len(self.qnode.tape.operations)
            n_params = len(self.qnode.tape.get_parameters())
            _str = f"{name} ({n_ops} ops, {n_params} params): "
        if self.is_fitted:
            _str += f"\n{qml.draw(self.qnode, **kwargs)(self.X_[0], self.X_[-1])}\n"
        else:
            _str += f"None"
        if logging_func is not None:
            logging_func(_str)
        return _str

    def draw_mpl(
            self,
            fig: Optional[plt.Figure] = None,
            ax: Optional[plt.Axes] = None,
            **kwargs
    ):
        x0, x1 = self.cast_tensor_to_interface(self.X_[:2]), self.cast_tensor_to_interface(self.X_[-2:])
        _fig, _ax = qml.draw_mpl(self.qnode, expansion_strategy=kwargs.get("expansion_strategy", "device"))(x0, x1)
        if fig is None or ax is None:
            fig, ax = _fig, _ax
        else:
            ax_position = ax.get_position()
            ax.remove()
            fig.axes.append(_ax)
            _ax.set_position(ax_position)
            _ax.figure = fig
            fig.add_axes(_ax)
            ax = _ax

        filepath: Optional[str] = kwargs.get("filepath", None)
        if filepath is not None:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            fig.savefig(filepath)

        if kwargs.get("show", False):
            plt.show()

        return fig, ax


class FermionicPQCKernel(NIFKernel):
    r"""

    Inspired from: https://iopscience.iop.org/article/10.1088/2632-2153/acb0b4/meta#artAbst


    The size of the kernel is computed as

    .. math::
        \text{size} = \max\left(2, \lceil\log_2(\text{n features} + 2)\rceil\right)

    and the depth is computed as

    .. math::
        \text{depth} = \max\left(1, \left(\frac{\text{n features}}{\text{size}} - 1\right)\right)

    """

    available_entangling_mth = {"fcnot", "fast_fcnot", "fswap", "identity", "hadamard"}

    def __init__(
            self,
            size: Optional[int] = None,
            **kwargs
    ):
        super().__init__(size=size, **kwargs)
        self._data_scaling = kwargs.get("data_scaling", np.pi / 2)
        self._parameter_scaling = kwargs.get("parameter_scaling", np.pi / 2)
        self._depth = kwargs.get("depth", None)
        self._rotations = kwargs.get("rotations", "Y,Z")
        self._entangling_mth = kwargs.get("entangling_mth", "fswap")
        if self._entangling_mth not in self.available_entangling_mth:
            raise ValueError(f"Unknown entangling method: {self._entangling_mth}.")

    @property
    def depth(self):
        return self._depth

    @property
    def data_scaling(self):
        return self._data_scaling

    @property
    def rotations(self):
        return self._rotations

    def _compute_default_size(self):
        _size = max(2, int(np.ceil(np.log2(self.X_.shape[-1] + 2) - 1)))
        if _size % 2 != 0:
            _size += 1
        return _size

    def initialize_parameters(self):
        self._depth = self.kwargs.get("depth", max(1, (self.X_.shape[-1] // self.size) - 1))
        self.parameters = pnp.random.uniform(0.0, 1.0, size=self.X_.shape[-1])

    def ansatz(self, x):
        wires_double = PATTERN_TO_WIRES["double"](self.wires)
        wires_double_odd = PATTERN_TO_WIRES["double_odd"](self.wires)
        wires_patterns = [wires_double, wires_double_odd]
        for layer in range(self.depth):
            sub_x = x[..., layer * self.size: (layer + 1) * self.size]
            MAngleEmbedding(sub_x, wires=self.wires, rotations=self.rotations)
            fcnot_wires = wires_patterns[layer % len(wires_patterns)]
            for wires in fcnot_wires:
                if self._entangling_mth == "fast_fcnot":
                    FastfCNOT(wires=wires)
                elif self._entangling_mth == "fswap":
                    fSWAP(wires=wires)
                elif self._entangling_mth == "fcnot":
                    fCNOT(wires=wires)
                elif self._entangling_mth == "hadamard":
                    fH(wires=wires)
                elif self._entangling_mth == "identity":
                    pass
                else:
                    raise ValueError(f"Unknown entangling method: {self._entangling_mth}")
        return

    def circuit(self, x0, x1):
        theta_x0 = self._parameter_scaling * self.parameters + self.data_scaling * x0
        theta_x1 = self._parameter_scaling * self.parameters + self.data_scaling * x1
        self.ansatz(theta_x0)
        qml.adjoint(self.ansatz)(theta_x1)
        projector: BasisStateProjector = qml.Projector(np.zeros(self.size), wires=self.wires)
        return qml.expval(projector)


class WideFermionicPQCKernel(FermionicPQCKernel):
    def _compute_default_size(self):
        _size = max(2, int(np.ceil(np.sqrt(self.X_.shape[-1] + 2))))
        if _size % 2 != 0:
            _size += 1
        return _size


class PennylaneFermionicPQCKernel(FermionicPQCKernel):
    def __init__(self, size: Optional[int] = None, **kwargs):
        super().__init__(size=size, **kwargs)
        self._device_name = kwargs.get("device", "default.qubit")
        self._device_kwargs = kwargs.get("device_kwargs", {})

    def pre_initialize(self):
        self._device = qml.device(self._device_name, wires=self.size, **self._device_kwargs)
        self._qnode = qml.QNode(self.circuit, self._device, **self.qnode_kwargs)
        if self.simpify_qnode:
            self._qnode = qml.simplify(self.qnode)


class FixedSizeSVC(StdEstimator):
    def __init__(
            self,
            kernel_cls: Union[Type[MLKernel], str],
            kernel_kwargs: Optional[dict] = None,
            max_gram_size: Optional[int] = np.inf,
            cache_size: int = 1024,
            random_state: int = 0,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.kernel_cls = kernel_cls
        self.kernel_kwargs = kernel_kwargs or {}
        self.max_gram_size = max_gram_size or np.inf
        self.cache_size = cache_size
        self.random_state = random_state
        self.kernels = None
        self.estimators_ = None
        self.train_gram_matrices = None

    @property
    def kernel(self):
        return self.kernel_cls

    @property
    def n_kernels(self):
        return len(self.kernels)

    @property
    def kernel_size(self):
        if self.kernels is None:
            return None
        return getattr(self.kernels[0], "size", None)

    @property
    def kernel_n_ops(self):
        if self.kernels is None:
            return None
        return getattr(self.kernels[0], "n_ops", None)

    @property
    def kernel_n_params(self):
        if self.kernels is None:
            return None
        return getattr(self.kernels[0], "n_params", None)

    @property
    def n_features(self):
        if self.X_ is None:
            return None
        return qml.math.shape(self.X_)[-1]

    def __getattr__(self, item):
        if item in self.__dict__:
            return self.__dict__[item]
        if item.startswith("kernel_"):
            if self.kernels is None:
                return None
            return getattr(self.kernels[0], item[7:])
        raise AttributeError(f"{self.__class__.__name__} has no attribute {item}.")

    def split_data(self, x, y=None):
        x_shape = qml.math.shape(x)
        if x_shape[0] <= self.max_gram_size:
            return [x], [y]
        n_splits = int(np.ceil(x_shape[0] / self.max_gram_size))
        x_splits = np.array_split(x, n_splits)
        if y is not None:
            y_splits = np.array_split(y, n_splits)
        else:
            y_splits = [None for _ in range(n_splits)]
        return x_splits, y_splits

    def get_gram_matrices(self, X, **kwargs):
        p_bar: Optional[tqdm] = kwargs.get("p_bar", None)
        p_bar_postfix_str = p_bar.postfix if p_bar is not None else ""
        x_splits, _ = self.split_data(X)
        gram_matrices = []
        for i, sub_x in enumerate(x_splits):
            if p_bar is not None:
                p_bar.set_postfix_str(f"{p_bar_postfix_str}:[{i+1}/{len(x_splits)}]")
            gram_matrices.append(self.kernels[i].compute_gram_matrix(sub_x, **kwargs))
        if p_bar is not None:
            p_bar.set_postfix_str(p_bar_postfix_str)
        return gram_matrices

    def get_gram_matrix(self, X):
        gram_matrices = self.get_gram_matrices(X)
        return qml.math.block_diag(gram_matrices)

    def get_pairwise_distances_matrices(self, x0, x1, **kwargs):
        x0_splits, _ = self.split_data(x0)
        x1_splits, _ = self.split_data(x1)
        pairwise_distances = []
        for i, (sub_x0, sub_x1) in enumerate(zip(x0_splits, x1_splits)):
            pairwise_distances.append(self.kernels[i].pairwise_distances(sub_x0, sub_x1, **kwargs))
        return pairwise_distances

    def get_pairwise_distances(self, x0, x1):
        pairwise_distances = self.get_pairwise_distances_matrices(x0, x1)
        return qml.math.block_diag(pairwise_distances)
    
    def fit(self, X, y=None, **kwargs):
        super().fit(X, y, **kwargs)
        x_splits, y_splits = self.split_data(X, y)
        self.kernels = [
            self.kernel_cls(**self.kernel_kwargs).fit(sub_x, sub_y, **kwargs)
            for i, (sub_x, sub_y) in enumerate(zip(x_splits, y_splits))
        ]
        self.train_gram_matrices = self.get_gram_matrices(X, **kwargs)
        self.estimators_ = [
            svm.SVC(kernel="precomputed", random_state=self.random_state, cache_size=self.cache_size)
            for _ in range(self.n_kernels)
        ]
        for i, (gram_matrix, sub_y) in enumerate(zip(self.train_gram_matrices, y_splits)):
            self.estimators_[i].fit(gram_matrix, sub_y)
        return self
    
    def predict(self, X, **kwargs):
        self.check_is_fitted()
        if qml.math.shape(X) == qml.math.shape(self.X_) and qml.math.allclose(self.X_, X):
            gram_matrices = self.train_gram_matrices
        else:
            gram_matrices = self.get_pairwise_distances_matrices(X, self.X_, **kwargs)
        votes = np.zeros((qml.math.shape(X)[0], len(self.classes_)), dtype=int)
        predictions_stack = []
        for i, gram_matrix in enumerate(gram_matrices):
            predictions = self.estimators_[i].predict(gram_matrix)
            predictions_stack.append(predictions)
        predictions_stack = np.concatenate(predictions_stack, axis=0)
        votes[np.arange(qml.math.shape(X)[0]), predictions_stack] += 1
        return self.classes_[np.argmax(votes, axis=-1)]

    def score(self, X, y, **kwargs):
        self.check_is_fitted()
        pred = self.predict(X, **kwargs)
        return np.mean(np.isclose(pred, y).astype(float))

    def save(self, filepath):
        import pickle

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        return self

    @classmethod
    def load(cls, filepath):
        import pickle

        with open(filepath, "rb") as f:
            return pickle.load(f)
