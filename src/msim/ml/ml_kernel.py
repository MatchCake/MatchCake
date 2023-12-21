from typing import Optional
import sys
import os
import numpy as np
from matplotlib import pyplot as plt
from pennylane import numpy as pnp
from pennylane import AngleEmbedding
from pennylane.templates.broadcast import wires_pyramid, PATTERN_TO_NUM_PARAMS, PATTERN_TO_WIRES
from pennylane.ops.qubit.observables import BasisStateProjector
from sklearn.utils.estimator_checks import check_estimator
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics.pairwise import pairwise_kernels
import pennylane as qml
from pennylane.wires import Wires
import pythonbasictools as pbt

from ..devices.nif_device import NonInteractingFermionicDevice
from ..operations import MAngleEmbedding, MAngleEmbeddings, fRZZ, fCNOT


def mrot_zz_template(param0, param1, wires):
    fRZZ([param0, param1], wires=wires)


class MLKernel(BaseEstimator):
    def __init__(
            self,
            size: Optional[int] = None,
            **kwargs
    ):
        self._size = size
        self.nb_workers = kwargs.get("nb_workers", 0)
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
    
    def _compute_default_size(self):
        return self.X_.shape[-1]

    def pre_initialize(self):
        pass

    def initialize_parameters(self):
        pass
    
    def fit(self, X, y=None):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y

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
    
    def single_distance(self, x0, x1):
        raise NotImplementedError(f"This method is not implemented for {self.__class__.__name__}.")
    
    def batch_distance(self, x0, x1):
        assert qml.math.ndim(x0) > 1, f"Expected x0 to be a batch of vectors, got {qml.math.shape(x0)}."
        assert qml.math.ndim(x1) == 1, f"Expected x1 to be a single vector, got {qml.math.shape(x1)}."
        distances = [
            self.single_distance(x, x1)
            for x in x0
        ]
        return qml.math.asarray(distances)
        
    def pairwise_distances(self, x0, x1, **kwargs):
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
    
    def compute_gram_matrix(self, x, **kwargs):
        kwargs.setdefault("desc", f"{self.__class__.__name__}: compute_gram_matrix(x:{qml.math.shape(x)})")
        return self.pairwise_distances(x, x, **kwargs)


class NIFKernel(MLKernel):
    UNPICKLABLE_ATTRIBUTES = ['_device', ]
    
    def __init__(
            self,
            size: Optional[int] = None,
            **kwargs
    ):
        super().__init__(size=size, **kwargs)
        self.qnode = None
        self._device = None
        self._parameters = self.kwargs.get("parameters", None)
    
    @property
    def size(self):
        return self._size
    
    @property
    def wires(self):
        return Wires(list(range(self.size)))
    
    @property
    def parameters(self):
        return self._parameters
    
    @parameters.setter
    def parameters(self, parameters):
        self._parameters = pnp.asarray(parameters)
    
    def __getstate__(self):
        state = {
            k: v
            for k, v in self.__dict__.items()
            if k not in self.UNPICKLABLE_ATTRIBUTES
        }
        return state
    
    def initialize_parameters(self):
        if self._parameters is None:
            n_parameters = self.kwargs.get("n_parameters", PATTERN_TO_NUM_PARAMS["pyramid"](self.wires))
            self._parameters = [pnp.random.uniform(0, 2 * np.pi, size=2) for _ in range(n_parameters)]

    def pre_initialize(self):
        self._device = NonInteractingFermionicDevice(wires=self.size)
        self.qnode = qml.QNode(self.circuit, self._device, **self.kwargs.get("qnode_kwargs", {}))
    
    def fit(self, X, y=None):
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
        qml.adjoint(qml.broadcast)(unitary=mrot_zz_template, pattern="pyramid", wires=self.wires, parameters=self.parameters)
        projector: BasisStateProjector = qml.Projector(np.zeros(self.size), wires=self.wires)
        return qml.expval(projector)
    
    def single_distance(self, x0, x1):
        return self.qnode(x0, x1)
    
    def batch_distance(self, x0, x1):
        # return self.qnode(x0, x1)  TODO: implement batch_distance
        return super().batch_distance(x0, x1)

    def draw(self, **kwargs):
        logging_func = kwargs.pop("logging_func", print)
        name = kwargs.pop("name", self.__class__.__name__)
        _str = f"{name}: \n{qml.draw(self.qnode, **kwargs)(self.X_[0], self.X_[-1])}\n"
        if logging_func is not None:
            logging_func(_str)
        return _str

    def draw_mpl(
            self,
            fig: Optional[plt.Figure] = None,
            ax: Optional[plt.Axes] = None,
            **kwargs
    ):
        _fig, _ax = qml.draw_mpl(self.qnode)(self.X_[0], self.X_[-1])
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

        filepath = kwargs.get("filepath", None)
        if filepath is not None:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            fig.savefig(filepath)

        if kwargs.get("show", False):
            plt.show()

        return fig, ax


class FermionicPQCKernel(NIFKernel):
    """

    Inspired from: https://iopscience.iop.org/article/10.1088/2632-2153/acb0b4/meta#artAbst

    """
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
        return max(2, int(np.ceil(np.log2(self.X_.shape[-1] + 2) - 1)))
    
    def initialize_parameters(self):
        self._depth = self.kwargs.get("depth", max(1, (self.X_.shape[-1]//self.size) - 1))
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
                fCNOT(wires=wires)

    def circuit(self, x0, x1):
        theta_x0 = self._parameter_scaling * self.parameters + self.data_scaling * x0
        theta_x1 = self._parameter_scaling * self.parameters + self.data_scaling * x1
        self.ansatz(theta_x0)
        qml.adjoint(self.ansatz)(theta_x1)
        projector: BasisStateProjector = qml.Projector(np.zeros(self.size), wires=self.wires)
        return qml.expval(projector)
