import argparse
import json
import os
import time
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import joblib
import numpy as np
import pennylane as qml
import pythonbasictools as pbt
import torch
import tqdm
from matplotlib import pyplot as plt
from pennylane.wires import Wires
from torch import nn

from ...templates import TensorLike
from ...utils import torch_utils
from ..optimizer_strategies import get_optimizer_strategy
from ..parameters_initialisation_strategies import (
    get_parameters_initialisation_strategy,
)


class TorchModel(nn.Module):
    DEFAULT_USE_CUDA = torch.cuda.is_available()
    DEFAULT_SEED = 0
    DEFAULT_SAVE_ROOT = os.path.join(os.getcwd(), "data", "models")
    DEFAULT_SAVE_DIR = None

    ATTRS_TO_STATE_DICT = []
    ATTRS_TO_HPARAMS = [
        "use_cuda",
        "seed",
        "max_grad_norm",
        "learning_rate",
        "optimizer",
        "params_init",
        "fit_patience",
    ]
    ATTRS_TO_PICKLE = ["fit_time", "start_fit_time", "end_fit_time"]
    ATTRS_TO_JSON = [
        "fit_history",
    ]
    MODEL_NAME = "TorchModel"
    DEFAULT_LOG_FUNC = print

    DEFAULT_OPTIMIZER = "SimulatedAnnealing"
    DEFAULT_PARAMETERS_INITIALISATION_STRATEGY = "Random"
    DEFAULT_MAX_GRAD_NORM = 1.0
    DEFAULT_LEARNING_RATE = 2e-4
    DEFAULT_FIT_PATIENCE = 10

    @classmethod
    def add_model_specific_args(cls, parent_parser: Optional[argparse.ArgumentParser] = None):
        if parent_parser is None:
            parent_parser = argparse.ArgumentParser()
        parser = parent_parser.add_argument_group("Torch Model")
        parser.add_argument(
            "--use_cuda",
            type=bool,
            action=argparse.BooleanOptionalAction,
            default=cls.DEFAULT_USE_CUDA,
            help="Whether to use CUDA.",
        )
        parser.add_argument("--seed", type=int, default=cls.DEFAULT_SEED, help="The seed to use.")
        parser.add_argument(
            "--save_root",
            type=str,
            default=cls.DEFAULT_SAVE_ROOT,
            help="The root directory to save the models. The current model will be save in save_root/save_dir.",
        )
        parser.add_argument(
            "--save_dir",
            type=str,
            default=cls.DEFAULT_SAVE_DIR,
            help="The directory to save the models. The current model will be save in save_root/save_dir.",
        )
        parser.add_argument(
            "--max_grad_norm",
            type=float,
            default=cls.DEFAULT_MAX_GRAD_NORM,
            help="The maximum gradient norm to use.",
        )
        parser.add_argument(
            "--learning_rate",
            type=float,
            default=cls.DEFAULT_LEARNING_RATE,
            help="The learning rate to use.",
        )
        parser.add_argument(
            "--optimizer",
            type=str,
            default=cls.DEFAULT_OPTIMIZER,
            help="The optimizer to use.",
        )
        parser.add_argument(
            "--params_init",
            type=str,
            default=cls.DEFAULT_PARAMETERS_INITIALISATION_STRATEGY,
            help="The parameters initialisation strategy to use.",
        )
        parser.add_argument(
            "--fit_patience",
            type=int,
            default=cls.DEFAULT_FIT_PATIENCE,
            help="The patience to use during the fit.",
        )
        return parent_parser

    @classmethod
    def default_save_dir_from_args(cls, args):
        if isinstance(args, dict):
            args_dict = args.copy()
        else:
            args_dict = vars(args)
        save_hash = pbt.hash_dict(args_dict)
        save_dir = f"{cls.MODEL_NAME}_{save_hash}"
        return save_dir

    def __init__(
        self,
        *,
        use_cuda: bool = DEFAULT_USE_CUDA,
        seed: int = DEFAULT_SEED,
        save_root: Optional[str] = DEFAULT_SAVE_ROOT,
        save_dir: Optional[str] = DEFAULT_SAVE_DIR,
        max_grad_norm: float = DEFAULT_MAX_GRAD_NORM,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        optimizer: str = DEFAULT_OPTIMIZER,
        params_init: str = DEFAULT_PARAMETERS_INITIALISATION_STRATEGY,
        fit_patience: Optional[int] = DEFAULT_FIT_PATIENCE,
        **kwargs,
    ):
        super().__init__()
        self.log_func = kwargs.pop("log_func", self.DEFAULT_LOG_FUNC)
        self.use_cuda = use_cuda
        self.seed = seed
        self.save_root = save_root
        self.save_dir = save_dir or self.default_save_dir_from_args(
            {
                "use_cuda": use_cuda,
                "seed": seed,
                "max_grad_norm": max_grad_norm,
                "learning_rate": learning_rate,
                "optimizer": optimizer,
                "params_init": params_init,
                "fit_patience": fit_patience,
                **kwargs,
            }
        )
        self.max_grad_norm = max_grad_norm
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.params_init = params_init
        self.fit_patience = fit_patience
        self.kwargs = kwargs

        self.show_progress = kwargs.get("show_progress", False)

        self.optimizer_strategy = get_optimizer_strategy(self.optimizer)
        self.parameters_initialisation_strategy = get_parameters_initialisation_strategy(self.params_init)

        self.parameters_rng = np.random.default_rng(seed=self.seed)
        self.torch_rng = torch.Generator()
        self.torch_rng.manual_seed(self.seed)

        self.fit_time = None
        self.start_fit_time = None
        self.end_fit_time = None
        self.fit_history = []
        self._cache_last_np_cost = None
        self._cache_last_cost = None
        self.p_bar_postfix = {}
        self._fit_p_bar = None
        self.fit_args = None
        self.fit_kwargs = None

    @property
    def save_path(self):
        return os.path.join(self.save_root, self.save_dir)

    @property
    def model_path(self):
        return os.path.abspath(os.path.join(self.save_path, "model.pt"))

    @property
    def best_model_path(self):
        return os.path.abspath(os.path.join(self.save_path, "best_model.pt"))

    @property
    def hparams_path(self):
        return os.path.abspath(os.path.join(self.save_path, "hparams.json"))

    @property
    def pickles_path(self):
        return os.path.abspath(os.path.join(self.save_path, "pickles.pkl"))

    @property
    def jsons_path(self):
        return os.path.abspath(os.path.join(self.save_path, "jsons.json"))

    @property
    def torch_device(self):
        return torch.device("cuda" if self.use_cuda else "cpu")

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        state = super().state_dict(*args, destination=destination, prefix=prefix, keep_vars=keep_vars)
        for attr in self.ATTRS_TO_STATE_DICT:
            state[attr] = getattr(self, attr)
        return state

    def load_state_dict(self, state_dict, strict=True):
        for attr in self.ATTRS_TO_STATE_DICT:
            if attr in state_dict:
                setattr(self, attr, state_dict.pop(attr))
        super().load_state_dict(state_dict, strict=strict)
        return self

    def cast_tensor_to_interface(self, tensor):
        tensor = torch_utils.to_tensor(tensor)
        if self.use_cuda:
            tensor = torch_utils.to_cuda(tensor)
        return tensor

    def initialize_parameters_(self):
        raise NotImplementedError("Subclasses must implement this method.")

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Subclasses must implement this method.")

    def predict(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement this method.")

    def score(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement this method.")

    def save_metrics(self, metrics: Dict[str, Any], filename: str = "metrics.json"):
        import json

        filepath = os.path.join(self.save_path, filename)
        os.makedirs(self.save_path, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(metrics, f, indent=4)
        return filepath

    def save(self, model_path: Optional[str] = None) -> "TorchModel":
        if model_path is None:
            model_path = self.model_path
        os.makedirs(self.save_path, exist_ok=True)
        torch.save(self.state_dict(), model_path)
        self.save_hparams()
        self.save_pickles()
        self.save_jsons()
        return self

    def save_best(self) -> "TorchModel":
        self.save(model_path=self.best_model_path)
        return self

    def save_hparams(self) -> "TorchModel":
        os.makedirs(self.save_path, exist_ok=True)
        hparams = {attr: getattr(self, attr) for attr in self.ATTRS_TO_HPARAMS}
        with open(self.hparams_path, "w") as f:
            json.dump(hparams, f, indent=4)
        return self

    def save_pickles(self) -> "TorchModel":
        os.makedirs(self.save_path, exist_ok=True)
        joblib.dump(
            {attr: getattr(self, attr) for attr in self.ATTRS_TO_PICKLE},
            self.pickles_path,
        )
        return self

    def save_jsons(self) -> "TorchModel":
        os.makedirs(self.save_path, exist_ok=True)
        jsons = {attr: getattr(self, attr) for attr in self.ATTRS_TO_JSON}
        with open(self.jsons_path, "w") as f:
            json.dump(jsons, f, indent=4)
        return self

    def load_hparams(self) -> "TorchModel":
        with open(self.hparams_path, "r") as f:
            hparams = json.load(f)
        for attr, value in hparams.items():
            setattr(self, attr, value)
        return self

    def load_pickles(self) -> "TorchModel":
        pickles = joblib.load(self.pickles_path)
        for attr, value in pickles.items():
            setattr(self, attr, value)
        return self

    def load_jsons(self) -> "TorchModel":
        with open(self.jsons_path, "r") as f:
            jsons = json.load(f)
        for attr, value in jsons.items():
            setattr(self, attr, value)
        return self

    def load(self, model_path: Optional[str] = None, load_hparams: bool = True, **kwargs) -> "TorchModel":
        if model_path is None:
            model_path = self.model_path
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.load_state_dict(torch.load(model_path))
        if load_hparams:
            self.load_hparams()
        self.load_pickles()
        self.load_jsons()
        return self

    def load_best(self, **kwargs) -> "TorchModel":
        self.load(model_path=self.best_model_path, **kwargs)
        return self

    def load_if_exists(self, model_path: Optional[str] = None, load_hparams: bool = True, **kwargs) -> "TorchModel":
        if model_path is None:
            model_path = self.model_path
        if os.path.exists(model_path):
            self.load(model_path=model_path, load_hparams=load_hparams, **kwargs)
        return self

    def load_best_if_exists(self, **kwargs) -> "TorchModel":
        self.load_if_exists(model_path=self.best_model_path, **kwargs)
        return self

    @classmethod
    def from_folder(
        cls,
        folder: str,
        model_args: Optional[Sequence[Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> "TorchModel":
        model = cls(*model_args, **model_kwargs)
        model.save_root = os.path.dirname(folder)
        model.save_dir = os.path.basename(folder)
        kwargs.setdefault("load_hparams", True)
        model.load(**kwargs)
        return model

    @classmethod
    def from_folder_or_new(
        cls,
        folder: str,
        model_args: Optional[Sequence[Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> "TorchModel":
        model = cls(*model_args, **model_kwargs)
        model.save_root = os.path.dirname(folder)
        model.save_dir = os.path.basename(folder)
        kwargs.setdefault("load_hparams", True)
        model.load_if_exists(**kwargs)
        return model

    def draw_mpl(self, fig: Optional[plt.Figure] = None, ax: Optional[plt.Axes] = None, **kwargs):
        x0, x1 = np.random.rand(2, *self.input_shape), np.random.rand(2, *self.input_shape)
        x0, x1 = self.cast_tensor_to_interface(x0), self.cast_tensor_to_interface(x1)
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

    def __del__(self):
        try:
            self.save()
        except Exception as e:
            self.log_func(f"Error while saving the model when deleting: {e}")

    def update_parameters(
        self,
        parameters: List[Union[torch.nn.Parameter, Tuple[str, torch.nn.Parameter]]],
    ):
        if len(parameters) == 0:
            return self
        if isinstance(parameters[0], tuple):
            parameters = [param[1] for param in parameters]
        parameters = torch_utils.to_tensor(parameters)
        for self_p, new_p in zip(self.parameters(), parameters):
            self_p.data = new_p.data
        return self

    def cost(self, *args, **kwargs) -> TensorLike:
        """
        This function should return the overall cost of the model. If the cost depend on a dataset, this function
        should be overridden and the cost should be computed using the whole dataset.

        This function will be called by the fit_closure function which will be used by the optimizer to optimize the
        model's parameters.

        :Note: The arguments and keyword arguments passed to the fit function are stored in the fit_args and fit_kwargs.
            They could be useful to compute the cost in this function. The dataset can be passed as a keyword argument
            to the fit function and accessed in this function using kwargs.get("dataset", None) as an example.

        :param args: Arguments to pass to the forward function.
        :param kwargs: Keyword arguments to pass to the forward function.
        :return: The overall cost of the model.
        """
        target = torch_utils.to_tensor(kwargs.get("target", 1.0))
        y_hat = torch_utils.to_tensor(self(*args, **kwargs))
        return torch.nn.MSELoss()(y_hat, target)

    def fit_closure(self, parameters: Optional[List[torch.nn.Parameter]] = None, *args, **kwargs) -> TensorLike:
        """
        Assigns the parameters to the model and returns the cost.

        :param parameters: The parameters to assign to the model.
        :type parameters: Optional[List[torch.nn.Parameter]]
        :param args: Arguments to pass to the cost function.
        :type args: Any
        :param kwargs: Keyword arguments to pass to the cost
        :type kwargs: Any
        :return: The cost.
        """
        self.zero_grad()
        if parameters is not None:
            self.update_parameters(parameters)
        self._cache_last_cost = self.cost(*args, **kwargs)
        self._cache_last_np_cost = float(torch_utils.to_numpy(self._cache_last_cost))
        return self._cache_last_cost

    def fit_callback(self, *args, **kwargs):
        is_best = self._cache_last_np_cost <= np.nanmin(self.fit_history) if len(self.fit_history) > 0 else True
        self.fit_history.append(self._cache_last_np_cost)
        self.save()
        self.plot_fit_history(show=False, save=True)
        if is_best:
            self.save_best()
        if self._fit_p_bar is not None:
            best_cost = np.nanmin(self.fit_history)
            prev_cost = self.fit_history[-2] if len(self.fit_history) > 1 else np.nan
            self.p_bar_postfix = {
                "Prev Cost": prev_cost,
                "Cost": self._cache_last_np_cost,
                "Best Cost": best_cost,
                **kwargs.get("postfix", {}),
            }
            self._fit_p_bar.set_postfix(self.p_bar_postfix)
            self._fit_p_bar.n = min(self._fit_p_bar.total, len(self.fit_history))
            self._fit_p_bar.refresh()
            if self._fit_p_bar.disable:
                self.log_func("-" * 120)
                self.log_func(f"Iteration {len(self.fit_history)}: {self.p_bar_postfix}")
                self.log_func(str(self._fit_p_bar))
                self.log_func("-" * 120)
        if self.fit_patience is not None:
            if len(self.fit_history) > self.fit_patience:
                if np.allclose(np.diff(self.fit_history[-self.fit_patience :]), 0.0):
                    self.optimizer_strategy.stop_training_flag = True
                    self.p_bar_postfix["stop_training_flag"] = True
                    self.p_bar_postfix["stop_training_reason"] = "loss is not changing"
                    self._fit_p_bar.set_postfix(self.p_bar_postfix)
        return self

    def fit(self, *args, n_iterations: int = 100, n_init_iterations: int = 1, **kwargs):
        self.fit_args = args
        self.fit_kwargs = kwargs
        if not kwargs.get("overwrite", False):
            self.load_if_exists()
        self.start_fit_time = time.perf_counter()
        self._fit_p_bar = tqdm.tqdm(
            total=n_iterations * n_init_iterations,
            initial=len(self.fit_history),
            desc=kwargs.get("desc", "Optimizing"),
            disable=not kwargs.get("verbose", getattr(self.q_device, "show_progress", False)),
        )
        current_iteration = len(self.fit_history) // n_init_iterations
        current_init_iteration = len(self.fit_history) % n_init_iterations

        for i in range(current_init_iteration, n_init_iterations):
            if i > 0:
                next_parameters = self.parameters_initialisation_strategy.get_next_parameters(
                    step_id=i,
                    n_layers=self.n_layers,
                    parameters_rng=self.parameters_rng,
                    seed=self.seed,
                    current_named_parameters=list(self.named_parameters()),
                )
                self.update_parameters(next_parameters)
                self._fit_p_bar.set_postfix(
                    {
                        "Cost": "N/A",
                        "Best Cost": np.nanmin(self.fit_history),
                        "Init Iteration": i,
                    }
                )
            self.optimizer_strategy.set_parameters(
                list(self.named_parameters()),
                learning_rate=self.learning_rate,
                max_grad_norm=self.max_grad_norm,
            )
            self.requires_grad_(self.optimizer_strategy.REQUIRES_GRAD)
            best_parameters = self.optimizer_strategy.optimize(
                closure=self.fit_closure,
                callback=self.fit_callback,
                n_iterations=max(n_iterations - current_iteration, 0),
            )
            self.update_parameters(best_parameters)
            self.save()
            current_iteration = 0

        self.end_fit_time = time.perf_counter()
        if self.fit_time is None:
            self.fit_time = self.end_fit_time - self.start_fit_time
        else:
            self.fit_time += self.end_fit_time - self.start_fit_time
        self._fit_p_bar.close()
        if kwargs.get("load_best", True):
            self.load_best()
        return self

    def plot_fit_history(self, fig=None, ax=None, **kwargs):
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        if kwargs.get("log_history", False):
            self.log_func(f"Fit history: {self.fit_history}")
        ax.plot(self.fit_history)
        ax.set_xlabel("Iterations [-]")
        ax.set_ylabel("Cost [-]")
        try:
            if kwargs.get("save", True):
                fig.savefig(os.path.join(self.save_path, "fit_history.pdf"))
        except Exception as e:
            self.log_func(f"Error while saving the fit history: {e}")
        try:
            if kwargs.get("show", False):
                plt.show()
        except Exception as e:
            self.log_func(f"Error while plotting the fit history: {e}")
        try:
            if kwargs.get("close", True):
                plt.close(fig)
        except Exception as e:
            self.log_func(f"Error while closing the figure: {e}")
        return fig, ax
