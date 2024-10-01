import argparse
import os
from typing import Sequence, Optional, Any, Dict

import joblib
import json
from matplotlib import pyplot as plt

from matchcake.utils import torch_utils
import torch
import pennylane as qml
from pennylane.wires import Wires
from torch import nn
import numpy as np
import pythonbasictools as pbt


class TorchModel(nn.Module):
    DEFAULT_USE_CUDA = torch.cuda.is_available()
    DEFAULT_SEED = 0
    DEFAULT_SAVE_ROOT = os.path.join(os.getcwd(), "data", "models")
    DEFAULT_SAVE_DIR = None

    ATTRS_TO_STATE_DICT = []
    ATTRS_TO_HPARAMS = ["use_cuda", "seed"]
    ATTRS_TO_PICKLE = []
    ATTRS_TO_JSON = []
    MODEL_NAME = "TorchModel"
    DEFAULT_LOG_FUNC = print

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
        parser.add_argument(
            "--seed", type=int, default=cls.DEFAULT_SEED, help="The seed to use."
        )
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
        return parent_parser

    @classmethod
    def default_save_dir_from_args(cls, args):
        save_hash = pbt.hash_dict(vars(args))
        save_dir = f"{cls.MODEL_NAME}_{save_hash}"
        return save_dir

    def __init__(
            self,
            *,
            use_cuda: bool = DEFAULT_USE_CUDA,
            seed: int = DEFAULT_SEED,
            save_root: Optional[str] = DEFAULT_SAVE_ROOT,
            save_dir: Optional[str] = DEFAULT_SAVE_DIR,
            **kwargs
    ):
        super().__init__()
        self.log_func = kwargs.pop("log_func", self.DEFAULT_LOG_FUNC)
        self.use_cuda = use_cuda
        self.seed = seed
        self.save_root = save_root
        self.save_dir = save_dir or self.default_save_dir_from_args({"use_cuda": use_cuda, "seed": seed, **kwargs})
        self.kwargs = kwargs

        self.show_progress = kwargs.get("show_progress", False)

        self.parameters_rng = np.random.default_rng(seed=self.seed)
        self.torch_rng = torch.Generator()
        self.torch_rng.manual_seed(self.seed)
        self.initialize_parameters_()

    @property
    def wires(self):
        return Wires(list(range(self.n_qubits)))

    @property
    def save_path(self):
        return os.path.join(self.save_root, self.save_dir)

    @property
    def model_path(self):
        return os.path.abspath(os.path.join(self.save_path, "model.pt"))

    @property
    def hparams_path(self):
        return os.path.abspath(os.path.join(self.save_path, "hparams.json"))

    @property
    def pickles_path(self):
        return os.path.abspath(os.path.join(self.save_path, "pickles.pkl"))

    @property
    def jsons_path(self):
        return os.path.abspath(os.path.join(self.save_path, "jsons.json"))

    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
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
        if self.qnode.interface == "torch":
            tensor = torch_utils.to_tensor(tensor)
        else:
            tensor = torch_utils.to_numpy(tensor)
        if self.use_cuda:
            tensor = torch_utils.to_cuda(tensor)
        return tensor

    def initialize_parameters_(self):
        raise NotImplementedError("Subclasses must implement this method.")

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Subclasses must implement this method.")

    def predict(self, x, **kwargs):
        raise NotImplementedError("Subclasses must implement this method.")

    def score(self, x, y, **kwargs):
        raise NotImplementedError("Subclasses must implement this method.")

    def save_metrics(self, metrics: Dict[str, Any], filename: str = "metrics.json"):
        import json

        filepath = os.path.join(self.save_path, filename)
        os.makedirs(self.save_path, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(metrics, f, indent=4)
        return filepath

    def save(self) -> "TorchModel":
        os.makedirs(self.save_path, exist_ok=True)
        torch.save(self.state_dict(), self.model_path)
        self.save_hparams()
        self.save_pickles()
        self.save_jsons()
        return self

    def save_hparams(self) -> "TorchModel":
        os.makedirs(self.save_path, exist_ok=True)
        hparams = {attr: getattr(self, attr) for attr in self.ATTRS_TO_HPARAMS}
        with open(self.hparams_path, "w") as f:
            json.dump(hparams, f, indent=4)
        return self

    def save_pickles(self) -> "TorchModel":
        os.makedirs(self.models_dir, exist_ok=True)
        joblib.dump({attr: getattr(self, attr) for attr in self.ATTRS_TO_PICKLE}, self.pickles_path)
        return self

    def save_jsons(self) -> "TorchModel":
        os.makedirs(self.models_dir, exist_ok=True)
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

    def load(self, load_hparams: bool = True, **kwargs) -> "TorchModel":
        self.load_state_dict(torch.load(self.model_path))
        if load_hparams:
            self.load_hparams()
        self.load_pickles()
        self.load_jsons()
        return self

    def load_if_exists(self, load_hparams: bool = True, **kwargs) -> "TorchModel":
        if os.path.exists(self.model_path):
            self.load(load_hparams=load_hparams, **kwargs)
        return self

    @classmethod
    def from_folder(
            cls,
            folder: str,
            model_args: Optional[Sequence[Any]] = None,
            model_kwargs: Optional[Dict[str, Any]] = None,
            **kwargs
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
            **kwargs
    ) -> "TorchModel":
        model = cls(*model_args, **model_kwargs)
        model.save_root = os.path.dirname(folder)
        model.save_dir = os.path.basename(folder)
        kwargs.setdefault("load_hparams", True)
        model.load_if_exists(**kwargs)
        return model

    def draw_mpl(
            self, fig: Optional[plt.Figure] = None, ax: Optional[plt.Axes] = None, **kwargs
    ):
        x0, x1 = np.random.rand(2, *self.input_shape), np.random.rand(
            2, *self.input_shape
        )
        x0, x1 = self.cast_tensor_to_interface(x0), self.cast_tensor_to_interface(x1)
        _fig, _ax = qml.draw_mpl(
            self.qnode, expansion_strategy=kwargs.get("expansion_strategy", "device")
        )(x0, x1)
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
