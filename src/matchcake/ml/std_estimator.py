import os

from sklearn.base import BaseEstimator
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted, check_X_y

from ..utils import torch_utils


class StdEstimator(BaseEstimator):
    UNPICKLABLE_ATTRIBUTES = []
    _TO_NUMPY_ON_PICKLE = []

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

    def fit(self, X, y=None, **kwargs) -> "StdEstimator":
        r"""
        Fit the model with the given data.

        :param X: The input data.
        :param y: The target values.
        :param kwargs: Additional arguments.

        :keyword check_X_y: Whether to check the input data and target values. Default is True.

        :return: The fitted model.
        """
        if kwargs.get("check_X_y", True):
            X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y
        return self

    def __getstate__(self):
        state = {k: v for k, v in self.__dict__.items() if k not in self.UNPICKLABLE_ATTRIBUTES}
        for attr in self._TO_NUMPY_ON_PICKLE:
            if state.get(attr, None) is not None:
                state[attr] = torch_utils.to_numpy(state[attr])
        return state

    def save(self, filepath) -> "StdEstimator":
        """
        Save the model to a joblib file. If the given filepath does not end with ".joblib", it will be appended.

        :param filepath: The path to save the model.
        :type filepath: str
        :return: The model instance.
        :rtype: self.__class__
        """
        from joblib import dump

        if not filepath.endswith(".joblib"):
            filepath += ".joblib"

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        dump(self, filepath)
        return self

    @classmethod
    def load(cls, filepath) -> "StdEstimator":
        """
        Load a model from a saved file. If the extension of the file is not specified, this method will try to load
        the model from the file with the ".joblib" extension. If the file does not exist, it will try to load the model
        from the file with the ".pkl" extension. Otherwise, it will try without any extension.

        :param filepath: The path to the saved model.
        :type filepath: str
        :return: The loaded model.
        :rtype: StdEstimator
        """
        import pickle

        from joblib import load

        exts = [".joblib", ".pkl", ""]
        filepath_ext = None

        for ext in exts:
            if os.path.exists(filepath + ext):
                filepath_ext = filepath + ext
                break

        if filepath_ext is None:
            raise FileNotFoundError(f"Could not find the file: {filepath} with any of the following extensions: {exts}")

        if filepath_ext.endswith(".joblib"):
            return load(filepath_ext)
        with open(filepath_ext, "rb") as f:
            return pickle.load(f)
