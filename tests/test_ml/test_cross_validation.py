import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from matchcake.ml.cross_validation import CrossValidation, CrossValidationOutput
from matchcake.ml.kernels import LinearNIFKernel


class TestCrossValidationOutput:
    @pytest.fixture
    def cvo(self):
        df = pd.DataFrame({"test_a": [1, 2, 3], "train_a": [4, 5, 6], "b": [4, 5, 6]})
        estimators = {"a": LinearNIFKernel(), "b": LinearNIFKernel()}
        return CrossValidationOutput(estimators, df)

    def test_init(self, cvo):
        assert isinstance(cvo.estimators, dict)
        assert isinstance(cvo.estimators[list(cvo.estimators.keys())[0]], BaseEstimator)
        assert isinstance(cvo.results_df, pd.DataFrame)

    def test_score_columns_property(self, cvo):
        assert set(cvo.score_columns) == {"test_a", "train_a"}


class TestCrossValidation:
    @pytest.fixture
    def cv(self):
        x = np.linspace(0, 10, num=100).reshape(10, 10)
        y = np.arange(10) % 2
        pipline = Pipeline(
            [
                ("kernel", LinearNIFKernel()),
                ("classifier", SVC(kernel="precomputed")),
            ]
        )
        return CrossValidation({"a": pipline, "b": pipline}, x=x, y=y)

    def test_init(self, cv):
        assert isinstance(cv.estimators, dict)
        assert isinstance(cv.estimators[list(cv.estimators.keys())[0]], BaseEstimator)

    def test_run(self, cv):
        cvo = cv.run(verbose=False)
        assert isinstance(cvo, CrossValidationOutput)
        assert set(cv.estimators.keys()) == set(cvo.estimators.keys())
        assert CrossValidation.ESTIMATOR_NAME_KEY in cvo.results_df.columns

    def test_run_with_kwargs(self, cv):
        kwargs = {
            "return_train_score": True,
            "return_indices": True,
            "return_estimator": True,
        }
        cv = CrossValidation(cv.estimators, x=cv.x, y=cv.y, cv=cv.cv, cross_validate_kwargs=kwargs)
        cvo = cv.run(verbose=False)
        assert isinstance(cvo, CrossValidationOutput)
        assert "train_score" in cvo.results_df.columns
        assert CrossValidation.INDICES_NAME_KEY in cvo.results_df.columns
        assert CrossValidation.ESTIMATOR_NAME_KEY in cvo.results_df.columns
