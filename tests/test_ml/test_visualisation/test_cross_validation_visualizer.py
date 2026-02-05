import pandas as pd
import pytest
from matplotlib import pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from matchcake.ml import CrossValidationVisualizer
from matchcake.ml.cross_validation import CrossValidationOutput
from matchcake.ml.kernels import LinearNIFKernel


class TestCrossValidationVisualizer:
    @pytest.fixture
    def cvo(self):
        df = pd.DataFrame(
            {"test_a": [1, 2, 3], "train_a": [4, 5, 6], "b": [4, 5, 6], "estimator_name": ["a", "a", "a"]}
        )
        pipline = Pipeline(
            [
                ("kernel", LinearNIFKernel()),
                ("classifier", SVC(kernel="precomputed")),
            ]
        )
        estimators = {"a": pipline}
        return CrossValidationOutput(estimators, df)

    @pytest.fixture
    def cvv(self, cvo):
        return CrossValidationVisualizer(cvo)

    def test_init(self, cvo, cvv):
        assert cvv.cvo == cvo

    @pytest.mark.parametrize(
        "estimator_name_key, score_name_map",
        [
            (None, {"test_a": "accuracy"}),
            ("Model", {"test_a": "accuracy"}),
            ("Model", None),
        ],
    )
    def test_plot(self, cvv, estimator_name_key, score_name_map):
        ax = cvv.plot(estimator_name_key=estimator_name_key, score_name_map=score_name_map)
        assert isinstance(ax, plt.Axes)
