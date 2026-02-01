from typing import Any, Dict, List, Optional

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedShuffleSplit, cross_validate
from tqdm import tqdm

from matchcake.typing import TensorLike


class CrossValidationOutput:
    ESTIMATOR_NAME_KEY = "estimator_name"

    def __init__(
        self,
        estimators: Dict[str, BaseEstimator],
        results_df: pd.DataFrame,
        *,
        estimator_name_key: str = ESTIMATOR_NAME_KEY,
    ):
        self.estimators = estimators
        self.results_df = results_df
        self.estimator_name_key = estimator_name_key

    @property
    def score_columns(self) -> List[str]:
        return [c for c in self.results_df.columns if c.startswith("test_") or c.startswith("train_")]


class CrossValidation:
    ESTIMATOR_NAME_KEY = "estimator_name"

    def __init__(
        self,
        estimators: Dict[str, BaseEstimator],
        x: TensorLike,
        y: Optional[TensorLike] = None,
        *,
        cv: Optional[Any] = None,
        cross_validate_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.estimators = estimators
        self.x = x
        self.y = y
        if cv is None:
            cv = StratifiedShuffleSplit(n_splits=20, test_size=0.2, random_state=0)
        self.cv = cv
        if cross_validate_kwargs is None:
            cross_validate_kwargs = {}
        self.cross_validate_kwargs = cross_validate_kwargs
        self.cross_validate_kwargs.setdefault("return_train_score", True)

    def run(self, verbose=True) -> CrossValidationOutput:
        results = []
        p_bar = tqdm(
            self.estimators.items(),
            disable=not verbose,
            desc="Cross-Validation",
        )

        for estimator_name, estimator in p_bar:
            cv_result = cross_validate(
                estimator=estimator, X=self.x, y=self.y, cv=self.cv, **self.cross_validate_kwargs
            )
            n_splits = len(cv_result[list(cv_result.keys())[0]])
            cv_result_list = [
                {self.ESTIMATOR_NAME_KEY: estimator_name, **{k: v[i] for k, v in cv_result.items()}}
                for i in range(n_splits)
            ]
            results.extend(cv_result_list)
        results_df = pd.DataFrame(results)
        cvo = CrossValidationOutput(self.estimators, results_df, estimator_name_key=self.ESTIMATOR_NAME_KEY)
        return cvo
