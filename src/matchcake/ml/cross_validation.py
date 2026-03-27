from typing import Any, Dict, List, Optional

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedShuffleSplit, cross_validate
from tqdm import tqdm

from matchcake.typing import TensorLike


class CrossValidationOutput:
    r"""
    The CrossValidation class defines an object to encapsulate the handling and the outputs of scikit-learn Cross-validation module.

    It can run multiple estimators one after another to test the performance of each of them in a reproducible manner.

    :ivar ESTIMATOR_NAME_KEY: Internal key name for estimator names
    :type ESTIMATOR_NAME_KEY: str
    :ivar estimators: The dictionary mapping estimator names to the estimator.
    :type estimators: dict
    :ivar results_df: The dataframe containing the results of the cross-validation test.
    :type results_df: Dataframe
    :ivar estimator_name_key: The key to the estimator name
    :type estimator_name_key: str
    """
    ESTIMATOR_NAME_KEY = "estimator_name"

    def __init__(
        self,
        estimators: Dict[str, BaseEstimator],
        results_df: pd.DataFrame,
        *,
        estimator_name_key: str = ESTIMATOR_NAME_KEY,
    ):
        """
        Initialize the CrossValidationOutput.

        :param estimators: The tested estimators
        :param results_df: The results of the cross-validation
        :param estimator_name_key: The name to access the estimator name in the Dataframe.        
        """
        self.estimators = estimators
        self.results_df = results_df
        self.estimator_name_key = estimator_name_key

    @property
    def score_columns(self) -> List[str]:
        """
        Returns only the columns with score values from the Dataframe.

        :return: A list of columns containing the score values. 
        """
        return [c for c in self.results_df.columns if c.startswith("test_") or c.startswith("train_")]


class CrossValidation:
    r"""
    The CrossValidation class defines an object to encapsulate the handling and the outputs of scikit-learn Cross-validation module.

    It can run multiple estimators one after another to test the performance of each of them in a reproducible manner.

    :ivar ESTIMATOR_NAME_KEY: Internal key name for estimator names.
    :type ESTIMATOR_NAME_KEY: str
    :ivar estimators: The dictionary mapping estimator names to the estimator.
    :type estimators: dict
    :ivar x: The independant variable.
    :type x: Any tensor-compatible object.
    :ivar y: The dependant variable.
    :type y: Any tensor-compatible object.
    :ivar cv: The Cross-Validation method. If none is passed, it will default to a StratifiedShuffleSplit with 20 splits and a train-test ration of 80-20
    :type cv: Cross-validator
    :ivar cross_validator_kwargs: Additional arguments to pass to the cross-validator. See https://scikit-learn.org/stable/modules/cross_validation.html#the-cross-validate-function-and-multiple-metric-evaluation
    :type cross_validator_kwargs: dict
    """
    ESTIMATOR_NAME_KEY = "estimator_name"

    def __init__(
        self,
        estimators: Dict[str, BaseEstimator],
        x: TensorLike,
        y: Optional[TensorLike] = None,
        *,
        cv: Optional[Any] = None,
        cross_validate_kwargs: Optional[Dict[str, bool]] = None,
    ):
        """
        Initialize the Cross-Validator with the estimators to validate and the data to train it on.

        Passed cross-validator kwargs must have the form:
        ```json
        {
            'return_train_score': True,
            'return_estimator': True,
            'return_estimator': True,
        }
        ```
        Each kwarg is optional and can be of value True or False. See https://scikit-learn.org/stable/modules/cross_validation.html#the-cross-validate-function-and-multiple-metric-evaluation for information on these parameters.

        :param estimators: Dictionary mapping estimator names to the estimator
        :param x: Independant variable to be processed.
        :param y: Dependant variable to be processed.
        :type y: Any tensor-compatible object.
        :param cv: The cross-validator. Will use a StratifiedShuffleSplit with 20 splits by default.
        :type cv: An initialised cross-validator.
        :ivar cross_validator_kwargs: Additional arguments to pass to the cross-validator. See https://scikit-learn.org/stable/modules/cross_validation.html#the-cross-validate-function-and-multiple-metric-evaluation .
        :type cross_validator_kwargs: A mapping between the arguments and their True/False value.
        """
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
        """
        Run the cross-validation. It will run all estimators with the same data and cross-validation method. Outputs of the cross-validation will be saved in a CrossValidationOutput object.

        :param verbose: If to show progress.
        :return: A CrossValidationOutput object.        
        """
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
                {
                    self.ESTIMATOR_NAME_KEY: estimator_name,
                    **{
                        k: {"train": v["train"][i], "test": v["test"][i]} if k == "indices" else v[i]
                        for k, v in cv_result.items()
                    },
                }
                for i in range(n_splits)
            ]
            results.extend(cv_result_list)
        results_df = pd.DataFrame(results)
        cvo = CrossValidationOutput(self.estimators, results_df, estimator_name_key=self.ESTIMATOR_NAME_KEY)
        return cvo
