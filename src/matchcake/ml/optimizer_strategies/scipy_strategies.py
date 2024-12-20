from .optimizer_strategy import ScipyOptimizerStrategy


class COBYLAStrategy(ScipyOptimizerStrategy):
    NAME: str = "COBYLA"


class L_BFGSStrategy(ScipyOptimizerStrategy):
    NAME: str = "L-BFGS"


class SLSQPStrategy(ScipyOptimizerStrategy):
    NAME: str = "SLSQP"
