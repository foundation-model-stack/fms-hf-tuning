# Standard
from math import sqrt

# Third Party
from simpleeval import DEFAULT_FUNCTIONS, DEFAULT_NAMES, EvalWithCompoundTypes


def get_evaluator(metrics: dict) -> EvalWithCompoundTypes:
    """Returns an evaluator that can be used to evaluate simple Python expressions."""
    all_names = {
        **metrics,
        **DEFAULT_NAMES.copy(),
    }
    all_funcs = {
        "abs": abs,
        "len": len,
        "sqrt": sqrt,
        **DEFAULT_FUNCTIONS.copy(),
    }
    return EvalWithCompoundTypes(functions=all_funcs, names=all_names)
