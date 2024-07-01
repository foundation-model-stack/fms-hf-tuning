# Standard
import math

# Third Party
from simpleeval import DEFAULT_FUNCTIONS, DEFAULT_NAMES, EvalWithCompoundTypes


class MetricUnavailableError(Exception):
    def __init__(self, name):
        super().__init__(f"The metric '{name}' is not available")
        self.name = name


class UnavailableMetric:
    def __init__(self, name: str) -> None:
        self.err = MetricUnavailableError(name=name)

    def raise_error(self):
        raise self.err

    # https://docs.python.org/3/reference/datamodel.html#object.__lt__
    def __lt__(self, _):
        raise self.err

    def __le__(self, _):
        raise self.err

    def __eq__(self, other):
        if other is None:
            return True
        raise self.err

    # Use the default implementation
    # def __ne__(self, _):
    #     raise self.err

    def __gt__(self, _):
        raise self.err

    def __ge__(self, _):
        raise self.err

    def __getitem__(self, _):
        raise self.err

    # https://docs.python.org/3/reference/datamodel.html#object.__add__
    def __add__(self, _):
        raise self.err

    def __sub__(self, _):
        raise self.err

    def __mul__(self, _):
        raise self.err

    def __truediv__(self, _):
        raise self.err

    def __floordiv__(self, _):
        raise self.err

    def __mod__(self, _):
        raise self.err

    def __and__(self, _):
        raise self.err

    def __xor__(self, _):
        raise self.err

    def __or__(self, _):
        raise self.err

    # https://docs.python.org/3/reference/datamodel.html#object.__neg__
    def __neg__(self):
        raise self.err

    def __pos__(self):
        raise self.err

    def __abs__(self):
        raise self.err

    def __invert__(self):
        raise self.err

    # https://docs.python.org/3/reference/datamodel.html#object.__int__
    def __int__(self):
        raise self.err

    def __float__(self):
        raise self.err

    # https://docs.python.org/3/reference/datamodel.html#object.__round__
    def __round__(self, _=None):
        raise self.err

    def __trunc__(self):
        raise self.err

    def __floor__(self):
        raise self.err

    def __ceil__(self):
        raise self.err


class RuleEvaluator(EvalWithCompoundTypes):
    """Returns an evaluator that can be used to evaluate simple Python expressions."""

    def __init__(self, metrics: dict):
        all_names = {
            **metrics,
            **DEFAULT_NAMES.copy(),
        }
        all_funcs = {
            "abs": abs,
            "len": len,
            "round": round,
            "math_trunc": math.trunc,
            "math_floor": math.floor,
            "math_ceil": math.ceil,
            "math_sqrt": math.sqrt,
            **DEFAULT_FUNCTIONS.copy(),
        }
        super().__init__(functions=all_funcs, names=all_names)
        self.metrics = metrics

    def _eval_name(self, node):
        name = node.id
        if (
            isinstance(name, str)
            and name in self.metrics
            and self.metrics[name] is None
        ):
            return UnavailableMetric(name=name)
        return super()._eval_name(node)

    def _eval_subscript(self, node):
        key = self._eval(node.slice)
        if isinstance(key, UnavailableMetric):
            key.raise_error()
        return super()._eval_subscript(node)
