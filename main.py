from src.utils import generate_x_points
from src.piyavskiy import piyavskiy_method
from src.export import create_csv
import numpy as np
from functools import wraps

allowed_names = {
    # базовые константы
    "pi": np.pi,
    "e": np.e,

    # тригонометрические
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "arcsin": np.arcsin,
    "arccos": np.arccos,
    "arctan": np.arctan,

    # гиперболические
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,

    # экспоненты и логарифмы
    "exp": np.exp,
    "log": np.log,        # натуральный логарифм
    "log10": np.log10,

    # степени и корни
    "sqrt": np.sqrt,
    "abs": np.abs,
    "power": np.power,

    # часто встречающиеся операции
    "sign": np.sign,
    "floor": np.floor,
    "ceil": np.ceil,

    # сам np — чтобы можно было писать np.sin, np.exp и т.д.
    "np": np,
}

def formula(expr: str):
    """
    Decorator that injects a mathematical expression string
    into a wrapped evaluation function.

    The decorator takes an expression (for example "sin(3 * x) + 0.3 * x")
    and passes it as the first argument when the wrapped function is called.
    This allows the wrapped function to handle all evaluation logic itself.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(x, *args, **kwargs):
            return func(expr, x, *args, **kwargs)

        return wrapper

    return decorator


def f(func: str, x: float | int | np.ndarray, context=allowed_names) -> np.ndarray:
    """
    Evaluate a user-defined mathematical expression.

    param func: Expression string using variable 'x'
    param x: Input value for evaluation.
    param context (optional): Allowed functions and constants

    Returns: np.ndarray: Result of evaluating the expression.
    """
    if context is None:
        context = {}

    context = {**context, "__builtins__": {}, "x": x}
    return eval(func.strip(), context)

def main():
    L = float(input("Введите константу Липшница: "))
    A = float(input("Введите нижную границу интервала: "))
    B = float(input("Введите верхную границу интервала: "))
    expr = input("Введите целевую функцию: ")

    decorated_f = formula(expr)(f)
    x_points = generate_x_points(A, B)
    y_points = decorated_f(x_points)

    results = piyavskiy_method(
        f=decorated_f,
        x_points=x_points,
        y_points=y_points,
        A=A,
        B=B,
        L=L,
        max_iter=50,
        save_plots=True
    )

    create_csv(results)

if __name__ == "__main__":
    main()







