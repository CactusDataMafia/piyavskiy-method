import numpy as np
from typing import Tuple
from src.function_handler import f


def generate_x_points(start: int | float, end: int | float, *, amount: int = 600) -> np.ndarray:
    """
    Generate 1-D array of evenly spaced x-coordinates

    :param start: the starting value of the interval
    :param end: the end value of the interval
    :param amount: number of points to generate (default is 600)

    Returns numpy.ndarray of x-coordinates evenly spaced between start and end
    """
    return np.linspace(start, end, amount)


def find_down_left(point: float, L: float, x_points: np.ndarray) -> np.ndarray:
    """
    Compute the left descending line of the Lipschitz lower bound.

    :param point: Reference point
    :param L: Lipschitz constant that bounds the slope of f(x).
    :param x_points: Array of x-values over which to compute the line.
    Returns: numpy.ndarray: Values of the line f(point) + L * (x_points - point).
    """
    return f(point) + L * (x_points - point)


def find_down_right(point: float, L: float, x_points: np.ndarray) -> np.ndarray:
    """
    Compute the right descending line of the Lipschitz lower bound.

    :param point: Reference point
    :param L: Lipschitz constant that bounds the slope of f(x).
    :param x_points: Array of x-values over which to compute the line.

    Returns: numpy.ndarray: Values of the line f(point) - L * (x_points - point).
    """
    return f(point) - L * (x_points - point)


def intersection_point(point1: float, point2: float, L: float) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the intersection of two Lipschitz lower-bound lines.

    :param point1: Left endpoint of the interval
    :param point2: Right endpoint of the interval
    :param L: Lipschitz constant that bounds the slope of f(x)

    Returns:
        tuple: (x_intersection, y_lower, y_upper)
            - x_intersection (np.ndarray): x-coordinate of the intersection
            - y_lower (np.ndarray): Value of the lower envelope at the intersection
            - y_upper (np.ndarray): True function value f(x_intersection)
    """
    intersection = (f(point1) - f(point2)) / (2 * L) + (point1 + point2) / 2
    intersection_l = f(point1) - L * (intersection - point1)
    intersection_u = f(intersection)
    return intersection, intersection_l, intersection_u


def find_minimum(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    """
    Compute element-wise minimum of two arrays.

    :param arr1: First array
    :param arr2: Second array

    Returns: numpy.ndarray: Element-wise minimum values.
    """
    return np.minimum(arr1, arr2)
