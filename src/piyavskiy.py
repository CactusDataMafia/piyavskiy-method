from typing import Callable
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from src.utils import find_down_left, find_down_right, find_minimum, intersection_point
from tqdm import tqdm

def piyavskiy_method(
        f: Callable[[float], float],
        x_points: np.ndarray,
        y_points: np.ndarray,
        A: float,
        B: float,
        L: float,
        EPS: float = 0.1,
        max_iter: int = 100,
        save_plots: bool = True
) -> list[dict]:
    """
    Run the Piyavskiy method for global optimization on a given function.

    Args:
        f: Target function to minimize.
        x_points: Array of x-values for plotting.
        y_points: Array of f(x) values for plotting.
        A: Left boundary of search interval.
        B: Right boundary of search interval.
        L: Lipschitz constant.
        EPS: Stopping tolerance (difference between upper and lower bounds).
        max_iter: Maximum number of iterations (default=100).
        save_plots: Whether to save plot images for each iteration.

    Returns:
        list of dict: Each dict contains iteration data:
            - iteration number
            - u_new
            - p_{n-1}(u_n)
            - f(u_n)
            - delta
            - plot_path (if saved)
    """
    points = [A, B]
    results = []

    for i in tqdm(range(max_iter), desc="Вычисление итераций", ncols=100, leave=False, colour="green"):
        points = sorted(points)
        inters = []

        # Find intersection points
        for j in range(len(points) - 1):
            x1, x2 = points[j], points[j + 1]
            mid, y_mid_l, y_mid_u = intersection_point(x1, x2, L, f)
            inters.append((mid, y_mid_l, y_mid_u))

        # Choose new point with minimal lower bound
        u_new, y_min_l, y_min_u = min(inters, key=lambda x: x[1])
        delta = y_min_u - y_min_l

        # Visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x_points, y_points, label="W(x)")

        for p in points:
            y_down_r = find_minimum(find_down_right(p, L, x_points, f), y_points)
            y_down_l = find_minimum(find_down_left(p, L, x_points, f), y_points)
            ax.plot(x_points, y_down_r, "g--", alpha=0.4)
            ax.plot(x_points, y_down_l, "g--", alpha=0.4)
            ax.scatter(p, f(p), color="black")


        p_n = np.maximum.reduce([f(p) - L * np.abs(x_points - p) for p in points])
        ax.plot(x_points, p_n, "g", lw=2)
        ax.scatter(u_new, y_min_l, color="red", s=60, label="$p_{n - 1}(u_n)$")
        ax.scatter(u_new, y_min_u, color="yellow", s=60, label="$W(u_n)$")
        ax.plot([u_new, u_new], [y_min_l, y_min_u], color="black", linestyle="--")
        ax.legend()
        ax.grid(alpha=0.3)

        if save_plots:
            output_dir = Path("results") / "piyavskiy_plots"
            output_dir.mkdir(parents=True, exist_ok=True)
            plot_path = output_dir / f"piyavskiy_iter_{i+1}.png"
            fig.savefig(plot_path, bbox_inches="tight", dpi=120)
        else:
            plt.show()
        plt.close(fig)

        # Save results for DataFrame
        results.append({
            "iteration": i + 1,
            "u": u_new,
            "$p_{n-1}(u_n)$": y_min_l,
            "$f(u_n)$": y_min_u,
            "delta": delta
        })
        # Check stop condition
        if delta < EPS:
            break

        # Add new point
        points.append(u_new)

    return results
