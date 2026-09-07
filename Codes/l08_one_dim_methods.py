"""Lecture 8: three one-dimensional optimization methods on one example.

Run this file directly to print the numerical comparison and save its figure
in the current working directory.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import exp, sqrt
from pathlib import Path
from typing import Callable

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ScalarFunction = Callable[[float], float]


@dataclass
class Result:
    estimate: float
    iterations: int
    history: list[float]
    evaluations: dict[str, int]


def objective(x: float) -> float:
    return exp(x) + x * x - 2.0 * x


def derivative(x: float) -> float:
    return exp(x) + 2.0 * x - 2.0


def second_derivative(x: float) -> float:
    return exp(x) + 2.0


def bisection_stationary(
    df: ScalarFunction,
    left: float,
    right: float,
    location_tol: float,
    derivative_tol: float = 0.0,
    max_iter: int = 100,
) -> Result:
    """Bisect a continuous derivative with a sign-changing bracket."""
    if not left < right or location_tol <= 0 or derivative_tol < 0:
        raise ValueError(
            "Require left < right, location_tol > 0, and derivative_tol >= 0."
        )
    d_left, d_right = df(left), df(right)
    evaluations = 2
    if d_left == 0:
        return Result(left, 0, [left], {"df": evaluations})
    if d_right == 0:
        return Result(right, 0, [right], {"df": evaluations})
    if d_left * d_right > 0:
        raise ValueError("The endpoint derivatives must have opposite signs.")

    history: list[float] = []
    for reductions in range(max_iter + 1):
        midpoint = 0.5 * (left + right)
        history.append(midpoint)
        if 0.5 * (right - left) <= location_tol:
            return Result(midpoint, reductions, history, {"df": evaluations})
        if reductions == max_iter:
            break
        d_midpoint = df(midpoint)
        evaluations += 1
        if abs(d_midpoint) <= derivative_tol:
            return Result(midpoint, reductions, history, {"df": evaluations})
        if d_left * d_midpoint < 0:
            right, d_right = midpoint, d_midpoint
        else:
            left, d_left = midpoint, d_midpoint
    raise RuntimeError("Bisection reached max_iter before meeting the tolerance.")


def newton_stationary(
    df: ScalarFunction,
    d2f: ScalarFunction,
    x0: float,
    gradient_tol: float,
    step_tol: float = 1e-14,
    max_iter: int = 50,
) -> Result:
    """Apply Newton's method to the stationarity equation df(x) = 0."""
    if gradient_tol <= 0 or step_tol <= 0:
        raise ValueError("Tolerances must be positive.")
    x = float(x0)
    history: list[float] = []
    gradient_evaluations = 0
    hessian_evaluations = 0
    for iteration in range(max_iter + 1):
        history.append(x)
        gradient = df(x)
        gradient_evaluations += 1
        if abs(gradient) <= gradient_tol:
            return Result(
                x,
                iteration,
                history,
                {"df": gradient_evaluations, "d2f": hessian_evaluations},
            )
        if iteration == max_iter:
            break
        curvature = d2f(x)
        hessian_evaluations += 1
        if abs(curvature) <= 1e-14:
            raise RuntimeError("Newton step is undefined: curvature is too small.")
        step = -gradient / curvature
        if abs(step) <= step_tol * max(1.0, abs(x)):
            raise RuntimeError("Newton iteration stagnated before stationarity.")
        x += step
    raise RuntimeError("Newton's method reached max_iter before stationarity.")


def golden_section(
    f: ScalarFunction,
    left: float,
    right: float,
    location_tol: float,
    max_iter: int = 100,
) -> Result:
    """Minimize a strictly unimodal function by golden-section search."""
    if not left < right or location_tol <= 0:
        raise ValueError("Require left < right and location_tol > 0.")
    if 0.5 * (right - left) <= location_tol:
        return Result(0.5 * (left + right), 0, [0.5 * (left + right)], {"f": 0})

    rho = (sqrt(5.0) - 1.0) / 2.0
    lam = right - rho * (right - left)
    mu = left + rho * (right - left)
    f_lam, f_mu = f(lam), f(mu)
    evaluations = 2
    history = [0.5 * (left + right)]

    for iteration in range(1, max_iter + 1):
        if f_lam > f_mu:
            left = lam
            lam, f_lam = mu, f_mu
            mu = left + rho * (right - left)
            f_mu = f(mu)
        else:
            right = mu
            mu, f_mu = lam, f_lam
            lam = right - rho * (right - left)
            f_lam = f(lam)
        evaluations += 1
        history.append(0.5 * (left + right))
        if 0.5 * (right - left) <= location_tol:
            return Result(
                history[-1], iteration, history, {"f": evaluations}
            )
    raise RuntimeError("Golden-section search reached max_iter before tolerance.")


def run_demo(location_tol: float = 1e-4) -> tuple[float, dict[str, Result]]:
    reference = newton_stationary(
        derivative, second_derivative, x0=0.0, gradient_tol=1e-14
    ).estimate
    results = {
        "Bisection": bisection_stationary(
            derivative, -1.0, 1.0, location_tol=location_tol
        ),
        "Newton": newton_stationary(
            derivative, second_derivative, x0=0.0, gradient_tol=1e-8
        ),
        "Golden section": golden_section(
            objective, -1.0, 1.0, location_tol=location_tol
        ),
    }
    return reference, results


def save_comparison_figure(
    path: Path, reference: float, results: dict[str, Result]
) -> None:
    grid = np.linspace(-1.0, 1.0, 500)
    values = np.exp(grid) + grid**2 - 2.0 * grid
    colors = {"Bisection": "#0072B2", "Newton": "#D55E00", "Golden section": "#009E73"}
    markers = {"Bisection": "o", "Newton": "s", "Golden section": "^"}

    figure, axes = plt.subplots(1, 2, figsize=(10.0, 3.6))
    axes[0].plot(grid, values, color="black", linewidth=1.8, label=r"$f(x)$")
    for name, result in results.items():
        points = np.asarray(result.history)
        shown = points[: min(7, len(points))]
        y = np.exp(shown) + shown**2 - 2.0 * shown
        axes[0].scatter(
            shown,
            y,
            s=34,
            marker=markers[name],
            color=colors[name],
            label=name,
            zorder=3,
        )
    axes[0].axvline(reference, color="0.45", linestyle="--", linewidth=1.2)
    axes[0].set(xlabel=r"$x$", ylabel=r"$f(x)$", xlim=(-1.0, 1.0))
    axes[0].legend(frameon=False, fontsize=8)

    for name, result in results.items():
        errors = np.maximum(
            np.abs(np.asarray(result.history) - reference), np.finfo(float).eps
        )
        axes[1].semilogy(
            range(len(errors)),
            errors,
            marker=markers[name],
            color=colors[name],
            linewidth=1.5,
            markersize=4,
            label=name,
        )
    axes[1].set(xlabel="iteration index", ylabel=r"$|x^{(k)}-x^*|$")
    axes[1].grid(True, which="both", linewidth=0.4, alpha=0.45)
    axes[1].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    default_figure = Path("l08_method_comparison.pdf")
    parser = argparse.ArgumentParser()
    parser.add_argument("--location-tol", type=float, default=1e-4)
    parser.add_argument(
        "--figure", type=Path, default=default_figure,
        help="output image path (default: ./l08_method_comparison.pdf)",
    )
    args = parser.parse_args()

    reference, results = run_demo(args.location_tol)
    print(f"reference x* = {reference:.12f}")
    print(f"reference f* = {objective(reference):.12f}\n")
    print("method           updates    estimate         |error|       evaluations")
    print("-----------------------------------------------------------------------")
    for name, result in results.items():
        error = abs(result.estimate - reference)
        calls = ", ".join(f"{key}={value}" for key, value in result.evaluations.items())
        print(
            f"{name:18s} {result.iterations:5d}    {result.estimate: .9f}"
            f"    {error:.3e}    {calls}"
        )
    save_comparison_figure(args.figure, reference, results)
    print(f"\nfigure saved to {args.figure}")


if __name__ == "__main__":
    main()
