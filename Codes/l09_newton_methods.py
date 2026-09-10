"""Newton-method examples for MA3236 Lecture 9.

The script compares three methods on a common quartic sum-of-squares function
from the starting point (1.23, 0.51): pure Newton, a shifted Newton method, and Newton with
a shifted-direction safeguard and Armijo backtracking line search.
NumPy is required; Matplotlib is needed
only for plots.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


Array = np.ndarray
COMMON_START = (1.23, 0.51)
COMMON_MINIMIZER = (1.0, 0.0)
# Shared by the final three-method comparisons, in run_comparison() order.
COMPARISON_COLORS = ("#7b3294", "#008837", "#2166ac")
COMPARISON_LINESTYLES = ("-", "--", "-.")
COMPARISON_MARKERS = ("o", "s", "^")


def objective(x: Array) -> float:
    """Common quartic sum-of-squares objective."""
    x1, x2 = x
    return float((x1**2 + x2 - 1.0) ** 2 + (x1 + x2**2 - 1.0) ** 2)


def gradient(x: Array) -> Array:
    """Gradient of the common objective."""
    x1, x2 = x
    a, b = x1**2 + x2 - 1.0, x1 + x2**2 - 1.0
    return np.array([4.0*x1*a + 2.0*b, 2.0*a + 4.0*x2*b])


def hessian(x: Array) -> Array:
    """Hessian of the common objective."""
    x1, x2 = x
    return np.array(
        [[12.0*x1**2 + 4.0*x2 - 2.0, 4.0*(x1 + x2)],
         [4.0*(x1 + x2), 4.0*x1 + 12.0*x2**2 - 2.0]]
    )


def double_well_objective(x: Array) -> float:
    """Rotated two-dimensional double-well objective."""
    x1, x2 = x
    return float((x1 + x2) ** 4 / 16.0 - x1 * x2)


def double_well_gradient(x: Array) -> Array:
    """Gradient of the rotated double-well objective."""
    x1, x2 = x
    s = x1 + x2
    return np.array([0.25 * s**3 - x2, 0.25 * s**3 - x1])


def double_well_hessian(x: Array) -> Array:
    """Hessian of the rotated double-well objective."""
    s = float(np.sum(x))
    curvature = 0.75 * s**2
    return np.array(
        [[curvature, curvature - 1.0],
         [curvature - 1.0, curvature]]
    )


@dataclass
class Result:
    name: str
    points: list[Array]
    values: list[float]
    grad_norms: list[float]
    alphas: list[float]
    shifts: list[float]
    f_evals: int
    g_evals: int
    h_evals: int
    solves: int
    newton_slopes: list[float] | None = None

    @property
    def updates(self) -> int:
        return len(self.points) - 1


def _validate_run(x0: Array, tol: float, max_iter: int) -> Array:
    """Check inputs without changing the mathematical stopping test."""
    x = np.asarray(x0, dtype=float)
    if x.ndim != 1 or not np.all(np.isfinite(x)):
        raise ValueError("x0 must be a finite one-dimensional array.")
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("tol must be finite and positive.")
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 0:
        raise ValueError("max_iter must be a nonnegative integer.")
    return x.copy()


def _validate_shifts(shift_seed: float, growth: float) -> None:
    if (not np.isfinite(shift_seed) or not np.isfinite(growth)
            or shift_seed <= 0.0 or growth <= 1.0):
        raise ValueError("Require finite shift_seed > 0 and growth > 1.")


def _armijo_step(
    function: Callable[[Array], float], x: Array, d: Array,
    f: float, slope: float, sigma: float, beta: float, max_backtracks: int,
) -> tuple[float, int]:
    """Return the accepted step and number of trial function evaluations.

    The initial trial is 1. This helper never evaluates a gradient. A finite
    trial cap and stagnation check report numerical failure instead of hanging.
    """
    if not np.isfinite(slope) or slope >= 0.0 or not np.all(np.isfinite(d)):
        raise ValueError("Armijo backtracking requires a finite descent direction.")
    alpha = 1.0
    for trial in range(max_backtracks + 1):
        trial_x = x + alpha * d
        if np.array_equal(trial_x, x):
            raise RuntimeError("Armijo backtracking stagnated at floating-point precision.")
        with np.errstate(over="ignore", invalid="ignore"):
            f_trial = function(trial_x)
        if np.isfinite(f_trial) and f_trial <= f + sigma * alpha * slope:
            return alpha, trial + 1
        alpha *= beta
    raise RuntimeError("Armijo backtracking exhausted max_backtracks.")


def shifted_direction(
    h: Array, g: Array, shift_seed: float = 0.1, growth: float = 2.0,
) -> tuple[Array, float]:
    """Find a positive-definite shifted model and its descent direction.

    Test tau = 0, shift_seed, growth * shift_seed, ... . This tests curvature,
    not objective decrease. A nonzero gradient is required.
    """
    _validate_shifts(shift_seed, growth)
    h, g = np.asarray(h, dtype=float), np.asarray(g, dtype=float)
    if (g.ndim != 1 or h.shape != (g.size, g.size)
            or not np.all(np.isfinite(h)) or not np.all(np.isfinite(g))
            or not np.allclose(h, h.T) or not np.any(g)):
        raise ValueError("Require a finite symmetric Hessian and a finite nonzero gradient.")
    tau = 0.0
    for _ in range(100):
        b = h + tau * np.eye(g.size)
        try:
            np.linalg.cholesky(b)
            d = np.linalg.solve(b, -g)
        except np.linalg.LinAlgError:
            tau = shift_seed if tau == 0.0 else growth * tau
            continue
        if np.all(np.isfinite(d)) and float(g @ d) < 0.0:
            return d, tau
        tau = shift_seed if tau == 0.0 else growth * tau
    raise RuntimeError("Could not find a finite shifted descent direction.")


def descent_shifted_direction(
    h: Array, g: Array, shift_seed: float = 0.1, growth: float = 2.0,
) -> tuple[Array, float, int]:
    """Find a shifted descent direction after the raw Newton trial failed.

    Test tau = shift_seed, growth * shift_seed, ... . Accept any finite solve
    with g @ d < 0; the shifted matrix may be indefinite. No positive-
    definiteness test is needed. Return all successful linear solves, including
    candidates rejected by the directional-derivative test.
    """
    _validate_shifts(shift_seed, growth)
    h, g = np.asarray(h, dtype=float), np.asarray(g, dtype=float)
    if (g.ndim != 1 or h.shape != (g.size, g.size)
            or not np.all(np.isfinite(h)) or not np.all(np.isfinite(g))
            or not np.allclose(h, h.T) or not np.any(g)):
        raise ValueError("Require a finite symmetric Hessian and a finite nonzero gradient.")
    tau, solves = shift_seed, 0
    for _ in range(100):
        if not np.isfinite(tau):
            break
        b = h + tau * np.eye(g.size)
        try:
            d = np.linalg.solve(b, -g)
            solves += 1
        except np.linalg.LinAlgError:
            tau *= growth
            continue
        slope = float(g @ d)
        if np.all(np.isfinite(d)) and np.isfinite(slope) and slope < 0.0:
            return d, tau, solves
        tau *= growth
    raise RuntimeError("Could not find a finite shifted descent direction.")


def pure_newton(x0: Array, tol: float = 1e-8, max_iter: int = 100) -> Result:
    """Full-step Newton method."""
    x = _validate_run(x0, tol, max_iter)
    points: list[Array] = []
    values: list[float] = []
    grad_norms: list[float] = []
    f_evals = g_evals = h_evals = solves = 0

    for iteration in range(max_iter + 1):
        f, g = objective(x), gradient(x)
        if not np.isfinite(f) or not np.all(np.isfinite(g)):
            raise RuntimeError("Pure Newton produced a non-finite iterate or objective.")
        f_evals += 1
        g_evals += 1
        points.append(x.copy())
        values.append(f)
        grad_norms.append(float(np.linalg.norm(g)))
        if grad_norms[-1] <= tol:
            return Result(
                "Pure Newton", points, values, grad_norms,
                [1.0] * (len(points) - 1), [0.0] * (len(points) - 1),
                f_evals, g_evals, h_evals, solves,
            )
        if iteration == max_iter:
            break
        h = hessian(x)
        h_evals += 1
        d = np.linalg.solve(h, -g)
        solves += 1
        x = x + d

    raise RuntimeError("Pure Newton reached max_iter before stationarity.")


def shifted_newton(
    x0: Array,
    tol: float = 1e-8,
    max_iter: int = 100,
    shift_seed: float = 0.1,
    growth: float = 2.0,
) -> Result:
    """Shifted Newton with a positive-definite quadratic model.

    At each outer iteration, first try the unshifted Hessian. If it is not
    positive definite, enlarge a nonnegative diagonal shift geometrically until
    the shifted model matrix is positive definite. The full step need not
    decrease the objective.
    """
    x = _validate_run(x0, tol, max_iter)
    _validate_shifts(shift_seed, growth)
    points: list[Array] = []
    values: list[float] = []
    grad_norms: list[float] = []
    shifts: list[float] = []
    f_evals = g_evals = h_evals = solves = 0

    for iteration in range(max_iter + 1):
        f, g = objective(x), gradient(x)
        if not np.isfinite(f) or not np.all(np.isfinite(g)):
            raise RuntimeError("Shifted Newton produced a non-finite iterate or objective.")
        f_evals += 1
        g_evals += 1
        points.append(x.copy())
        values.append(f)
        grad_norms.append(float(np.linalg.norm(g)))
        if grad_norms[-1] <= tol:
            return Result(
                "Shifted Newton", points, values, grad_norms,
                [1.0] * (len(points) - 1), shifts,
                f_evals, g_evals, h_evals, solves,
            )
        if iteration == max_iter:
            break
        h = hessian(x)
        h_evals += 1

        d, tau = shifted_direction(h, g, shift_seed, growth)
        solves += 1
        shifts.append(tau)
        x = x + d

    raise RuntimeError("Shifted Newton reached max_iter before stationarity.")


def newton_armijo(
    x0: Array,
    tol: float = 1e-8,
    max_iter: int = 100,
    sigma: float = 1e-4,
    beta: float = 0.5,
    shift_seed: float = 0.1,
    growth: float = 2.0,
    max_backtracks: int = 100,
) -> Result:
    """Newton--Armijo with a shifted-direction safeguard.

    First try the Newton direction. If it is unavailable or not descent,
    replace it using descent_shifted_direction. A negative directional
    derivative suffices; the shifted matrix need not be positive definite.
    Only then run Armijo backtracking.
    """
    if not 0.0 < sigma < 1.0 or not 0.0 < beta < 1.0:
        raise ValueError("Require sigma and beta in (0, 1).")
    if not isinstance(max_backtracks, (int, np.integer)) or max_backtracks < 0:
        raise ValueError("max_backtracks must be a nonnegative integer.")
    x = _validate_run(x0, tol, max_iter)
    _validate_shifts(shift_seed, growth)
    points: list[Array] = []
    values: list[float] = []
    grad_norms: list[float] = []
    alphas: list[float] = []
    shifts: list[float] = []
    newton_slopes: list[float] = []
    f_evals = g_evals = h_evals = solves = 0

    for iteration in range(max_iter + 1):
        f, g = objective(x), gradient(x)
        if not np.isfinite(f) or not np.all(np.isfinite(g)):
            raise RuntimeError("Newton--Armijo produced a non-finite iterate or objective.")
        f_evals += 1
        g_evals += 1
        points.append(x.copy())
        values.append(f)
        grad_norms.append(float(np.linalg.norm(g)))
        if grad_norms[-1] <= tol:
            return Result(
                "Newton--Armijo", points, values, grad_norms, alphas, shifts,
                f_evals, g_evals, h_evals, solves,
                newton_slopes=newton_slopes,
            )
        if iteration == max_iter:
            break
        h = hessian(x)
        h_evals += 1

        tau = 0.0
        try:
            d = np.linalg.solve(h, -g)
            solves += 1
            slope = float(g @ d)
        except np.linalg.LinAlgError:
            slope = float("nan")
        newton_slopes.append(slope)

        # Safeguard the direction before adjusting its step length.
        if not np.isfinite(slope) or slope >= 0.0 or not np.all(np.isfinite(d)):
            d, tau, shifted_solves = descent_shifted_direction(h, g, shift_seed, growth)
            solves += shifted_solves
            slope = float(g @ d)

        # slope < 0: sufficiently small steps satisfy the Armijo condition.
        alpha, trial_evals = _armijo_step(
            objective, x, d, f, slope, sigma, beta, max_backtracks,
        )
        f_evals += trial_evals
        alphas.append(alpha)
        shifts.append(tau)
        x = x + alpha * d

    raise RuntimeError("Newton--Armijo reached max_iter before stationarity.")


def run_comparison(start: tuple[float, float] = COMMON_START) -> list[Result]:
    """Run the same three algorithms and stopping tests from a chosen start."""
    x0 = np.array(start)
    return [pure_newton(x0), shifted_newton(x0), newton_armijo(x0)]


def run_shifting_benefit(
    tol: float = 1e-8,
    max_iter: int = 20,
    shift_seed: float = 0.1,
    growth: float = 2.0,
) -> tuple[list[Array], list[Array], list[float], list[float]]:
    """Compare pure and shifted Newton on a double-well problem."""
    _validate_shifts(shift_seed, growth)
    x0 = _validate_run(np.array([0.4, -0.3]), tol, max_iter)

    pure_x = x0.copy()
    pure_points = [pure_x.copy()]
    for iteration in range(max_iter + 1):
        pure_g = double_well_gradient(pure_x)
        if np.linalg.norm(pure_g) <= tol:
            break
        if iteration == max_iter:
            raise RuntimeError("Pure Newton reached max_iter in the shifting example.")
        pure_d = np.linalg.solve(double_well_hessian(pure_x), -pure_g)
        pure_x = pure_x + pure_d
        pure_points.append(pure_x.copy())

    shifted_x = x0.copy()
    shifted_points = [shifted_x.copy()]
    shifted_values = [double_well_objective(shifted_x)]
    shifts: list[float] = []

    for iteration in range(max_iter + 1):
        g = double_well_gradient(shifted_x)
        if np.linalg.norm(g) <= tol:
            return pure_points, shifted_points, shifted_values, shifts
        if iteration == max_iter:
            break
        h = double_well_hessian(shifted_x)
        d, tau = shifted_direction(h, g, shift_seed, growth)
        shifts.append(tau)
        shifted_x = shifted_x + d
        shifted_points.append(shifted_x.copy())
        shifted_values.append(double_well_objective(shifted_x))

    raise RuntimeError("The shifting-benefit example reached max_iter.")


def run_divergence_example(
    tol: float = 1e-8, max_iter: int = 20,
) -> tuple[list[float], list[float], list[float]]:
    """Reproduce the slide f(x)=sqrt(1+x^2), starting from x=2.

    Return the first three pure-Newton updates, all safeguarded Newton--Armijo
    iterates, and accepted step lengths. The Hessian is positive everywhere,
    so no shift is needed. Backtracking uses sigma=1e-4 and beta=1/2.
    """
    x = _validate_run(np.array([2.0]), tol, max_iter)
    pure_points = [float(x[0])]
    for _ in range(3):
        pure_points.append(-pure_points[-1]**3)

    armijo_points = [float(x[0])]
    alphas: list[float] = []
    function = lambda point: float(np.hypot(1.0, point[0]))
    for iteration in range(max_iter + 1):
        g = float(x[0] / np.hypot(1.0, x[0]))
        if abs(g) <= tol:
            return pure_points, armijo_points, alphas
        if iteration == max_iter:
            break
        d = -x * (1.0 + x*x)
        alpha, _ = _armijo_step(
            function, x, d, function(x), float(g*d[0]), 1e-4, 0.5, 100,
        )
        x = x + alpha*d
        armijo_points.append(float(x[0]))
        alphas.append(alpha)
    raise RuntimeError("The scalar Newton--Armijo example reached max_iter.")


def print_divergence_example() -> None:
    pure_points, armijo_points, alphas = run_divergence_example()
    print("\nf(x)=sqrt(1+x^2), x^(0)=2; positive Hessian everywhere")
    print("Pure Newton (first three updates): "
          + " -> ".join(f"{x:.9g}" for x in pure_points) + " -> ... (diverges)")
    print("Newton--Armijo: " + " -> ".join(f"{x:.9g}" for x in armijo_points))
    print(f"Accepted steps: {alphas}; {len(alphas)} updates to gradient tolerance 1e-8.")
    print("The final iterate is approximately zero, not exactly zero.")


def print_results(results: list[Result]) -> None:
    start = tuple(float(value) for value in results[0].points[0])
    print(f"Common quartic function, x^(0) = {start}, tolerance = 1e-8")
    print("method              updates   f evals   g evals   H evals   solves      final f")
    for result in results:
        print(
            f"{result.name:20s}{result.updates:8d}{result.f_evals:10d}"
            f"{result.g_evals:10d}{result.h_evals:10d}{result.solves:9d}"
            f"{result.values[-1]:13.3e}"
        )
    for result in results:
        increases = np.flatnonzero(np.diff(result.values) > 1e-12)
        print(f"\n{result.name}: {len(increases)} objective increases")
        for k in increases:
            print(f"  k={k} -> {k+1}: {result.values[k]:.6g} -> {result.values[k+1]:.6g}")
    armijo = results[2]
    for k, tau in enumerate(armijo.shifts):
        if tau > 0.0:
            print(f"Newton--Armijo safeguard at k={k}: "
                  f"g^T d_N = {armijo.newton_slopes[k]:.6g}, "
                  f"tau = {tau:g}, alpha = {armijo.alphas[k]:g}")
    if armijo.alphas:
        first_final_unit = next(
            (k for k in range(len(armijo.alphas))
             if all(alpha == 1.0 for alpha in armijo.alphas[k:])), None,
        )
        if first_final_unit is not None:
            print(f"Newton--Armijo: alpha^(k) = 1 for every k >= {first_final_unit}")


def print_iteration_tables(results: list[Result]) -> None:
    """Print the complete data underlying the selected-iterate slide tables."""
    for result in results:
        print(f"\n{result.name}: full iteration table")
        print("k       x1             x2             f             ||g||        tau     alpha")
        for k, (x, f, gn) in enumerate(zip(result.points, result.values, result.grad_norms)):
            step = (f"{result.shifts[k]:8g}{result.alphas[k]:9g}"
                    if k < result.updates else "    stop")
            print(f"{k:2d}{x[0]:14.7g}{x[1]:14.7g}{f:14.7g}{gn:14.7g} {step}")


def print_shifting_benefit() -> None:
    pure_points, shifted_points, _, shifts = run_shifting_benefit()
    pure_limit = pure_points[-1]
    shifted_limit = shifted_points[-1]
    positive_shifts = [shift for shift in shifts if shift > 0.0]
    print("\nRotated double well, x^(0) = (0.4, -0.3), tolerance = 1e-8")
    print(
        f"Pure Newton: {len(pure_points) - 1} updates, "
        f"limit = ({pure_limit[0]:.6g}, {pure_limit[1]:.6g}) (saddle)"
    )
    print(
        f"Shifted Newton: {len(shifted_points) - 1} updates, "
        f"limit = ({shifted_limit[0]:.6g}, {shifted_limit[1]:.6g})"
    )
    print(f"Positive shifts: {positive_shifts}")


def save_plot(results: list[Result], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    zorders = [3, 4, 3]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), constrained_layout=True)
    for result, color, linestyle, marker, zorder in zip(
        results, COMPARISON_COLORS, COMPARISON_LINESTYLES, COMPARISON_MARKERS, zorders
    ):
        iterations = np.arange(len(result.values))
        axes[0].semilogy(
            iterations, np.maximum(result.values, 1e-30),
            marker=marker, markersize=3.2, linewidth=1.6,
            linestyle=linestyle, color=color, label=result.name, zorder=zorder,
        )
        axes[1].semilogy(
            iterations, np.maximum(result.grad_norms, 1e-30),
            marker=marker, markersize=3.2, linewidth=1.6,
            linestyle=linestyle, color=color, label=result.name, zorder=zorder,
        )

    axes[0].set_title("Objective value")
    axes[0].set_xlabel("iteration $k$")
    axes[0].set_ylabel("$f(x^{(k)})$")
    axes[1].set_title("Stationarity measure")
    axes[1].set_xlabel("iteration $k$")
    axes[1].set_ylabel(r"$\|\nabla f(x^{(k)})\|_2$")
    for axis in axes:
        axis.grid(True, which="both", linewidth=0.45, alpha=0.45)
    axes[1].legend(frameon=False, fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def save_oscillation_plot(results: list[Result], output: Path) -> None:
    """Compare early objective rebounds at full scale and at a vertical zoom."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    last_k = min(8, max(result.updates for result in results))
    visible_values = [np.asarray(result.values[:last_k + 1]) for result in results]
    full_ceiling = max(float(values.max()) for values in visible_values)
    # The vertical zoom resolves the smaller rebound of shifted Newton.
    zoom_ceiling = 1.25 * max(float(values[2:].max()) for values in visible_values[1:])
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), constrained_layout=True)
    for result, color, style, marker in zip(
        results, COMPARISON_COLORS, COMPARISON_LINESTYLES, COMPARISON_MARKERS,
    ):
        values = result.values[:last_k + 1]
        for axis in axes:
            axis.plot(range(len(values)), values, color=color, linestyle=style,
                      marker=marker, markersize=4, linewidth=1.8, label=result.name)
    axes[0].set(title="Early iterations: full objective scale",
                ylim=(-0.025 * full_ceiling, 1.18 * full_ceiling))
    axes[1].set(title="Vertical zoom: smaller objective changes",
                ylim=(-0.025 * zoom_ceiling, zoom_ceiling))
    peak_k = int(np.argmax(visible_values[0]))
    axes[0].annotate(f"{results[0].values[peak_k]:.3f}",
                     (peak_k, results[0].values[peak_k]), xytext=(0, 9),
                     textcoords="offset points", ha="center", fontsize=10,
                     color=COMPARISON_COLORS[0])
    for result, color in zip(results[:2], COMPARISON_COLORS):
        for k in np.flatnonzero(np.diff(result.values) > 1e-12) + 1:
            if k <= last_k and result.values[k] < zoom_ceiling:
                axes[1].annotate(f"{result.values[k]:.5f}", (k, result.values[k]),
                                 xytext=(7, 10), textcoords="offset points",
                                 fontsize=9, color=color)
    axes[1].text(0.03, 0.98, "The large pure Newton excursion is outside this zoom.",
                 transform=axes[1].transAxes, va="top", fontsize=8, color=COMPARISON_COLORS[0],
                 bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85})
    for axis in axes:
        axis.set(xlim=(-0.2, last_k + 0.2), xticks=range(last_k + 1), xlabel="iteration $k$",
                 ylabel="$f(x^{(k)})$")
        axis.grid(True, linewidth=0.45, alpha=0.45)
    axes[0].legend(loc="upper left", fontsize=8.5, frameon=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def common_landscape(axis, xlim: tuple[float, float], ylim: tuple[float, float]):
    """Draw the common objective using a consistent, light grayscale scale."""
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    xx1, xx2 = np.meshgrid(np.linspace(*xlim, 400), np.linspace(*ylim, 400))
    values = (xx1**2 + xx2 - 1.0)**2 + (xx1 + xx2**2 - 1.0)**2
    height = np.log10(1.0 + values)
    cmap = LinearSegmentedColormap.from_list("light_greys", ["#ffffff", "#d6d6d6"])
    landscape = axis.pcolormesh(
        xx1, xx2, height, shading="auto", cmap=cmap,
        norm=Normalize(vmin=0.0, vmax=2.3), rasterized=True,
    )
    axis.contour(
        xx1, xx2, height,
        levels=[0.0001, 0.001, 0.004, 0.015, 0.05, 0.15, 0.40, 0.80, 1.30, 1.90],
        colors="#9a9a9a", linewidths=0.55, alpha=0.75,
    )
    axis.set(xlim=xlim, ylim=ylim, xlabel=r"$x_1$", ylabel=r"$x_2$")
    axis.set_aspect("equal", adjustable="box")
    return landscape


def save_start_comparison_plot(output: Path) -> None:
    """Compare actual trajectories and gradient histories for the two slide starts."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    starts = (COMMON_START, (0.0, 0.15))
    runs = [run_comparison(start) for start in starts]
    points = np.vstack([point for results in runs for result in results
                        for point in result.points])
    lower, upper = points.min(axis=0), points.max(axis=0)
    padding = 0.10 * (upper - lower)
    xlim, ylim = tuple(zip(lower - padding, upper + padding))
    label_box = {"facecolor": "white", "edgecolor": "none", "pad": 0.8}

    with plt.rc_context({"font.size": 10.5, "axes.titlesize": 12,
                         "axes.labelsize": 11.5, "legend.fontsize": 11}):
        fig, axes = plt.subplots(2, 2, figsize=(10.4, 6.2))
        fig.subplots_adjust(left=0.065, right=0.98, bottom=0.075, top=0.895,
                            wspace=0.24, hspace=0.45)
        for row, (start, results) in enumerate(zip(starts, runs)):
            trajectory, convergence = axes[row]
            common_landscape(trajectory, xlim, ylim)
            for result, color, style, marker in zip(
                results, COMPARISON_COLORS, COMPARISON_LINESTYLES, COMPARISON_MARKERS
            ):
                xy = np.vstack(result.points)
                trajectory.plot(xy[:, 0], xy[:, 1], color=color, linestyle=style,
                                marker=marker, markersize=3.9, linewidth=1.8,
                                markeredgecolor="white", markeredgewidth=0.45,
                                label=result.name, zorder=4)
                convergence.semilogy(range(result.updates + 1), result.grad_norms,
                                     color=color, linestyle=style, marker=marker,
                                     markersize=4.2, linewidth=1.8)
                offset = (5, -5 if result.grad_norms[-1] > 1e-9 else 2)
                if row == 1 and result.name == "Shifted Newton":
                    offset = (-33, 6)
                elif row == 1 and result.name == "Newton--Armijo":
                    offset = (7, 6)
                convergence.annotate(rf"$k={result.updates}$",
                                     (result.updates, result.grad_norms[-1]),
                                     xytext=offset,
                                     va="top" if result.grad_norms[-1] > 1e-9 else "baseline",
                                     textcoords="offset points",
                                     color=color, fontsize=10, bbox=label_box)
            trajectory.scatter(*start, marker="D", s=52, facecolor="white",
                               edgecolor="black", linewidth=1, zorder=7)
            trajectory.annotate(r"$x^{(0)}$", start, xytext=(6, 7) if row == 0 else (7, -16),
                                textcoords="offset points", fontsize=10,
                                bbox=label_box, zorder=8)
            ends = [result.points[-1] for result in results]
            common_end = all(np.allclose(end, ends[0], atol=1e-6) for end in ends)
            for index, end in enumerate(ends[:1] if common_end else ends):
                color = "black" if common_end else COMPARISON_COLORS[index]
                trajectory.scatter(*end, marker="*", s=108, facecolor="white",
                                   edgecolor=color, linewidth=1.25, zorder=8)
                coords = [0.0 if abs(value) < 1e-6 else round(float(value), 3)
                          for value in end]
                offset = (-8, -17) if common_end else [(4, -17), (-18, 8), (9, -10)][index]
                trajectory.annotate(f"({coords[0]:g}, {coords[1]:g})", end,
                                    xytext=offset, textcoords="offset points",
                                    fontsize=9.5, color=color,
                                    ha="right" if index == 1 else "left",
                                    bbox=label_box, zorder=9)
            trajectory.set_title(rf"$x^{{(0)}}=({start[0]:g},{start[1]:g})^T$: trajectories")
            trajectory.set_xticks([-2, 0, 2, 4])
            trajectory.set_yticks([-2, -1, 0, 1])
            convergence.axhline(1e-8, color="#777777", linewidth=1, linestyle="--")
            convergence.text(0.25, 1.8e-8, r"stop: $10^{-8}$", color="#666666",
                             fontsize=9.5, ha="left")
            convergence.set(title="Gradient-norm convergence", xlim=(-0.3, 18.3),
                            ylim=(1e-16, 1e3), xticks=[0, 4, 8, 12, 16],
                            yticks=[1e-14, 1e-8, 1e-2, 1e2],
                            xlabel=r"iteration $k$", ylabel=r"$\|\nabla f(x^{(k)})\|_2$")
            convergence.grid(True, linewidth=0.45, alpha=0.4)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.53, 1.015),
                   ncol=3, frameon=False, handlelength=3)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, bbox_inches="tight")
        plt.close(fig)


def trajectory_bounds(results: list[Result]) -> tuple[tuple[float, float], tuple[float, float]]:
    """Include every iterate and the common limiting minimizer."""
    points = np.vstack([point for result in results for point in result.points]
                       + [np.asarray(COMMON_MINIMIZER)])
    lower, upper = points.min(axis=0), points.max(axis=0)
    padding = 0.09 * np.maximum(upper - lower, 0.5)
    return tuple(zip(lower - padding, upper + padding))


def largest_ascent_step(result: Result) -> int:
    """Locate the largest objective increase along an indefinite Newton ascent step."""
    candidates = []
    for k, (x, next_x) in enumerate(zip(result.points[:-1], result.points[1:])):
        if (result.values[k+1] > result.values[k]
                and np.linalg.eigvalsh(hessian(x))[0] < 0.0
                and float(gradient(x) @ (next_x - x)) > 0.0):
            candidates.append(k)
    if not candidates:
        raise ValueError("The pure-Newton example has no indefinite ascent step to explain.")
    return max(candidates, key=lambda k: result.values[k+1] - result.values[k])


def save_full_step_plot(results: list[Result], output: Path) -> None:
    """Show that a positive-definite model and descent do not validate a full step."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), constrained_layout=True)
    alphas = np.linspace(0.0, 1.0, 400)
    for axis, result, color in zip(axes, results[:2], COMPARISON_COLORS):
        candidates = []
        for k, x in enumerate(result.points[:-1]):
            d = result.points[k+1] - x
            b = hessian(x) + result.shifts[k] * np.eye(x.size)
            if (result.values[k+1] > result.values[k]
                    and np.linalg.eigvalsh(b)[0] > 0.0
                    and float(gradient(x) @ d) < 0.0):
                candidates.append(k)
        if not candidates:
            raise ValueError(f"{result.name} has no full-step increase along a PD-model descent direction.")
        k = max(candidates, key=lambda index: result.values[index+1] - result.values[index])
        x, d = result.points[k], result.points[k+1] - result.points[k]
        b = hessian(x) + result.shifts[k] * np.eye(x.size)
        slope = float(gradient(x) @ d)
        actual = np.array([objective(x + alpha*d) for alpha in alphas])
        model = actual[0] + alphas*slope + 0.5*alphas**2*float(d @ b @ d)
        axis.plot(alphas, actual, color=color, linewidth=2.1, label=r"actual $f(x+\alpha d)$")
        axis.plot(alphas, model, color="#666666", linewidth=1.7, linestyle="--",
                  label="quadratic model")
        axis.axhline(actual[0], color="#999999", linewidth=0.9, linestyle=":",
                      label=rf"starting value ${actual[0]:.5f}$")
        axis.scatter([0, 1], [actual[0], actual[-1]], s=27, color=color, zorder=5)
        axis.annotate(rf"full step: ${actual[-1]:.5f}$", (1, actual[-1]),
                       xytext=(-8, 9), textcoords="offset points", ha="right",
                       fontsize=9, color=color)
        matrix = "H" if result.shifts[k] == 0.0 else "B"
        axis.text(0.04, 0.59,
                   rf"$\lambda_{{\min}}({matrix})={np.linalg.eigvalsh(b)[0]:.5f}>0$"
                   "\n" + rf"$g^Td={slope:.5f}<0$",
                   transform=axis.transAxes, fontsize=8.3, linespacing=1.55,
                   bbox={"facecolor": "white", "edgecolor": "none", "pad": 2.0})
        axis.set_yscale("symlog", linthresh=0.005)
        axis.set(xlim=(0, 1.03), ylim=(min(0.0, float(model.min())) - 0.0008,
                                      float(actual.max()) * 2.2),
                  xlabel=r"step length $\alpha$", ylabel="function / model value (symlog scale)",
                  title=rf"{result.name}: $k={k}$")
        axis.grid(True, linewidth=0.45, alpha=0.4)
        axis.legend(loc="upper left", frameon=False, fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def save_problem_plot(output: Path) -> None:
    """Show the common starting point and the selected global minimizer."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(5.3, 4.4), constrained_layout=True)
    landscape = common_landscape(axis, (0.55, 1.4), (-0.45, 0.70))
    start, minimum = np.asarray(COMMON_START), np.asarray(COMMON_MINIMIZER)
    label_box = {"facecolor": "white", "edgecolor": "none", "pad": 1.4}
    axis.scatter(*start, s=54, marker="D", facecolor="white", edgecolor="black", zorder=5)
    axis.scatter(*minimum, s=95, marker="*", facecolor="white", edgecolor="black", zorder=5)
    axis.annotate(rf"$x^{{(0)}}=({start[0]:g},{start[1]:g})^T$", start,
                  xytext=(-8, 12), textcoords="offset points", ha="right",
                  fontsize=9, bbox=label_box)
    axis.annotate(rf"$x^*=({minimum[0]:g},{minimum[1]:g})^T$", minimum,
                  xytext=(9, -15), textcoords="offset points", fontsize=9, bbox=label_box)
    axis.set_title("Common starting point and target minimizer")
    colorbar = fig.colorbar(landscape, ax=axis, fraction=0.055, pad=0.04)
    colorbar.set_label(r"background shading: $\log_{10}(1+f)$", fontsize=9)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def save_trajectory_plot(
    results: list[Result], output: Path, explain_overshoot: bool = False,
) -> None:
    """Plot a prefix of the common comparison and optionally explain its ascent step."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as path_effects

    zorders = [4, 5, 4]
    full_xlim, full_ylim = trajectory_bounds(results)
    early_points = np.vstack([result.points[:4] for result in results]
                             + [[COMMON_MINIMIZER]])
    early_lower, early_upper = early_points.min(axis=0), early_points.max(axis=0)
    early_center = 0.5 * (early_lower + early_upper)
    early_radius = 0.60 * max(float((early_upper - early_lower).max()), 0.4)
    zoom_xlim, zoom_ylim = tuple(zip(early_center - early_radius, early_center + early_radius))
    views = [
        (full_xlim, full_ylim, "Full trajectory"),
        (zoom_xlim, zoom_ylim, "Early steps and minimizer (zoom)"),
    ]
    if explain_overshoot:
        views = views[:1]
    fig, axes = plt.subplots(
        1, 2, figsize=(9.2, 4.4),
        gridspec_kw={"width_ratios": [1.20, 1.0]},
        constrained_layout=True,
    )

    landscape = None
    for axis, (xlim, ylim, title) in zip(axes, views):
        landscape = common_landscape(axis, xlim, ylim)
        for result, color, marker, linestyle, zorder in zip(
            results, COMPARISON_COLORS, COMPARISON_MARKERS, COMPARISON_LINESTYLES, zorders
        ):
            points = np.vstack(result.points)
            axis.plot(
                points[:, 0], points[:, 1],
                color=color, marker=marker, markersize=4.4,
                markeredgecolor="white", markeredgewidth=0.65,
                linewidth=2.1, linestyle=linestyle,
                # A narrow white outline separates paths from grey contour lines.
                path_effects=[path_effects.Stroke(linewidth=3.1, foreground="white"),
                              path_effects.Normal()],
                label=result.name, zorder=zorder,
            )
        x0 = results[0].points[0]
        if xlim[0] <= x0[0] <= xlim[1] and ylim[0] <= x0[1] <= ylim[1]:
            axis.scatter(
                [x0[0]], [x0[1]], s=42, marker="D",
                facecolor="white", edgecolor="black", linewidth=0.8,
                zorder=5,
            )
            axis.annotate(
                r"$x^{(0)}$", x0, xytext=(4, 5),
                textcoords="offset points", fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0},
            )
        minimum = np.asarray(COMMON_MINIMIZER)
        if xlim[0] <= minimum[0] <= xlim[1] and ylim[0] <= minimum[1] <= ylim[1]:
            axis.scatter(
                [minimum[0]], [minimum[1]], s=75, marker="*",
                facecolor="white", edgecolor="black", linewidth=0.8, zorder=5,
            )
            axis.annotate(
                r"$x^*$", minimum, xytext=(-23, -9),
                textcoords="offset points", fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0},
            )
        axis.set_title(title)

    pure_points = np.vstack(results[0].points)
    k = largest_ascent_step(results[0])
    axes[0].annotate(
        rf"pure Newton: $k={k+1}$", pure_points[k+1], xytext=(-12, 12),
        textcoords="offset points", ha="right", fontsize=8, color=COMPARISON_COLORS[0], zorder=8,
        arrowprops={"arrowstyle": "->", "color": COMPARISON_COLORS[0], "linewidth": 0.9},
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.2},
    )
    if explain_overshoot:
        x = pure_points[k]
        d = pure_points[k+1] - x
        g, h = gradient(x), hessian(x)
        slope = float(g @ d)
        alphas = np.linspace(0, 1, 300)
        actual = np.array([objective(x + alpha*d) for alpha in alphas])
        model = objective(x) + alphas*slope + 0.5*alphas**2*(d @ h @ d)
        color = COMPARISON_COLORS[0]
        label_box = {"facecolor": "white", "edgecolor": "none", "pad": 2.0}
        axes[0].set_title(rf"Pure Newton: excursion $k={k}$ to ${k+1}$")
        axes[0].annotate(rf"$x^{{({k})}}$", x, xytext=(-16, -17),
                         textcoords="offset points", fontsize=9, color=color,
                         bbox=label_box, zorder=8)
        axes[1].plot(alphas, actual, color=color, linewidth=2.1,
                     label=rf"actual $f(x^{{({k})}}+\alpha d^{{({k})}})$")
        axes[1].plot(alphas, model, color="#555555", linestyle="--", linewidth=1.8,
                     label=rf"model $m^{{({k})}}(\alpha d^{{({k})}})$")
        axes[1].scatter([0, 1], [actual[0], actual[-1]], color=color, s=26, zorder=5)
        axes[1].annotate(f"actual: {actual[-1]:.3f}", (1, actual[-1]), xytext=(-8, 8),
                         textcoords="offset points", ha="right", fontsize=9, color=color)
        axes[1].annotate(f"model: {model[-1]:.3f}", (1, model[-1]), xytext=(-8, -22),
                         textcoords="offset points", ha="right", fontsize=9,
                         arrowprops={"arrowstyle": "->", "color": "#555555"})
        axes[1].set_yscale("symlog", linthresh=0.01)
        axes[1].set(xlim=(0, 1.03), ylim=(0, 3.2 * max(actual)), xlabel=r"step length $\alpha$",
                     ylabel="function / model value (symlog scale)",
                     title="Actual function versus quadratic model")
        axes[1].grid(True, linewidth=0.45, alpha=0.4)
        axes[1].legend(loc="upper left", frameon=False, fontsize=8)
        axes[0].set_xlabel(r"$x_1$" + "\nGrey shading: $\log_{10}(1+f)$")
    else:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncol=len(results),
                    frameon=False, fontsize=8, handlelength=3.0)
        colorbar = fig.colorbar(
            landscape, ax=axes, orientation="horizontal",
            fraction=0.055, pad=0.08, aspect=48,
        )
        colorbar.set_label(r"background shading: $\log_{10}(1+f)$", fontsize=9)
        colorbar.ax.tick_params(labelsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def run_saddle_example(tol: float = 1e-8) -> tuple[Array, Array]:
    """Reuse the exact pure-Newton iterates from the double-well comparison."""
    pure_points, _, _, _ = run_shifting_benefit(tol=tol)
    return np.vstack(pure_points), np.array([
        np.linalg.norm(double_well_gradient(x)) for x in pure_points
    ])


def save_saddle_plot(output: Path) -> None:
    """Show a trajectory toward a saddle and its decaying gradient norm."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    points, grad_norms = run_saddle_example()
    color = COMPARISON_COLORS[0]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), constrained_layout=True)
    x1 = np.linspace(-0.15, 0.48, 360)
    x2 = np.linspace(-0.38, 0.22, 400)
    xx1, xx2 = np.meshgrid(x1, x2)
    values = (xx1 + xx2)**4 / 16.0 - xx1*xx2
    terrain_cmap = LinearSegmentedColormap.from_list("saddle_greys", ["#ffffff", "#dedede"])
    axes[0].pcolormesh(xx1, xx2, values, shading="auto", cmap=terrain_cmap,
                       rasterized=True)
    contours = axes[0].contour(
        xx1, xx2, values, levels=[-0.08, -0.04, -0.01, 0.01, 0.04, 0.08, 0.12, 0.16],
        colors="#999999", linewidths=0.65, linestyles="solid",
    )
    axes[0].clabel(contours, levels=[-0.04, 0.04, 0.08, 0.12], fontsize=7, fmt="%g")
    axes[0].contour(xx1, xx2, values, levels=[0], colors="#777777",
                    linewidths=0.9, linestyles="dashed")
    axes[0].plot(points[:, 0], points[:, 1], color=color, marker="o", markersize=5,
                 markeredgecolor="white", linewidth=2.0, zorder=4)
    for start, end in zip(points[:-1], points[1:]):
        axes[0].annotate("", start + 0.65*(end-start),
                         xytext=start + 0.40*(end-start),
                         arrowprops={"arrowstyle": "->", "color": color, "lw": 1.8},
                         zorder=5)
    label_box = {"facecolor": "white", "edgecolor": "none", "pad": 1.0}
    axes[0].annotate(r"$x^{(0)}$", points[0], xytext=(-20, -15),
                     textcoords="offset points", fontsize=9, color=color,
                     bbox=label_box, zorder=6)
    axes[0].annotate(rf"$x^{{(1)}},\ldots,x^{{({len(points)-1})}}$ near the saddle", points[1],
                     xytext=(12, 28), textcoords="offset points", fontsize=8.5, color=color,
                     bbox=label_box, arrowprops={"arrowstyle": "->", "color": color}, zorder=6)
    axes[0].scatter([0], [0], marker="*", s=90, facecolor="white", edgecolor="black",
                     linewidth=0.8, zorder=5)
    axes[0].annotate(r"saddle $\bar{x}=(0,0)^T$", (0, 0), xytext=(-24, -60),
                     textcoords="offset points", fontsize=9, bbox=label_box,
                     arrowprops={"arrowstyle": "->", "color": "#555555"}, zorder=6)
    axes[0].set(xlim=(x1[0], x1[-1]), ylim=(x2[0], x2[-1]), xlabel="$x_1$", ylabel="$x_2$",
                 title="Function contours and Newton iterates")
    axes[0].set_aspect("equal", adjustable="box")

    iterations = np.arange(len(grad_norms))
    axes[1].semilogy(iterations, grad_norms, "o-", color=color, linewidth=2,
                     markersize=5, label="Pure Newton")
    axes[1].axhline(1e-8, color="#777777", linestyle="--", linewidth=1.0,
                    label=r"tolerance $10^{-8}$")
    axes[1].set(xlabel="iteration $k$", ylabel=r"$\|\nabla F(x^{(k)})\|_2$",
                 title="Gradient norm", xticks=iterations,
                 ylim=(max(grad_norms[-1] * 0.1, 1e-30), grad_norms[0] * 5))
    axes[1].grid(True, which="major", linewidth=0.45, alpha=0.45)
    axes[1].legend(loc="upper right", frameon=False, fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(f"\nDouble well: {len(points)-1} Newton updates to the saddle (0,0); "
          f"final gradient norm = {grad_norms[-1]:.3e}")


def save_shifting_benefit_plot(output: Path) -> None:
    """Show how diagonal shifting steers Newton away from a saddle point."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    pure_points, shifted_points, _, _ = run_shifting_benefit()
    pure_array = np.vstack(pure_points)
    shifted_array = np.vstack(shifted_points)
    x1 = np.linspace(-1.2, 1.2, 420)
    x2 = np.linspace(-1.2, 1.2, 420)
    xx1, xx2 = np.meshgrid(x1, x2)
    values = (xx1 + xx2) ** 4 / 16.0 - xx1 * xx2
    height = np.log10(1.0 + values + 0.25)

    fig, axis = plt.subplots(figsize=(5.35, 4.75), constrained_layout=True)
    terrain_cmap = LinearSegmentedColormap.from_list("double_well_greys", ["#ffffff", "#d6d6d6"])
    landscape = axis.contourf(
        xx1, xx2, height, levels=28, cmap=terrain_cmap
    )
    axis.contour(
        xx1, xx2, height,
        levels=[0.02, 0.05, 0.10, 0.18, 0.28, 0.42, 0.60],
        colors="#999999", linewidths=0.55, alpha=0.75,
    )
    boundary = 1.0 - 1.5 * (xx1 + xx2) ** 2
    axis.contour(
        xx1, xx2, boundary, levels=[0.0], colors=["#666666"],
        linestyles=[":"], linewidths=[1.7],
    )
    axis.plot(
        pure_array[:, 0], pure_array[:, 1],
        color=COMPARISON_COLORS[0], linewidth=2.0, linestyle=COMPARISON_LINESTYLES[0],
        marker="o", markersize=4.2, markeredgecolor="white",
        markeredgewidth=0.55, label="pure Newton",
    )
    axis.plot(
        shifted_array[:, 0], shifted_array[:, 1],
        color=COMPARISON_COLORS[1], linewidth=2.1, linestyle=COMPARISON_LINESTYLES[1],
        marker="s", markersize=4.4,
        markeredgecolor="white", markeredgewidth=0.55,
        label="shifted Newton",
    )
    axis.scatter(
        [0.4], [-0.3], s=52, marker="D", facecolor="white",
        edgecolor="black", linewidth=0.65, zorder=5,
    )
    axis.scatter(
        [0.0], [0.0], s=62, marker="X", facecolor=COMPARISON_COLORS[0],
        edgecolor="black", linewidth=0.65, zorder=5,
    )
    minimizer = 1.0 / np.sqrt(2.0)
    axis.scatter(
        [minimizer, -minimizer], [minimizer, -minimizer],
        s=92, marker="*", facecolor="white", edgecolor="black",
        linewidth=0.65, zorder=5,
    )
    label_box = {"facecolor": "white", "edgecolor": "none", "pad": 1.0}
    axis.annotate(r"$x^{(0)}$", (0.4, -0.3), xytext=(7, -14),
                  textcoords="offset points", fontsize=10, bbox=label_box)
    axis.annotate("saddle", (0.0, 0.0), xytext=(-45, 7),
                  textcoords="offset points", fontsize=9, color=COMPARISON_COLORS[0],
                  bbox=label_box)
    axis.annotate(r"$x^*_+$", (minimizer, minimizer), xytext=(7, -14),
                  textcoords="offset points", fontsize=10, bbox=label_box)
    axis.text(
        -1.13, 0.63, "indefinite Hessian", color="#555555", fontsize=8.5,
        rotation=-43, bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8},
    )
    axis.set(
        xlim=(-1.2, 1.2), ylim=(-1.2, 1.2),
        xlabel=r"$x_1$", ylabel=r"$x_2$",
    )
    axis.set_aspect("equal", adjustable="box")
    axis.legend(loc="lower left", frameon=True, framealpha=0.92, fontsize=8.5)
    colorbar = fig.colorbar(landscape, ax=axis, fraction=0.055, pad=0.04)
    colorbar.set_label(r"$\log_{10}(1+f-f^*)$", fontsize=9)
    colorbar.ax.tick_params(labelsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", type=Path, help="optional output path for the comparison plot")
    parser.add_argument("--details", action="store_true", help="print complete iteration tables")
    parser.add_argument(
        "--divergence-demo", action="store_true",
        help="also reproduce the scalar full-step-divergence / Armijo-convergence example",
    )
    parser.add_argument(
        "--compare-start", type=float, nargs=2, metavar=("X1", "X2"),
        help="also compare the unchanged algorithms from another initial point",
    )
    parser.add_argument("--oscillation-plot", type=Path, help="output path for early objective rebounds")
    parser.add_argument(
        "--trajectory-plot", type=Path,
        help="optional output path for the common-problem trajectory plot",
    )
    parser.add_argument(
        "--pure-trajectory-plot", type=Path,
        help="optional output path for the pure Newton trajectory alone",
    )
    parser.add_argument("--saddle-plot", type=Path,
                        help="output path for the double-well saddle example")
    parser.add_argument("--shifted-trajectory-plot", type=Path,
                        help="output path for the pure and shifted Newton trajectories")
    parser.add_argument("--problem-plot", type=Path,
                        help="output path for the introductory common landscape")
    parser.add_argument("--full-step-plot", type=Path,
                        help="output path for two descent directions whose full steps increase f")
    parser.add_argument("--start-comparison-plot", type=Path,
                        help="output path for the two-start trajectory/convergence comparison")
    parser.add_argument(
        "--shift-benefit-plot", type=Path,
        help="optional output path for the shifting-benefit comparison plot",
    )
    args = parser.parse_args()
    results = run_comparison()
    print_results(results)
    print_shifting_benefit()
    if args.divergence_demo:
        print_divergence_example()
    if args.details:
        print_iteration_tables(results)
    if args.compare_start is not None:
        print("\nAdditional starting-point comparison (same algorithms and parameters)")
        additional_results = run_comparison(tuple(args.compare_start))
        print_results(additional_results)
        if args.details:
            print_iteration_tables(additional_results)
    if args.plot is not None:
        save_plot(results, args.plot)
    if args.oscillation_plot is not None:
        save_oscillation_plot(results, args.oscillation_plot)
    if args.trajectory_plot is not None:
        save_trajectory_plot(results, args.trajectory_plot)
    if args.pure_trajectory_plot is not None:
        save_trajectory_plot(results[:1], args.pure_trajectory_plot, explain_overshoot=True)
    if args.shifted_trajectory_plot is not None:
        save_trajectory_plot(results[:2], args.shifted_trajectory_plot)
    if args.problem_plot is not None:
        save_problem_plot(args.problem_plot)
    if args.full_step_plot is not None:
        save_full_step_plot(results, args.full_step_plot)
    if args.start_comparison_plot is not None:
        save_start_comparison_plot(args.start_comparison_plot)
    if args.saddle_plot is not None:
        save_saddle_plot(args.saddle_plot)
    if args.shift_benefit_plot is not None:
        save_shifting_benefit_plot(args.shift_benefit_plot)


if __name__ == "__main__":
    main()
