import argparse
from functools import partial
from pathlib import Path
from typing import Callable, Optional

import numpy as np


LineSearch = Callable[
    [Callable, np.ndarray, np.ndarray, float, np.ndarray],
    float,
]


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

Q = np.array([[2.0, -2.0], [-2.0, 4.0]])
c = np.array([0.0, 2.0])

ROTATION_ANGLE = np.pi / 4
KAPPAS = (5.0, 50.0)
RELATIVE_GAP_TOL = 1e-6
FLAT_L = 2.0 + 4.0 * np.exp(-1.5)
PRECONDITION_KAPPA = KAPPAS[1]
PRECONDITION_ANGLE = np.deg2rad(10.0)


# -----------------------------------------------------------------------------
# Algorithms
# -----------------------------------------------------------------------------

def _as_symmetric_pd(matrix, name):
    """Validate and return a floating-point symmetric PD matrix."""
    matrix = np.asarray(matrix, dtype=float)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or not np.all(np.isfinite(matrix))
        or not np.allclose(matrix, matrix.T)
    ):
        raise ValueError(f"{name} must be a finite symmetric square matrix")
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as error:
        raise ValueError(f"{name} must be positive definite") from error
    return matrix


def armijo_backtracking(
    objective,
    x,
    direction,
    value,
    gradient,
    initial_step=1.0,
    contraction=0.5,
    sigma=1e-4,
    max_backtracks=50,
):
    """Return an Armijo backtracking step along a descent direction."""
    if not np.isfinite(initial_step) or initial_step <= 0.0:
        raise ValueError("initial_step must be finite and positive")
    if not np.isfinite(contraction) or not 0.0 < contraction < 1.0:
        raise ValueError("contraction must lie in (0,1)")
    if not np.isfinite(sigma) or not 0.0 < sigma < 1.0:
        raise ValueError("sigma must lie in (0,1)")
    if not isinstance(max_backtracks, (int, np.integer)) or max_backtracks < 0:
        raise ValueError("max_backtracks must be a nonnegative integer")

    x = np.asarray(x, dtype=float)
    direction = np.asarray(direction, dtype=float)
    gradient = np.asarray(gradient, dtype=float)
    slope = float(gradient @ direction)
    if not np.isfinite(slope) or slope >= 0.0:
        raise ValueError("Armijo backtracking requires a descent direction")

    alpha = float(initial_step)
    for _ in range(max_backtracks + 1):
        trial = x + alpha * direction
        if np.array_equal(trial, x):
            raise RuntimeError("Armijo backtracking stagnated numerically")
        with np.errstate(over="ignore", invalid="ignore"):
            trial_value = float(objective(trial))
        if np.isfinite(trial_value) and trial_value <= value + sigma * alpha * slope:
            return alpha
        alpha *= contraction
    raise RuntimeError("Armijo backtracking exhausted max_backtracks")


def make_quadratic_exact_line_search(hessian):
    """Return the closed-form exact line search for a symmetric PD quadratic."""
    matrix = _as_symmetric_pd(hessian, "hessian")

    def exact_line_search(_objective, _x, direction, _value, gradient):
        denominator = float(direction @ matrix @ direction)
        if not np.isfinite(denominator) or denominator <= 0.0:
            raise RuntimeError(
                "quadratic line search encountered nonpositive curvature"
            )
        return -float(gradient @ direction) / denominator

    return exact_line_search


def steepest_descent(
    objective,
    grad,
    x0,
    line_search: Optional[LineSearch] = None,
    tol=1e-6,
    max_iter=100,
):
    """Line-search steepest descent for a differentiable objective.

    A line-search callback receives ``(objective, x, direction, value,
    gradient)`` and returns a positive step. By default, Armijo backtracking
    is used. The history rows are ``(k, x, ||gradient||, step)``.
    """
    x = np.asarray(x0, dtype=float)
    if x.ndim != 1 or x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError("x0 must be a nonempty finite one-dimensional array")
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("tol must be finite and positive")
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 0:
        raise ValueError("max_iter must be a nonnegative integer")
    if line_search is not None and not callable(line_search):
        raise ValueError("line_search must be callable or None")

    x = x.copy()
    history = []
    for k in range(max_iter + 1):
        value = float(objective(x))
        g = np.asarray(grad(x), dtype=float)
        if (
            not np.isfinite(value)
            or g.shape != x.shape
            or not np.all(np.isfinite(g))
        ):
            raise RuntimeError("objective or gradient returned invalid values")
        norm_g = float(np.linalg.norm(g))
        if norm_g <= tol:
            history.append((k, x.copy(), norm_g, None))
            return x, history
        if k == max_iter:
            break
        direction = -g
        if line_search is None:
            alpha = armijo_backtracking(objective, x, direction, value, g)
        else:
            alpha = float(line_search(objective, x, direction, value, g))
        if not np.isfinite(alpha) or alpha <= 0.0:
            raise RuntimeError("line search must return a finite positive step")
        history.append((k, x.copy(), norm_g, alpha))
        x = x + alpha * direction
    raise RuntimeError("maximum number of iterations reached")


def quadratic_steepest_descent(
    hessian,
    linear_term,
    x0,
    tol=1e-6,
    max_iter=100,
):
    """Apply the general SD driver with the quadratic exact-step formula."""
    matrix = np.asarray(hessian, dtype=float)
    linear_term = np.asarray(linear_term, dtype=float)
    exact_search = make_quadratic_exact_line_search(matrix)
    if linear_term.shape != (matrix.shape[0],) or not np.all(
        np.isfinite(linear_term)
    ):
        raise ValueError("linear_term has incompatible or non-finite entries")

    def quadratic_objective(x):
        return float(0.5 * x @ matrix @ x - linear_term @ x)

    def quadratic_gradient(x):
        return matrix @ x - linear_term

    return steepest_descent(
        quadratic_objective,
        quadratic_gradient,
        x0,
        line_search=exact_search,
        tol=tol,
        max_iter=max_iter,
    )


def symmetric_pd_inverse_half(matrix):
    """Return the inverse square root of a symmetric PD matrix."""
    matrix = _as_symmetric_pd(matrix, "matrix")
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    inverse_half = (eigenvectors * (1.0 / np.sqrt(eigenvalues))) @ eigenvectors.T
    return inverse_half


def preconditioned_quadratic_trajectory(
    matrix,
    start,
    preconditioner=None,
    rel_tol=RELATIVE_GAP_TOL,
    max_updates=10000,
):
    """Return an exact-line-search preconditioned SD trajectory."""
    matrix = _as_symmetric_pd(matrix, "matrix")
    start = np.asarray(start, dtype=float)
    if start.shape != (matrix.shape[0],) or not np.all(np.isfinite(start)):
        raise ValueError("start has incompatible or non-finite entries")
    if not np.isfinite(rel_tol) or not 0.0 < rel_tol < 1.0:
        raise ValueError("rel_tol must lie in (0,1)")
    if not isinstance(max_updates, (int, np.integer)) or max_updates <= 0:
        raise ValueError("max_updates must be a positive integer")

    if preconditioner is None:
        preconditioner = np.eye(matrix.shape[0])
    else:
        preconditioner = _as_symmetric_pd(preconditioner, "preconditioner")
    if preconditioner.shape != matrix.shape:
        raise ValueError("preconditioner has incompatible dimensions")

    x = start.copy()
    initial_gap = float(0.5 * x @ matrix @ x)
    if initial_gap == 0.0:
        return np.asarray([x]), np.asarray([0.0]), np.asarray([])

    points = [x.copy()]
    gaps = [initial_gap]
    steps = []
    while gaps[-1] / initial_gap > rel_tol:
        gradient = matrix @ x
        direction = -np.linalg.solve(preconditioner, gradient)
        denominator = float(direction @ matrix @ direction)
        alpha = -float(gradient @ direction) / denominator
        x = x + alpha * direction
        points.append(x.copy())
        gaps.append(float(0.5 * x @ matrix @ x))
        steps.append(alpha)
        if len(steps) >= max_updates and gaps[-1] / initial_gap > rel_tol:
            raise RuntimeError("preconditioned SD did not reach its tolerance")
    return np.asarray(points), np.asarray(gaps), np.asarray(steps)


# -----------------------------------------------------------------------------
# Test problems
# -----------------------------------------------------------------------------

def objective(x):
    """Objective in the hand-worked quadratic example."""
    return 0.5 * x @ Q @ x - c @ x


def gradient(x):
    """Gradient in the hand-worked quadratic example."""
    return Q @ x - c


def quartic_objective(x):
    """A smooth nonquadratic example with minimizer (0,1)."""
    x = np.asarray(x, dtype=float)
    return float(
        0.25 * x[0] ** 4
        + 0.5 * x[0] ** 2
        + 0.5 * (x[1] - 1.0) ** 2
    )


def quartic_gradient(x):
    x = np.asarray(x, dtype=float)
    return np.array([x[0] ** 3 + x[0], x[1] - 1.0])


def quadratic_family(kappa, angle=ROTATION_ANGLE):
    """Return Q_kappa and a worst-case exact-SD starting point.

    Q_kappa = R diag(1,kappa) R^T and R^T x^(0) = (4,4/kappa).
    The minimizer is the origin.
    """
    cosine, sine = np.cos(angle), np.sin(angle)
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    matrix = rotation @ np.diag([1.0, float(kappa)]) @ rotation.T
    start = rotation @ np.array([4.0, 4.0 / float(kappa)])
    return matrix, start


def preconditioning_example():
    """Return the common quadratic, start, and three preconditioners."""
    matrix, start = quadratic_family(PRECONDITION_KAPPA, PRECONDITION_ANGLE)
    preconditioners = {
        "pure": np.eye(2),
        "jacobi": np.diag(np.diag(matrix)),
        "ideal": matrix.copy(),
    }
    return matrix, start, preconditioners


def flat_phi(x):
    """varphi(x) = exp(-x^2) + x^2 - 1, with minimum varphi(0)=0."""
    return np.exp(-(x ** 2)) + x ** 2 - 1.0


def flat_phi_gradient(x):
    return 2.0 * x * (1.0 - np.exp(-(x ** 2)))


# -----------------------------------------------------------------------------
# Numerical experiments
# -----------------------------------------------------------------------------

def quadratic_relative_gap_trajectory(kappa, rel_tol=RELATIVE_GAP_TOL):
    """Return exact-SD iterates until the relative gap reaches ``rel_tol``."""
    matrix, start = quadratic_family(kappa)
    points, gaps, steps = preconditioned_quadratic_trajectory(
        matrix,
        start,
        rel_tol=rel_tol,
    )
    return matrix, points, gaps, steps


def flat_sublinear_history(x0=1.0, step=None, updates=10000):
    """Fixed-step SD data for the smooth convex function flat_phi."""
    if step is None:
        step = 1.0 / FLAT_L
    x = float(x0)
    xs = np.empty(updates + 1)
    gaps = np.empty(updates + 1)
    for k in range(updates + 1):
        xs[k] = x
        gaps[k] = flat_phi(x)
        if k < updates:
            x -= step * flat_phi_gradient(x)
    ratios = gaps[1:] / gaps[:-1]
    return xs, gaps, ratios


# -----------------------------------------------------------------------------
# Experimental figures
# -----------------------------------------------------------------------------

def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    return plt


def _save_and_close(plt, fig, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return output


def _quadratic_contours(ax, matrix):
    bound = 4.35
    grid = np.linspace(-0.8, bound, 360)
    x1, x2 = np.meshgrid(grid, grid)
    points = np.stack((x1, x2), axis=-1)
    values = 0.5 * np.einsum("...i,ij,...j->...", points, matrix, points)
    levels = np.array([0.15, 0.35, 0.75, 1.5, 3.0, 5.5, 8.5])
    ax.contour(x1, x2, values, levels=levels, colors="0.67", linewidths=0.8)
    ax.set_xlim(-0.8, bound)
    ax.set_ylim(-0.8, bound)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.tick_params(direction="out", length=3)


def save_quadratic_trajectory(kappa, output):
    """Save exact-SD iterates on one member of the quadratic family."""
    plt = _pyplot()
    matrix, points, _, _ = quadratic_relative_gap_trajectory(kappa)
    fig, ax = plt.subplots(figsize=(4.9, 4.25))
    _quadratic_contours(ax, matrix)
    ax.plot(points[:, 0], points[:, 1], color="#0072B2", lw=1.25, zorder=3)
    visible = min(len(points), 55 if kappa > 10 else len(points))
    ax.plot(
        points[:visible, 0],
        points[:visible, 1],
        color="#0072B2",
        marker="o",
        markersize=2.5,
        lw=1.25,
        zorder=4,
    )
    ax.scatter(*points[0], marker="s", s=34, color="0.15", zorder=5, label=r"$x^{(0)}$")
    ax.scatter(0, 0, marker="*", s=85, color="#D55E00", zorder=5, label=r"$x^*$")
    ax.text(
        0.03,
        0.93,
        rf"curvature ratio $r={kappa:g}$" + "\n" + rf"SD: {len(points)-1} updates",
        transform=ax.transAxes,
        va="top",
    )
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    return _save_and_close(plt, fig, output)


def save_condition_rates(output):
    """Save observed relative objective gaps for two condition numbers."""
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    styles = (("#0072B2", "o"), ("#D55E00", "s"))
    for kappa, (color, marker) in zip(KAPPAS, styles):
        _, _, gaps, _ = quadratic_relative_gap_trajectory(kappa)
        relative_gaps = gaps / gaps[0]
        iterations = np.arange(len(relative_gaps))
        mark_every = 1 if kappa == KAPPAS[0] else 10
        ax.semilogy(
            iterations,
            relative_gaps,
            color=color,
            marker=marker,
            markevery=mark_every,
            markersize=3.5,
            lw=1.8,
            label=rf"$\kappa={kappa:g}$: {len(iterations)-1} updates",
        )
    ax.axhline(
        RELATIVE_GAP_TOL,
        color="0.35",
        lw=1.0,
        ls="--",
        label=r"target $10^{-6}$",
    )
    ax.set_xlabel(r"iteration $k$")
    ax.set_ylabel(r"relative objective gap $E^{(k)}/E^{(0)}$")
    ax.set_xlim(0, 180)
    ax.set_ylim(3e-7, 1.4)
    ax.grid(True, color="0.90", linewidth=0.6)
    ax.tick_params(direction="out", length=3)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    return _save_and_close(plt, fig, output)


def _symmetric_quadratic_contours(ax, matrix, outer_level, coordinate):
    """Draw centered quadratic level curves on equal-scale coordinate axes."""
    smallest_eigenvalue = float(np.min(np.linalg.eigvalsh(matrix)))
    radius = 1.08 * np.sqrt(2.0 * outer_level / smallest_eigenvalue)
    grid = np.linspace(-radius, radius, 420)
    coordinate_1, coordinate_2 = np.meshgrid(grid, grid)
    points = np.stack((coordinate_1, coordinate_2), axis=-1)
    values = 0.5 * np.einsum("...i,ij,...j->...", points, matrix, points)
    fractions = np.array([0.025, 0.08, 0.20, 0.45, 0.72, 1.0])
    ax.contour(
        coordinate_1,
        coordinate_2,
        values,
        levels=outer_level * fractions,
        colors=["0.82", "0.78", "0.73", "0.67", "0.60", "0.45"],
        linewidths=[0.65, 0.65, 0.7, 0.75, 0.85, 1.05],
    )
    ax.axhline(0.0, color="0.90", lw=0.6)
    ax.axvline(0.0, color="0.90", lw=0.6)
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(rf"${coordinate}_1$")
    ax.set_ylabel(rf"${coordinate}_2$")
    ax.tick_params(direction="out", length=2.5)


def save_preconditioner_trajectories(output):
    """Compare three SD trajectories in a zoomed view of the relevant valley."""
    plt = _pyplot()
    matrix, start, preconditioners = preconditioning_example()
    methods = (
        ("pure", "Pure SD ($B=I$)", "#0072B2", "o", "-"),
        ("jacobi", r"Jacobi ($B=\operatorname{diag}(Q)$)", "#D55E00", "s", "--"),
        ("ideal", "Ideal ($B=Q$)", "#009E73", "^", "-."),
    )
    trajectories = {}
    for key, *_ in methods:
        trajectories[key] = preconditioned_quadratic_trajectory(
            matrix,
            start,
            preconditioner=preconditioners[key],
        )

    fig, path_ax = plt.subplots(figsize=(8.4, 3.25))
    initial_gap = float(0.5 * start @ matrix @ start)
    _symmetric_quadratic_contours(path_ax, matrix, initial_gap, "x")
    for key, label, color, marker, line_style in methods:
        points, _, _ = trajectories[key]
        marker_spacing = max(1, (len(points) - 1) // 12)
        path_ax.plot(
            points[:, 0],
            points[:, 1],
            color=color,
            lw=2.2,
            ls=line_style,
            marker=marker,
            markevery=marker_spacing,
            markersize=4.2,
            zorder=3,
            label=f"{label}: {len(points)-1} updates",
        )
        path_ax.annotate(
            "",
            xy=points[1],
            xytext=points[0],
            arrowprops={"arrowstyle": "->", "color": color, "lw": 2.0},
            zorder=4,
        )
    path_ax.scatter(*start, marker="s", s=38, color="0.12", zorder=5)
    path_ax.scatter(0.0, 0.0, marker="*", s=88, color="0.12", zorder=5)
    path_ax.annotate(
        r"$\mathbf{x}^{(0)}$",
        xy=start,
        xytext=(-28, -15),
        textcoords="offset points",
        fontsize=9,
    )
    path_ax.annotate(
        r"$\mathbf{x}^*$",
        xy=(0.0, 0.0),
        xytext=(7, 7),
        textcoords="offset points",
        fontsize=9,
    )
    # The iterates occupy a thin strip of the original square plotting window.
    # Zooming to that strip makes the distinct search directions visible while
    # retaining equal physical scaling on the two coordinate axes.
    path_ax.set_xlim(-0.12, 4.08)
    path_ax.set_ylim(-0.08, 0.90)
    path_ax.set_aspect("equal", adjustable="box")
    path_ax.set_title(
        "zoomed trajectories in $x$-space; same start and exact line search"
    )
    path_ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        fontsize=9.2,
        handlelength=2.6,
        columnspacing=1.2,
    )
    fig.tight_layout()
    return _save_and_close(plt, fig, output)


def save_sublinear_rate(output):
    """Save numerical evidence of Q-sublinear convergence for flat_phi."""
    plt = _pyplot()
    _, gaps, ratios = flat_sublinear_history()
    k = np.arange(1, len(gaps))
    reference = gaps[1000] * (1000.0 / k) ** 2
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.15))
    axes[0].loglog(
        k,
        gaps[1:],
        color="#0072B2",
        lw=2.0,
        label=r"$\varphi(x^{(k)})$",
    )
    axes[0].loglog(
        k,
        reference,
        color="#D55E00",
        lw=1.5,
        ls="--",
        label=r"reference $C/k^2$",
    )
    axes[0].set_xlabel(r"iteration $k$")
    axes[0].set_ylabel("objective gap")
    axes[0].legend(frameon=False)
    axes[1].semilogx(k, ratios, color="#0072B2", lw=1.7)
    axes[1].axhline(1.0, color="0.35", lw=1.0, ls="--")
    axes[1].set_xlabel(r"iteration $k$")
    axes[1].set_ylabel(r"$\varphi(x^{(k+1)})/\varphi(x^{(k)})$")
    axes[1].set_ylim(0.75, 1.005)
    axes[1].annotate(
        "ratio tends to 1",
        xy=(5000, ratios[4999]),
        xytext=(180, 0.91),
        arrowprops={"arrowstyle": "->", "color": "0.35"},
    )
    for ax in axes:
        ax.grid(True, color="0.90", linewidth=0.6)
        ax.tick_params(direction="out", length=3)
    fig.tight_layout(w_pad=1.5)
    return _save_and_close(plt, fig, output)


FIGURE_SPECS = {
    "moderate": (
        "l10_quadratic_moderate.pdf",
        partial(save_quadratic_trajectory, KAPPAS[0]),
    ),
    "flat": (
        "l10_quadratic_flat.pdf",
        partial(save_quadratic_trajectory, KAPPAS[1]),
    ),
    "rates": ("l10_condition_rates.pdf", save_condition_rates),
    "preconditioners": (
        "l10_preconditioner_trajectories.pdf",
        save_preconditioner_trajectories,
    ),
    "sublinear": ("l10_sublinear_rate.pdf", save_sublinear_rate),
}

def save_figure(name, output_dir=Path(".")):
    """Save one named experimental figure in ``output_dir``."""
    try:
        filename, save = FIGURE_SPECS[name]
    except KeyError as error:
        raise ValueError(f"unknown figure: {name}") from error
    return save(Path(output_dir) / filename)


def save_all_figures(output_dir=Path(".")):
    """Save all experimental figures in ``output_dir``."""
    return [save_figure(name, output_dir) for name in FIGURE_SPECS]


# -----------------------------------------------------------------------------
# Command-line interface
# -----------------------------------------------------------------------------


def print_general_example():
    solution, rows = steepest_descent(
        quartic_objective,
        quartic_gradient,
        [2.0, -1.0],
        tol=1e-6,
        max_iter=100,
    )
    accepted_steps = [row[3] for row in rows if row[3] is not None]
    print("General line-search SD example (Armijo backtracking)")
    print("f(x) = 0.25*x1^4 + 0.5*x1^2 + 0.5*(x2-1)^2")
    print(f"updates = {len(accepted_steps)}")
    print(f"final x = ({solution[0]:.8f}, {solution[1]:.8f})")
    print(f"final gradient norm = {rows[-1][2]:.3e}")
    print(f"accepted steps = {accepted_steps}")


def print_original_example():
    solution, rows = quadratic_steepest_descent(Q, c, [0.0, 0.0])
    print("Original quadratic example")
    print("iter   x[0]    x[1]      ||grad||      step")
    print("------------------------------------------------")
    shown = rows[:4] + [None] + rows[-4:]
    for row in shown:
        if row is None:
            print(" ...")
            continue
        k, x, norm_g, alpha = row
        step = "   --" if alpha is None else f"{alpha:6.3f}"
        print(f"{k:3d}  {x[0]:7.3f} {x[1]:7.3f}   {norm_g:9.2e}   {step}")
    print(f"final x = ({solution[0]:.8f}, {solution[1]:.8f})")


def print_reference_results():
    print("\nConditioning comparison (relative gap <= 1e-6)")
    print("kappa   SD updates")
    print("------------------")
    for kappa in KAPPAS:
        _, points, _, _ = quadratic_relative_gap_trajectory(kappa)
        print(f"{kappa:5.0f}       {len(points)-1:4d}")

    matrix, start, preconditioners = preconditioning_example()
    labels = {"pure": "Pure SD", "jacobi": "Jacobi SD", "ideal": "Ideal SD"}
    print("\nPreconditioning comparison (relative gap <= 1e-6)")
    print("method          effective kappa   updates")
    print("-----------------------------------------")
    for name, preconditioner in preconditioners.items():
        inverse_half = symmetric_pd_inverse_half(preconditioner)
        transformed = inverse_half @ matrix @ inverse_half
        points, _, _ = preconditioned_quadratic_trajectory(
            matrix,
            start,
            preconditioner,
        )
        print(
            f"{labels[name]:12s}   {np.linalg.cond(transformed):15.5f}"
            f"   {len(points)-1:7d}"
        )

    _, gaps, ratios = flat_sublinear_history()
    print(
        f"\nFlat smooth convex example: L = {FLAT_L:.6f}, "
        f"alpha = 1/L = {1/FLAT_L:.6f}"
    )
    print("k       gap            one-step ratio       k^2 gap")
    print("----------------------------------------------------")
    for k in (100, 1000, 10000):
        print(
            f"{k:5d}   {gaps[k]:.6e}      {ratios[k-1]:.9f}      "
            f"{k*k*gaps[k]:.6f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--example",
        choices=["armijo", "quadratic-exact", "all"],
        help="numerical example to print (default: all when no figure is requested)",
    )
    parser.add_argument(
        "--figure",
        choices=[*FIGURE_SPECS, "all"],
        help="save one experimental figure or all experimental figures",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="figure directory (default: current working directory)",
    )
    args = parser.parse_args()
    selected_example = args.example
    if selected_example is None and args.figure is None:
        selected_example = "all"
    if selected_example in ("armijo", "all"):
        print_general_example()
    if selected_example in ("quadratic-exact", "all"):
        if selected_example == "all":
            print()
        print_original_example()
    if selected_example == "all":
        print_reference_results()
    if args.figure == "all":
        paths = save_all_figures(args.output_dir)
    elif args.figure is not None:
        paths = [save_figure(args.figure, args.output_dir)]
    else:
        paths = []
    for path in paths:
        print(f"saved {path}")


if __name__ == "__main__":
    main()
