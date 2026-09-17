"""Numerical examples for Lecture 11: conjugate directions and CG."""

import argparse
from pathlib import Path

import numpy as np


# Constants and shared configuration

# One example is used throughout the lecture.
Q = np.array([[5.0, -4.0], [-4.0, 5.0]])
c = np.zeros(2)
X0 = np.array([1.0, 0.8])
X_STAR = np.linalg.solve(Q, c)

# Rows are search directions. Each pair is Q-conjugate.
CD_BASES = {
    "cd-a": np.array([[-1.0, -1.0], [-1.0, 1.0]]),
    "cd-b": np.array([[-1.0, -2.0], [-2.0, -1.0]]),
}

COMMON_METHODS = ("sd", "cd-a", "cd-b", "cg")
RELATIVE_GAP_TOL = 1e-6
LOG_FLOOR = 1e-15
PCG_DIMENSION = 100
PCG_TARGET = 1e-8
HILBERT_DIMENSION = 4
RESTART_DIMENSION = 20
RESTART_PERIOD = 30
MATVEC_BUDGET = 100


# Algorithms


def _energy_gap(Q, x, x_star):
    """Return one-half the squared Q-norm of x-x_star."""
    error = np.asarray(x, dtype=float) - x_star
    return 0.5 * error @ Q @ error


def quadratic_gradient(Q, c, x):
    """Return g(x) = Qx - c."""
    return Q @ np.asarray(x, dtype=float) - c


def _exact_line_step_from_gradient(Q, g, d):
    """Return the exact step using a gradient already available to the caller."""
    d = np.asarray(d, dtype=float)
    denominator = d @ Q @ d
    if denominator <= 0.0:
        raise ValueError("Q must be symmetric PD and d must be nonzero")
    return -(g @ d) / denominator


def exact_line_step(Q, c, x, d):
    """Return the exact minimizer along x + alpha d."""
    g = quadratic_gradient(Q, c, x)
    return _exact_line_step_from_gradient(Q, g, d)


def steepest_descent(
    Q, c, x0, relative_gap_tol=RELATIVE_GAP_TOL, max_iter=1000
):
    """Exact-line-search steepest descent for a PD quadratic."""
    x = np.asarray(x0, dtype=float).copy()
    x_star = np.linalg.solve(Q, c)
    points = [x.copy()]
    gaps = [_energy_gap(Q, x, x_star)]
    if gaps[0] == 0.0:
        return np.asarray(points), np.asarray(gaps)

    for _ in range(max_iter):
        if gaps[-1] / gaps[0] <= relative_gap_tol:
            return np.asarray(points), np.asarray(gaps)
        g = quadratic_gradient(Q, c, x)
        d = -g
        alpha = _exact_line_step_from_gradient(Q, g, d)
        x = x + alpha * d
        points.append(x.copy())
        gaps.append(_energy_gap(Q, x, x_star))
    if gaps[-1] / gaps[0] <= relative_gap_tol:
        return np.asarray(points), np.asarray(gaps)
    raise RuntimeError("steepest descent reached max_iter before convergence")


def conjugate_direction(Q, c, x0, directions):
    """Run exact line searches along supplied Q-conjugate directions."""
    x = np.asarray(x0, dtype=float).copy()
    x_star = np.linalg.solve(Q, c)
    directions = np.asarray(directions, dtype=float)
    points = [x.copy()]
    gaps = [_energy_gap(Q, x, x_star)]
    for d in directions:
        alpha = exact_line_step(Q, c, x, d)
        x = x + alpha * d
        points.append(x.copy())
        gaps.append(_energy_gap(Q, x, x_star))
    return np.asarray(points), np.asarray(gaps)


def conjugate_gradient(Q, c, x0, tol=1e-12, max_iter=None):
    """Linear CG for Qx=c, using the gradient residual g=Qx-c."""
    x = np.asarray(x0, dtype=float).copy()
    x_star = np.linalg.solve(Q, c)
    g = quadratic_gradient(Q, c, x)
    d = -g
    points = [x.copy()]
    gaps = [_energy_gap(Q, x, x_star)]
    if max_iter is None:
        max_iter = Q.shape[0]

    for _ in range(max_iter):
        if np.linalg.norm(g) <= tol:
            break
        Qd = Q @ d
        denominator = d @ Qd
        if denominator <= 0.0:
            raise ValueError("CG requires a symmetric PD matrix Q")
        alpha = (g @ g) / denominator
        x = x + alpha * d
        g_next = g + alpha * Qd
        points.append(x.copy())
        gaps.append(_energy_gap(Q, x, x_star))
        if np.linalg.norm(g_next) <= tol:
            break
        beta = (g_next @ g_next) / (g @ g)
        d = -g_next + beta * d
        g = g_next
    return np.asarray(points), np.asarray(gaps)


def preconditioned_conjugate_gradient(
    Q,
    c,
    x0,
    solve_preconditioner=None,
    relative_energy_tol=1e-8,
    max_iter=None,
):
    """Run CG or PCG and record quadratic-energy gaps.

    ``solve_preconditioner(g)`` must return ``B^{-1} g`` for a fixed symmetric PD
    preconditioner B. Passing ``None`` recovers ordinary CG.
    """
    x = np.asarray(x0, dtype=float).copy()
    x_star = np.linalg.solve(Q, c)
    g = quadratic_gradient(Q, c, x)

    def apply_preconditioner(vector):
        if solve_preconditioner is None:
            return vector.copy()
        return np.asarray(solve_preconditioner(vector), dtype=float)

    z = apply_preconditioner(g)
    d = -z
    gamma = g @ z
    points = [x.copy()]
    gaps = [_energy_gap(Q, x, x_star)]
    if gaps[0] == 0.0:
        return np.asarray(points), np.asarray(gaps)
    if max_iter is None:
        max_iter = Q.shape[0]

    for _ in range(max_iter):
        if np.sqrt(gaps[-1] / gaps[0]) <= relative_energy_tol:
            return np.asarray(points), np.asarray(gaps)
        if gamma <= 0.0:
            raise ValueError("the preconditioner must be symmetric PD")
        Qd = Q @ d
        denominator = d @ Qd
        if denominator <= 0.0:
            raise ValueError("PCG requires a symmetric PD matrix Q")
        alpha = gamma / denominator
        x = x + alpha * d
        g_next = g + alpha * Qd
        points.append(x.copy())
        gaps.append(_energy_gap(Q, x, x_star))
        if np.sqrt(gaps[-1] / gaps[0]) <= relative_energy_tol:
            return np.asarray(points), np.asarray(gaps)
        z_next = apply_preconditioner(g_next)
        gamma_next = g_next @ z_next
        beta = gamma_next / gamma
        d = -z_next + beta * d
        g = g_next
        gamma = gamma_next
    raise RuntimeError("PCG reached max_iter before the requested tolerance")


# Test problems and numerical experiments


def run_method(name):
    """Run one method on the common lecture example."""
    if name == "sd":
        return steepest_descent(Q, c, X0)
    if name in CD_BASES:
        return conjugate_direction(Q, c, X0, CD_BASES[name])
    if name == "cg":
        return conjugate_gradient(Q, c, X0)
    raise ValueError(f"unknown method: {name}")


def common_example_results(names=COMMON_METHODS):
    """Return trajectories and gaps for selected common-example methods."""
    return {name: run_method(name) for name in names}


def _pcg_test_problem(dimension=PCG_DIMENSION):
    """Return the scaled tridiagonal system used for the PCG comparison."""
    off_diagonal = -0.25 * np.ones(dimension - 1)
    tridiagonal = (
        np.eye(dimension)
        + np.diag(off_diagonal, 1)
        + np.diag(off_diagonal, -1)
    )
    scales = 10.0 ** (np.arange(dimension) / (dimension - 1))
    matrix = (scales[:, None] * tridiagonal) * scales[None, :]
    solution = np.ones(dimension)
    right_hand_side = matrix @ solution
    initial_point = np.zeros(dimension)
    return matrix, right_hand_side, initial_point, np.diag(matrix)


def pcg_experiment_results():
    """Return histories and condition numbers for the Jacobi-PCG experiment."""
    matrix, right_hand_side, initial_point, jacobi_diagonal = _pcg_test_problem()
    _, cg_gaps = preconditioned_conjugate_gradient(
        matrix,
        right_hand_side,
        initial_point,
        relative_energy_tol=PCG_TARGET,
        max_iter=PCG_DIMENSION,
    )
    _, pcg_gaps = preconditioned_conjugate_gradient(
        matrix,
        right_hand_side,
        initial_point,
        solve_preconditioner=lambda vector: vector / jacobi_diagonal,
        relative_energy_tol=PCG_TARGET,
        max_iter=PCG_DIMENSION,
    )
    inverse_sqrt_diagonal = 1.0 / np.sqrt(jacobi_diagonal)
    preconditioned_matrix = (
        inverse_sqrt_diagonal[:, None]
        * matrix
        * inverse_sqrt_diagonal[None, :]
    )
    return {
        "target": PCG_TARGET,
        "histories": {
            "CG": np.sqrt(cg_gaps / cg_gaps[0]),
            "Jacobi PCG": np.sqrt(pcg_gaps / pcg_gaps[0]),
        },
        "condition_numbers": {
            "CG": np.linalg.cond(matrix),
            "Jacobi PCG": np.linalg.cond(preconditioned_matrix),
        },
    }


def _rounded_hilbert_system(dimension=HILBERT_DIMENSION):
    """Return one stored float32 Hilbert system and its float64 matrix."""
    indices = np.arange(1, dimension + 1, dtype=np.float64)
    hilbert = 1.0 / (indices[:, None] + indices[None, :] - 1.0)
    matrix32 = hilbert.astype(np.float32)
    right_hand_side32 = (
        hilbert @ np.ones(dimension, dtype=np.float64)
    ).astype(np.float32)
    return matrix32, right_hand_side32, matrix32.astype(np.float64)


def _scaled_poisson_system(dimension=RESTART_DIMENSION):
    """Return the scaled-Poisson test problem for residual restarts."""
    tridiagonal = (
        2.0 * np.eye(dimension)
        - np.eye(dimension, k=1)
        - np.eye(dimension, k=-1)
    )
    scaling = np.geomspace(1.0, 10.0, dimension)
    matrix = scaling[:, None] * tridiagonal * scaling[None, :]
    matrix /= np.linalg.eigvalsh(matrix)[-1]
    matrix32 = matrix.astype(np.float32)
    right_hand_side32 = matrix32[:, 0].copy()
    solution = np.zeros(dimension, dtype=np.float64)
    solution[0] = 1.0
    return (
        matrix32,
        right_hand_side32,
        matrix32.astype(np.float64),
        solution,
    )


def _cg_conjugacy_defects(dtype):
    """Measure loss of mutual Q-conjugacy among Hilbert CG directions."""
    matrix32, right_hand_side32, matrix64 = _rounded_hilbert_system()
    matrix = matrix32.astype(dtype)
    residual = right_hand_side32.astype(dtype)
    direction = residual.copy()
    residual_norm_squared = residual @ residual
    directions = []
    defects = []

    for index in range(matrix.shape[0]):
        direction64 = direction.astype(np.float64)
        if directions:
            matrix_direction = matrix64 @ direction64
            current_energy = direction64 @ matrix_direction
            normalized_cross_terms = [
                abs(previous @ matrix_direction)
                / np.sqrt(
                    (previous @ matrix64 @ previous) * current_energy
                )
                for previous in directions
            ]
            defects.append(max(normalized_cross_terms))
        directions.append(direction64.copy())

        matrix_direction = matrix @ direction
        alpha = residual_norm_squared / (direction @ matrix_direction)
        residual_next = residual - alpha * matrix_direction
        residual_norm_squared_next = residual_next @ residual_next
        if index + 1 < matrix.shape[0]:
            beta = residual_norm_squared_next / residual_norm_squared
            direction = residual_next + beta * direction
        residual = residual_next
        residual_norm_squared = residual_norm_squared_next

    return np.arange(1, matrix.shape[0]), np.asarray(defects)


def _finite_precision_cg_history(
    problem, restart_period=None, matvec_budget=MATVEC_BUDGET
):
    """Run float32 CG with float64 energy-error diagnostics."""
    matrix32, right_hand_side32, matrix64, solution = problem
    x = np.zeros(right_hand_side32.size, dtype=np.float32)
    residual = right_hand_side32.copy()
    direction = residual.copy()
    residual_norm_squared = residual @ residual
    matvecs = 0
    updates = 0
    costs = []
    energy_errors = []
    refresh_costs = []
    initial_error = np.sqrt(solution @ matrix64 @ solution)

    def record():
        error = x.astype(np.float64) - solution
        costs.append(matvecs)
        energy_errors.append(
            np.sqrt(error @ matrix64 @ error) / initial_error
        )

    record()
    while matvecs < matvec_budget:
        matrix_direction = matrix32 @ direction
        matvecs += 1
        alpha = residual_norm_squared / (direction @ matrix_direction)
        x = x + alpha * direction
        residual = residual - alpha * matrix_direction
        updates += 1
        record()

        if (
            restart_period is not None
            and updates % restart_period == 0
            and matvecs < matvec_budget
        ):
            residual = right_hand_side32 - matrix32 @ x
            matvecs += 1
            refresh_costs.append(matvecs)
            record()
            residual_norm_squared_next = residual @ residual
            direction = residual.copy()
        else:
            residual_norm_squared_next = residual @ residual
            beta = residual_norm_squared_next / residual_norm_squared
            direction = residual + beta * direction
        residual_norm_squared = residual_norm_squared_next

    return {
        "costs": np.asarray(costs),
        "energy_errors": np.asarray(energy_errors),
        "refresh_costs": np.asarray(refresh_costs, dtype=int),
    }


def finite_precision_experiment_results():
    """Return conjugacy-defect and residual-restart experiment data."""
    indices, defects64 = _cg_conjugacy_defects(np.float64)
    _, defects32 = _cg_conjugacy_defects(np.float32)
    problem = _scaled_poisson_system()
    return {
        "indices": indices,
        "defects64": defects64,
        "defects32": defects32,
        "dimension": RESTART_DIMENSION,
        "matvec_budget": MATVEC_BUDGET,
        "restart_period": RESTART_PERIOD,
        "ordinary": _finite_precision_cg_history(problem),
        "restarted": _finite_precision_cg_history(
            problem, restart_period=RESTART_PERIOD
        ),
    }


# Experiment figures


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "legend.fontsize": 8.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    return plt


def _save(fig, output):
    import matplotlib.pyplot as plt

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return output


def _latex_scientific(value, decimal_places=2):
    """Format a positive scalar for a compact LaTeX annotation."""
    exponent = int(np.floor(np.log10(value)))
    coefficient = value / 10.0**exponent
    return rf"{coefficient:.{decimal_places}f}\times10^{{{exponent}}}"


def _add_quadratic_contours(ax):
    grid = np.linspace(-0.35, 1.2, 420)
    x1, x2 = np.meshgrid(grid, grid)
    values = 0.5 * (
        Q[0, 0] * x1**2
        + 2.0 * Q[0, 1] * x1 * x2
        + Q[1, 1] * x2**2
    ) - c[0] * x1 - c[1] * x2
    levels = [0.01, 0.03, 0.09, 0.18, 0.36, 0.72, 0.9, 1.25]
    ax.contour(x1, x2, values, levels=levels, colors="0.76", linewidths=0.8)
    ax.set_xlim(-0.35, 1.2)
    ax.set_ylim(-0.35, 1.15)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.tick_params(direction="out", length=3)


def save_cd_two_bases(output):
    """Compare two Q-conjugate bases and exact-search SD in one plot."""
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(6.8, 4.15))
    _add_quadratic_contours(ax)

    specifications = (
        (
            "cd-a",
            "#0072B2",
            "s",
            "-",
            r"CD-A: $(1,1),(1,-1)$",
        ),
        (
            "cd-b",
            "#CC79A7",
            "D",
            "--",
            r"CD-B: $(1,2),(2,1)$",
        ),
        ("sd", "#D55E00", "o", "-", "exact-search SD"),
    )
    results = common_example_results(
        tuple(name for name, *_ in specifications)
    )
    for name, color, marker, line_style, label in specifications:
        points, _ = results[name]
        mark_every = (
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 24, 28]
            if name == "sd"
            else 1
        )
        ax.plot(
            points[:, 0],
            points[:, 1],
            color=color,
            marker=marker,
            markevery=mark_every,
            ls=line_style,
            lw=2.0 if name != "sd" else 1.7,
            markersize=4.2,
            label=label,
            zorder=3 if name == "sd" else 4,
        )

    sd_points, _ = results["sd"]
    cd_a_points, _ = results["cd-a"]
    cd_b_points, _ = results["cd-b"]
    ax.scatter(*X0, marker="s", s=48, color="0.12", zorder=6)
    ax.scatter(*X_STAR, marker="*", s=110, color="0.12", zorder=6)
    ax.text(X0[0] + 0.025, X0[1] + 0.04, r"$x^{(0)}$", fontsize=9)
    ax.text(X_STAR[0] - 0.11, X_STAR[1] + 0.06, r"$x^*$", fontsize=9)
    ax.annotate(
        r"$x_{\rm SD}^{(1)}$",
        xy=sd_points[1],
        xytext=(0.48, 0.98),
        color="#D55E00",
        fontsize=8.5,
        arrowprops={"arrowstyle": "-", "color": "#D55E00", "lw": 0.8},
    )
    ax.annotate(
        r"$x_{\rm A}^{(1)}$",
        xy=cd_a_points[1],
        xytext=(0.19, -0.19),
        color="#0072B2",
        fontsize=8.5,
        arrowprops={"arrowstyle": "-", "color": "#0072B2", "lw": 0.8},
    )
    ax.annotate(
        r"$x_{\rm B}^{(1)}$",
        xy=cd_b_points[1],
        xytext=(0.88, 0.30),
        color="#CC79A7",
        fontsize=8.5,
        arrowprops={"arrowstyle": "-", "color": "#CC79A7", "lw": 0.8},
    )
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    return _save(fig, output)


def save_method_comparison(output):
    """Compare trajectories and relative objective gaps for all methods."""
    plt = _pyplot()
    results = common_example_results()
    styles = {
        "sd": ("#D55E00", "o", "-", "SD"),
        "cd-a": ("#0072B2", "s", "-", "CD-A"),
        "cd-b": ("#CC79A7", "D", "-", "CD-B"),
        "cg": ("#009E73", "^", "--", "CG"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(9.7, 4.15))
    trajectory_ax, gap_ax = axes
    _add_quadratic_contours(trajectory_ax)
    for name in COMMON_METHODS:
        points, _ = results[name]
        color, marker, line_style, label = styles[name]
        trajectory_ax.plot(
            points[:, 0],
            points[:, 1],
            color=color,
            marker=marker,
            markevery=2 if name == "sd" else 1,
            markersize=3.8,
            lw=1.7,
            ls=line_style,
            label=label,
            zorder=4,
        )
    trajectory_ax.scatter(
        0.0, 0.0, marker="*", s=100, color="0.12", zorder=6
    )
    trajectory_ax.annotate(
        "CG and SD share the first step",
        xy=(0.82, 0.8),
        xytext=(0.24, 1.055),
        fontsize=8.5,
        arrowprops={"arrowstyle": "->", "color": "0.35", "lw": 0.9},
    )
    trajectory_ax.legend(frameon=False, loc="lower right")

    for name in COMMON_METHODS:
        _, gaps = results[name]
        color, marker, line_style, label = styles[name]
        relative_gaps = np.maximum(gaps / gaps[0], LOG_FLOOR)
        iterations = np.arange(len(relative_gaps))
        gap_ax.semilogy(
            iterations,
            relative_gaps,
            color=color,
            marker=marker,
            markevery=2 if name == "sd" else 1,
            markersize=4.0,
            lw=1.7,
            ls=line_style,
            label=label,
        )
    gap_ax.set_xlim(0, 31)
    gap_ax.set_ylim(3e-16, 1.6)
    gap_ax.set_xlabel(r"iteration $k$")
    gap_ax.set_ylabel("relative objective gap")
    gap_ax.grid(True, color="0.90", linewidth=0.6)
    gap_ax.tick_params(direction="out", length=3)
    gap_ax.legend(frameon=False, loc="center right")
    gap_ax.text(
        0.98,
        0.035,
        r"Exact zeros are displayed at $10^{-15}$.",
        transform=gap_ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="0.35",
    )
    fig.tight_layout(w_pad=1.35)
    return _save(fig, output)


def save_pcg_comparison(output):
    """Show how Jacobi PCG removes diagonal scale imbalance."""
    plt = _pyplot()
    experiment = pcg_experiment_results()
    histories = experiment["histories"]
    target = experiment["target"]
    styles = {
        "CG": ("#0072B2", "o", 8),
        "Jacobi PCG": ("#D55E00", "s", 2),
    }

    fig, ax = plt.subplots(figsize=(6.6, 3.65))
    for name, history in histories.items():
        color, marker, marker_interval = styles[name]
        iterations = np.arange(len(history))
        ax.semilogy(
            iterations,
            history,
            color=color,
            marker=marker,
            markevery=marker_interval,
            markersize=4.0,
            lw=1.9,
            label=name,
        )
        ax.scatter(
            iterations[-1],
            history[-1],
            color=color,
            marker=marker,
            s=34,
            zorder=5,
        )
        annotation_offset = (-27, 20) if name == "CG" else (34, 20)
        ax.annotate(
            rf"${iterations[-1]}$ updates",
            xy=(iterations[-1], history[-1]),
            xytext=annotation_offset,
            textcoords="offset points",
            ha="right" if name == "CG" else "left",
            color=color,
            fontsize=8.5,
            arrowprops={"arrowstyle": "-", "color": color, "lw": 0.8},
        )
    ax.axhline(target, color="0.35", lw=1.0, ls="--")
    ax.text(
        0.58,
        target * 1.45,
        rf"target $10^{{{int(np.log10(target))}}}$",
        transform=ax.get_yaxis_transform(),
        ha="left",
        va="bottom",
        fontsize=8.5,
        color="0.30",
    )
    max_updates = max(len(history) - 1 for history in histories.values())
    ax.set_xlim(0, max_updates + 4)
    ax.set_ylim(2e-9, 1.5)
    ax.set_xlabel(r"iteration $k$")
    ax.set_ylabel(r"relative $Q$-norm error")
    ax.grid(True, color="0.90", linewidth=0.6)
    ax.tick_params(direction="out", length=3)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    return _save(fig, output)


def save_cg_restart_comparison(output):
    """Show conjugacy loss and a clear benefit from residual restarts."""
    plt = _pyplot()
    experiment = finite_precision_experiment_results()
    indices = experiment["indices"]
    defects64 = experiment["defects64"]
    defects32 = experiment["defects32"]
    ordinary = experiment["ordinary"]
    restarted = experiment["restarted"]
    ordinary_costs = ordinary["costs"]
    ordinary_errors = ordinary["energy_errors"]
    restart_costs_axis = restarted["costs"]
    restart_errors = restarted["energy_errors"]
    refresh_costs = restarted["refresh_costs"]
    improvement = ordinary_errors[-1] / restart_errors[-1]

    fig, (left, right) = plt.subplots(1, 2, figsize=(10.1, 3.45))
    left.semilogy(
        indices,
        defects64,
        color="0.35",
        marker="o",
        markersize=4.5,
        lw=1.8,
        label="float64",
    )
    left.semilogy(
        indices,
        defects32,
        color="#0072B2",
        marker="s",
        markersize=4.5,
        lw=1.8,
        label="float32",
    )
    left.annotate(
        rf"$\eta_{{{indices[-1]}}}={defects32[-1]:.4f}$",
        xy=(indices[-1], defects32[-1]),
        xytext=(-8, -28),
        textcoords="offset points",
        ha="right",
        color="#0072B2",
        fontsize=9.2,
        arrowprops={"arrowstyle": "-", "color": "#0072B2", "lw": 0.8},
    )
    left.set_xticks(indices)
    left.set_xlim(indices[0] - 0.2, indices[-1] + 0.2)
    left.set_ylim(4e-15, 1.8)
    left.set_xlabel(r"new direction index $k$")
    left.set_ylabel(r"conjugacy defect $\eta_k$")
    left.set_title(
        rf"Hilbert-{HILBERT_DIMENSION}: loss of $Q$-conjugacy"
    )
    left.grid(True, color="0.90", linewidth=0.6)
    left.tick_params(direction="out", length=3)
    left.legend(frameon=False, loc="lower right")

    right.semilogy(
        ordinary_costs,
        ordinary_errors,
        color="#0072B2",
        marker="o",
        markevery=10,
        markersize=3.8,
        lw=1.9,
        label="ordinary CG",
    )
    right.semilogy(
        restart_costs_axis,
        restart_errors,
        color="#D55E00",
        marker="s",
        markevery=10,
        markersize=3.8,
        lw=1.9,
        label=f"restart every {experiment['restart_period']} updates",
    )
    refresh_indices = [
        int(np.flatnonzero(restart_costs_axis == cost)[0])
        for cost in refresh_costs
    ]
    right.scatter(
        refresh_costs,
        restart_errors[refresh_indices],
        marker="v",
        s=42,
        facecolors="white",
        edgecolors="#D55E00",
        linewidths=1.1,
        zorder=5,
        label="residual refresh",
    )
    right.axvline(experiment["dimension"], color="0.55", lw=1.0, ls=":")
    right.text(
        experiment["dimension"] + 0.8,
        0.22,
        rf"exact-arithmetic limit $n={experiment['dimension']}$",
        rotation=90,
        va="center",
        ha="left",
        fontsize=8.0,
        color="0.38",
    )
    right.annotate(
        "ordinary CG stalls\n"
        + f"at ${_latex_scientific(ordinary_errors[-1])}$",
        xy=(ordinary_costs[-1], ordinary_errors[-1]),
        xytext=(0.51, 0.53),
        textcoords="axes fraction",
        ha="center",
        va="center",
        color="#0072B2",
        fontsize=9.0,
        arrowprops={"arrowstyle": "-", "color": "#0072B2", "lw": 0.8},
    )
    right.annotate(
        "restart repairs the residual gap\n"
        + f"${_latex_scientific(restart_errors[-1])}$ "
        + "($"
        + f"{improvement:.0f}"
        + r"\times$ lower)",
        xy=(restart_costs_axis[-1], restart_errors[-1]),
        xytext=(0.51, 0.31),
        textcoords="axes fraction",
        ha="center",
        va="center",
        color="#D55E00",
        fontsize=9.0,
        arrowprops={"arrowstyle": "-", "color": "#D55E00", "lw": 0.8},
    )
    right.set_xlim(0, experiment["matvec_budget"] + 1.5)
    right.set_ylim(4e-10, 1.5)
    right.set_xlabel("matrix-vector products")
    right.set_ylabel(r"relative $Q$-norm error")
    right.set_title("Scaled-Poisson residual gap")
    right.grid(True, color="0.90", linewidth=0.6)
    right.tick_params(direction="out", length=3)
    right.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    return _save(fig, output)


FIGURE_SPECS = {
    "bases": ("l11_cd_two_bases.pdf", save_cd_two_bases),
    "comparison": ("l11_method_comparison.pdf", save_method_comparison),
    "pcg": ("l11_pcg_comparison.pdf", save_pcg_comparison),
    "restart": ("l11_cg_restart_comparison.pdf", save_cg_restart_comparison),
}


def save_figure(name, output_dir=Path(".")):
    """Save one named experiment figure in ``output_dir``."""
    try:
        filename, save = FIGURE_SPECS[name]
    except KeyError as error:
        raise ValueError(f"unknown figure: {name}") from error
    return save(Path(output_dir) / filename)


def save_all_figures(output_dir=Path(".")):
    """Save every retained experiment figure in ``output_dir``."""
    return [save_figure(name, output_dir) for name in FIGURE_SPECS]


# Command-line interface


def print_common_results(names):
    results = common_example_results(names)
    print("Common PD quadratic")
    print("Q = [[5,-4],[-4,5]], c = (0,0), x0 = (1,0.8)")
    print("eigenvalues = (1,9), condition number = 9")
    print("method   q1/q0      updates   final relative gap")
    print("------------------------------------------------")
    for name in names:
        _, gaps = results[name]
        first_ratio = gaps[1] / gaps[0] if len(gaps) > 1 else 0.0
        print(
            f"{name:6s}   {first_ratio:8.6f}   {len(gaps)-1:7d}   "
            f"{gaps[-1] / gaps[0]:.6e}"
        )


def print_pcg_results():
    experiment = pcg_experiment_results()
    print("Jacobi-PCG comparison")
    print("method          condition number   updates   final relative Q-norm error")
    print("-----------------------------------------------------------------------")
    for name, history in experiment["histories"].items():
        print(
            f"{name:12s}   {experiment['condition_numbers'][name]:16.4f}"
            f"   {len(history)-1:7d}   {history[-1]:27.4e}"
        )


def print_finite_precision_results():
    experiment = finite_precision_experiment_results()
    ordinary_error = experiment["ordinary"]["energy_errors"][-1]
    restarted_error = experiment["restarted"]["energy_errors"][-1]
    refresh_costs = ", ".join(
        str(cost) for cost in experiment["restarted"]["refresh_costs"]
    )
    defects = ", ".join(f"{value:.4e}" for value in experiment["defects32"])
    print(f"Finite-precision CG: float32 Hilbert-{HILBERT_DIMENSION}")
    print(f"conjugacy defects = [{defects}]")
    print("Scaled-Poisson restart comparison")
    print(f"matrix-vector-product budget = {experiment['matvec_budget']}")
    print(f"ordinary CG final relative Q-norm error = {ordinary_error:.4e}")
    print(f"restart final relative Q-norm error = {restarted_error:.4e}")
    print(f"residual refresh costs = [{refresh_costs}]")
    print(f"improvement factor = {ordinary_error / restarted_error:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algorithm",
        choices=[*COMMON_METHODS, "pcg", "cg-restart", "all"],
        help=(
            "algorithm or experiment to run "
            "(default: all when no figure is requested)"
        ),
    )
    parser.add_argument(
        "--figure",
        choices=[*FIGURE_SPECS, "all"],
        help="save one experiment figure or all four figures",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="figure directory (default: current working directory)",
    )
    args = parser.parse_args()

    selected_algorithm = args.algorithm
    if selected_algorithm is None and args.figure is None:
        selected_algorithm = "all"
    if selected_algorithm == "all":
        print_common_results(COMMON_METHODS)
        print()
        print_pcg_results()
        print()
        print_finite_precision_results()
    elif selected_algorithm == "pcg":
        print_pcg_results()
    elif selected_algorithm == "cg-restart":
        print_finite_precision_results()
    elif selected_algorithm is not None:
        print_common_results((selected_algorithm,))
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
