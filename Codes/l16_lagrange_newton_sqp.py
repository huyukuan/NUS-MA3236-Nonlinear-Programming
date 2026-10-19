"""L16: full-step Lagrange--Newton and damped equality SQP.

Common test: minimize 0.5*((x1-1)^2+(x2-2)^2), x1^2+x2-1=0.
SQP uses an l1 merit function with a fixed penalty and Armijo backtracking.
Both methods check the same fixed normalized residual after each KKT solve.
"""

import argparse
from dataclasses import dataclass
from time import perf_counter

import numpy as np


COMMON_START = (0.5, 0.75)


# Common problem and KKT quantities


def objective(x):
    return float(0.5 * ((x[0] - 1.0)**2 + (x[1] - 2.0)**2))


def gradient(x):
    return np.asarray(x) - np.array([1.0, 2.0])


def constraint(x):
    return float(x[0]**2 + x[1] - 1.0)


def normal(x):
    return np.array([2.0*x[0], 1.0])


def lagrangian_hessian(multiplier):
    return np.diag([1.0 + 2.0*multiplier, 1.0])


def kkt_residuals(x, multiplier):
    return (float(np.linalg.norm(gradient(x) + multiplier*normal(x), np.inf)),
            abs(constraint(x)))


def kkt_matrix(x, multiplier, shift=0.0):
    a = normal(x)[:, None]
    b = lagrangian_hessian(multiplier) + shift*np.eye(2)
    return np.block([[b, a], [a.T, np.zeros((1, 1))]])


# Directions and merit line search


def lagrange_newton_step(x, multiplier):
    """Solve the Newton equation for (d, delta_lambda)."""
    rhs = -np.r_[gradient(x) + multiplier*normal(x), constraint(x)]
    step = np.linalg.solve(kkt_matrix(x, multiplier), rhs)
    return step[:2], float(step[2])


def equality_sqp_step(x, multiplier, shift=0.0):
    """Solve the equality QP for (d, lambda_hat), with B=H_L+shift*I.

    On this problem LICQ always holds. Positive reduced curvature is needed
    for a unique QP minimum; the fixed shift is not automatically adjusted.
    """
    tangent = np.array([1.0, -2.0*x[0]])
    b = lagrangian_hessian(multiplier) + shift*np.eye(2)
    if tangent @ b @ tangent <= 0:
        raise ValueError("nonpositive_reduced_curvature")
    step = np.linalg.solve(kkt_matrix(x, multiplier, shift),
                           -np.r_[gradient(x), constraint(x)])
    return step[:2], float(step[2])


def merit_backtracking(x, direction, rho=2.0, sigma=1e-4, beta=0.5):
    """Armijo for phi=f+rho*abs(g), using A^T d=-g.

    Exact quadratic differences avoid subtracting nearby objective values.
    The slope formula uses the accurately solved linearized constraint.
    The QP is solved before this check: rho affects acceptance, not the QP.
    A failed descent check is reported without changing rho.
    """
    g, grad = constraint(x), gradient(x)
    slope = float(grad @ direction - rho*abs(g))
    if not np.isfinite(slope) or slope >= 0:
        raise ValueError("non_descent_merit_direction")
    alpha = 1.0
    for _ in range(60):
        step = alpha*direction
        g_trial = g + normal(x) @ step + step[0]**2
        change = float(grad @ step + 0.5*(step @ step)
                       + rho*(abs(g_trial) - abs(g)))
        if np.isfinite(change) and change <= sigma*alpha*slope:
            return alpha
        alpha *= beta
    raise ValueError("line_search_failed")


@dataclass
class Result:
    method: str
    x: np.ndarray
    multiplier: float
    history: list
    elapsed: float
    status: str
    linear_solves: int


def _iterate(method, x0, multiplier0, tol, max_iter, inner_tol,
             rho=2.0, shift=0.0):
    """Shared stopping, recording, and update loop for the two basic methods."""
    x, multiplier = np.asarray(x0, dtype=float).copy(), float(multiplier0)
    if (x.shape != (2,) or not np.all(np.isfinite(x))
            or not np.all(np.isfinite([multiplier, tol, inner_tol, rho, shift]))
            or min(tol, inner_tol, rho) <= 0 or shift < 0 or max_iter < 0):
        raise ValueError("require finite data, x0 in R^2, tol,inner_tol,rho>0, shift,max_iter>=0")
    start, history, status = perf_counter(), [], "iteration_limit"
    alpha, linear_residual, linear_solves = 0.0, 0.0, 0
    for k in range(max_iter + 1):
        stat, feas = kkt_residuals(x, multiplier)
        history.append(dict(k=k, x=x.copy(), multiplier=multiplier,
                            stationarity=stat, feasibility=feas, kkt=max(stat, feas),
                            alpha=alpha, shift=shift, linear_residual=linear_residual,
                            inner_tolerance=inner_tol))
        if not np.all(np.isfinite([stat, feas])):
            status = "nonfinite"
            break
        if max(stat, feas) <= tol:
            status = "converged"
            break
        if k == max_iter:
            break
        try:
            if method == "LN":
                direction, delta = lagrange_newton_step(x, multiplier)
                unknown = np.r_[direction, delta]
                rhs = -np.r_[gradient(x) + multiplier*normal(x), constraint(x)]
            else:
                direction, lambda_hat = equality_sqp_step(x, multiplier, shift)
                delta = lambda_hat - multiplier
                unknown = np.r_[direction, lambda_hat]
                rhs = -np.r_[gradient(x), constraint(x)]
            linear_solves += 1
            linear_residual = float(np.linalg.norm(
                kkt_matrix(x, multiplier, shift) @ unknown - rhs, np.inf)
                / (1 + np.linalg.norm(rhs, np.inf)))
            if not np.isfinite(linear_residual):
                status = "nonfinite"
                break
            if linear_residual > inner_tol:
                status = "inaccurate_linear_solve"
                break
            if method != "LN" and max(kkt_residuals(x, lambda_hat)) <= tol:
                # A zero primal step can recover the correct multiplier.
                direction, alpha = np.zeros_like(x), 1.0
            else:
                alpha = 1.0 if method == "LN" else merit_backtracking(x, direction, rho)
        except np.linalg.LinAlgError:
            status = "singular_kkt_matrix"
            break
        except ValueError as error:
            status = str(error)
            break
        trial, new_multiplier = x + alpha*direction, multiplier + alpha*delta
        if not np.all(np.isfinite(np.r_[trial, new_multiplier])):
            status = "nonfinite"
            break
        if np.array_equal(trial, x) and new_multiplier == multiplier:
            status = "stagnation"
            break
        x, multiplier = trial, float(new_multiplier)
    return Result(method, x, multiplier, history, perf_counter()-start, status,
                  linear_solves)


def lagrange_newton(x0=COMMON_START, multiplier0=0.0, tol=1e-6, max_iter=50,
                    inner_tol=1e-10):
    """Pure LN: full steps in both the primal variable and the multiplier."""
    return _iterate("LN", x0, multiplier0, tol, max_iter, inner_tol)


def damped_sqp(x0=COMMON_START, multiplier0=0.0, tol=1e-6, max_iter=50,
               rho=2.0, shift=0.0, inner_tol=1e-10):
    """Fixed rho, shift, and inner accuracy; Armijo has sigma=1e-4, beta=.5.

    Solve for (d, lambda_hat), check merit descent, then choose alpha.
    Update lambda unconditionally by alpha*(lambda_hat-lambda) after an
    accepted step. A fixed rho or shift can fail; neither is adapted.
    """
    return _iterate("damped SQP", x0, multiplier0, tol, max_iter, inner_tol,
                    rho, shift)


def run_comparison(method="all", tol=1e-6, rho=2.0, shift=0.0, inner_tol=1e-10):
    if method not in ("ln", "sqp", "all"):
        raise ValueError("unknown method")
    runs = []
    if method in ("ln", "all"):
        runs.append(lagrange_newton(tol=tol, inner_tol=inner_tol))
    if method in ("sqp", "all"):
        runs.append(damped_sqp(tol=tol, rho=rho, shift=shift, inner_tol=inner_tol))
    return runs


def print_results(runs):
    print("method      KKT_solves    seconds       r_stat       r_feas        r_KKT status")
    for run in runs:
        last = run.history[-1]
        print(f"{run.method:10} {run.linear_solves:10d} {run.elapsed:10.4f} "
              f"{last['stationarity']:12.3e} {last['feasibility']:12.3e} "
              f"{last['kkt']:12.3e} {run.status}")
        print(f"  x={np.array2string(run.x, precision=9)}, lambda={run.multiplier:.9f}")
        print("  accepted alpha:", ", ".join(f"{row['alpha']:g}" for row in run.history[1:]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("ln", "sqp", "all"), default="all")
    parser.add_argument("--tol", type=float, default=1e-6, help="outer KKT tolerance")
    parser.add_argument("--inner-tol", type=float, default=1e-10,
                        help="fixed normalized KKT linear-system tolerance")
    parser.add_argument("--rho", type=float, default=2.0, help="fixed SQP merit penalty")
    parser.add_argument("--shift", type=float, default=0.0, help="fixed SQP Hessian shift")
    args = parser.parse_args()
    print_results(run_comparison(args.method, args.tol, args.rho, args.shift,
                                 args.inner_tol))


if __name__ == "__main__":
    main()
