"""L15: quadratic penalty (QP) and augmented Lagrangian method (ALM).

Compare the traditional outer algorithms on a common equality-constrained
quadratic, using warm-started SD or Newton with Armijo backtracking.
"""

import argparse
from dataclasses import dataclass
from time import perf_counter

import numpy as np


# Problem data and results


@dataclass
class EqualityProblem:
    """jac_g(x) has constraint gradients as columns: shape (n,m).

    hess_g(x), if supplied, has shape (m,n,n); omitted means affine constraints.
    quadratic_objective explicitly certifies a quadratic f. With affine g,
    Armijo can evaluate objective changes without subtracting nearby values.
    """
    f: object
    grad_f: object
    hess_f: object
    g: object
    jac_g: object
    hess_g: object = None
    quadratic_objective: bool = False


@dataclass
class InnerResult:
    x: np.ndarray
    iterations: int
    residual: float
    status: str


@dataclass
class Run:
    method: str
    solver: str
    x: np.ndarray
    multipliers: np.ndarray
    history: list
    elapsed: float
    status: str


def inf_norm(vector):
    return (float(np.linalg.norm(vector, ord=np.inf))
            if np.all(np.isfinite(vector)) else float("inf"))


def kkt_residuals(problem, x, multipliers):
    """Return equality KKT stationarity and primal-feasibility violations."""
    return (inf_norm(problem.grad_f(x) + problem.jac_g(x) @ multipliers),
            inf_norm(problem.g(x)))


# Inner solves


def augmented_model(problem, rho, multipliers):
    """Subproblem objective, gradient and exact Hessian, including curvature."""
    if not np.isfinite(rho) or rho <= 0:
        raise ValueError("rho must be finite and positive")
    multipliers = np.asarray(multipliers, dtype=float).copy()

    def value(x):
        constraints = problem.g(x)
        return float(problem.f(x) + multipliers @ constraints
                     + 0.5 * rho * constraints @ constraints)

    def gradient(x):
        return (problem.grad_f(x)
                + problem.jac_g(x) @ (multipliers + rho * problem.g(x)))

    def hessian(x):
        jacobian = problem.jac_g(x)
        result = problem.hess_f(x) + rho * jacobian @ jacobian.T
        if problem.hess_g is not None:
            result = result + np.einsum(
                "i,ijk->jk", multipliers + rho * problem.g(x),
                problem.hess_g(x))
        return result

    return value, gradient, hessian


def descent_shifted_direction(hessian, gradient, max_trials=30):
    """Solve (H+tau I)d=-gradient; safeguard descent, not positive definiteness."""
    identity = np.eye(gradient.size)
    shift = 0.0
    for _ in range(max_trials):
        try:
            direction = np.linalg.solve(hessian + shift * identity, -gradient)
        except np.linalg.LinAlgError:
            direction = np.full_like(gradient, np.nan)
        if np.all(np.isfinite(direction)) and gradient @ direction < 0:
            return direction
        shift = (max(1e-6, 1e-3 * np.linalg.norm(hessian, np.inf))
                 if shift == 0 else 10 * shift)
    return -gradient


def armijo_backtracking(value, x, direction, gradient, sigma=1e-4, beta=0.5,
                        quadratic_hessian=None):
    """Backtrack from one; use an exact change for certified quadratic models.

    For Q(x+s)-Q(x)=gradient@s+0.5*s@H@s, avoid cancellation between
    nearly equal absolute objective values. The Armijo rule is unchanged.
    """
    slope = float(gradient @ direction)
    if slope >= 0:
        raise ValueError("Armijo backtracking needs a descent direction")
    initial_value, alpha = value(x), 1.0
    for _ in range(60):
        trial = x + alpha * direction
        trial_value = value(trial)
        if quadratic_hessian is None:
            sufficient_decrease = trial_value <= initial_value + sigma * alpha * slope
        else:
            step = trial - x
            decrease = float(gradient @ step
                             + 0.5 * step @ quadratic_hessian @ step)
            sufficient_decrease = (np.isfinite(decrease)
                                   and decrease <= sigma * alpha * slope)
        if np.isfinite(trial_value) and sufficient_decrease:
            return alpha
        alpha *= beta
    raise RuntimeError("Armijo backtracking failed")


def solve_subproblem(problem, rho, multipliers, x0, solver="newton",
                     tol=1e-7):
    """Warm-started SD/Newton until inner accuracy, without a normal step cap.

    The same requested tolerance is used at every outer iteration. Numerical
    breakdown and stagnation are reported; globality is not claimed.
    """
    value, gradient, hessian = augmented_model(problem, rho, multipliers)
    x = np.asarray(x0, dtype=float).copy()
    if x.ndim != 1 or x.size == 0 or not np.isfinite(tol) or tol <= 0:
        raise ValueError("require nonempty vector x0 and finite tol>0")
    if not np.all(np.isfinite(x)):
        return InnerResult(x, 0, float("inf"), "nonfinite")
    if solver not in ("sd", "newton"):
        raise ValueError("solver must be sd or newton")
    quadratic_hessian = (hessian(x) if problem.quadratic_objective
                         and problem.hess_g is None else None)
    j = 0
    while True:
        grad = gradient(x)
        residual = inf_norm(grad)
        if not np.isfinite(residual) or not np.isfinite(value(x)):
            return InnerResult(x, j, float("inf"), "nonfinite")
        if residual <= tol:
            return InnerResult(x, j, residual, "converged")
        if solver == "newton":
            matrix = hessian(x)
            if not np.all(np.isfinite(matrix)):
                return InnerResult(x, j, float("inf"), "nonfinite")
            direction = descent_shifted_direction(matrix, grad)
        else:
            direction = -grad
        try:
            alpha = armijo_backtracking(value, x, direction, grad,
                                        quadratic_hessian=quadratic_hessian)
        except RuntimeError:
            return InnerResult(x, j, residual, "line_search_failed")
        trial = x + alpha * direction
        if np.array_equal(trial, x):
            return InnerResult(x, j, residual, "stagnation")
        x = trial
        j += 1


# QP and ALM


def _row(k, rho, before, current, inner, problem, omega):
    stat, feas = kkt_residuals(problem, inner.x, current)
    return dict(k=k, rho=rho, lambda_before=before.copy(),
                lambda_current=current.copy(),
                x=inner.x.copy(), stationarity=stat, feasibility=feas,
                kkt=max(stat, feas), inner_iterations=inner.iterations,
                inner_status=inner.status, inner_tolerance=omega)


def quadratic_penalty(problem, x0, solver="newton", tol=1e-6, rho0=1.0,
                      growth=5.0, max_outer=15, inner_tol=1e-7):
    """Traditional QP: fixed inner accuracy and geometric penalty growth.

    Estimate lambda=rho*g after every finite inner result. Each completed,
    nonterminal outer iteration multiplies rho by the same fixed growth factor.
    Numerical inner failures stop the run and are reported without a KKT claim.
    """
    if (not np.all(np.isfinite([rho0, growth, tol, inner_tol]))
            or rho0 <= 0 or growth <= 1 or tol <= 0 or inner_tol <= 0
            or max_outer < 1):
        raise ValueError("require finite rho0>0, growth>1, tol>0, inner_tol>0 and max_outer>=1")
    start = perf_counter()
    x = np.asarray(x0, dtype=float).copy()
    zeros = np.zeros_like(problem.g(x))
    multipliers, rho = zeros.copy(), float(rho0)
    history, status = [], "outer_limit"
    for k in range(max_outer):
        before = multipliers.copy()
        inner = solve_subproblem(problem, rho, zeros, x, solver, inner_tol)
        x = inner.x
        if not np.isfinite(inner.residual) or inner.status == "nonfinite":
            history.append(_row(k, rho, before, before, inner, problem, inner_tol))
            status = "nonfinite"
            break
        multipliers = rho * problem.g(x)
        row = _row(k, rho, before, multipliers, inner, problem, inner_tol)
        history.append(row)
        if inner.residual > inner_tol:
            status = ("inner_accuracy" if inner.status == "converged"
                      else inner.status)
            break
        if row["kkt"] <= tol:
            status = "converged"
            break
        rho *= growth
    return Run("QP", solver, x, multipliers, history, perf_counter() - start, status)


def augmented_lagrangian(problem, x0, solver="newton", tol=1e-6, rho0=10.0,
                         max_outer=20, multipliers0=None, inner_tol=1e-7):
    """Traditional ALM: fixed rho, fixed accuracy, unconditional multiplier update.

    After every finite inner result use lambda <- lambda+rho*g(x). There is no
    multiplier acceptance test, tolerance schedule, or penalty adaptation.
    Numerical inner failures stop the run and are reported without a KKT claim.
    """
    if (not np.all(np.isfinite([rho0, tol, inner_tol]))
            or rho0 <= 0 or tol <= 0 or inner_tol <= 0 or max_outer < 1):
        raise ValueError("require finite rho0>0, tol>0, inner_tol>0 and max_outer>=1")
    start = perf_counter()
    x = np.asarray(x0, dtype=float).copy()
    multipliers = (np.zeros_like(problem.g(x)) if multipliers0 is None
                   else np.asarray(multipliers0, dtype=float).copy())
    rho = float(rho0)
    history, status = [], "outer_limit"
    for k in range(max_outer):
        before = multipliers.copy()
        inner = solve_subproblem(problem, rho, before, x, solver, inner_tol)
        x = inner.x
        if not np.isfinite(inner.residual) or inner.status == "nonfinite":
            history.append(_row(k, rho, before, before, inner, problem, inner_tol))
            status = "nonfinite"
            break
        multipliers = before + rho * problem.g(x)
        row = _row(k, rho, before, multipliers, inner, problem, inner_tol)
        history.append(row)
        if inner.residual > inner_tol:
            status = ("inner_accuracy" if inner.status == "converged"
                      else inner.status)
            break
        if row["kkt"] <= tol:
            status = "converged"
            break
    return Run("ALM", solver, x, multipliers, history, perf_counter() - start, status)


# Common test and comparison


def quadratic_test_problem():
    """x*=(0.4,0.6), lambda*=1.6, f*=1.6 for the common two-dimensional test."""
    q, center, a = np.diag([1.0, 4.0]), np.array([2.0, 1.0]), np.ones((2, 1))

    def f(x):
        return float(0.5 * (x - center) @ q @ (x - center))

    return EqualityProblem(f, lambda x: q @ (x - center), lambda x: q,
                           lambda x: np.array([x.sum() - 1]), lambda x: a,
                           quadratic_objective=True)


def run_comparison(method="all", solver="newton", tol=1e-6, inner_tol=1e-7):
    """Run QP and/or ALM on the common quadratic with the selected inner solver."""
    if method not in ("qp", "alm", "all"):
        raise ValueError("method must be qp, alm or all")
    if solver not in ("sd", "newton", "all"):
        raise ValueError("solver must be sd, newton or all")
    problem, x0 = quadratic_test_problem(), np.array([2.0, 1.0])
    solvers = ("sd", "newton") if solver == "all" else (solver,)
    drivers = {"qp": quadratic_penalty, "alm": augmented_lagrangian}
    methods = ("qp", "alm") if method == "all" else (method,)
    return [drivers[name](problem, x0, solver=inner_solver,
                          tol=tol, inner_tol=inner_tol)
            for name in methods for inner_solver in solvers]


def print_results(runs):
    """Print iteration counts, final residuals, points, and multipliers."""
    print("method/solver outer    inner   seconds        rho       r_stat        r_KKT status")
    for run in runs:
        last = run.history[-1]
        total = sum(row["inner_iterations"] for row in run.history)
        print(f"{run.method:>3}/{run.solver:<6} {len(run.history):5d} {total:8d} "
              f"{run.elapsed:9.4f} {last['rho']:10.8g} "
              f"{last['stationarity']:12.3e} {last['kkt']:12.3e} {run.status}")
        print(f"  x={np.array2string(run.x, precision=8)}, "
              f"lambda={np.array2string(run.multipliers, precision=8)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("qp", "alm", "all"), default="all")
    parser.add_argument("--solver", choices=("sd", "newton", "all"), default="newton")
    parser.add_argument("--tol", type=float, default=1e-6, help="outer KKT tolerance")
    parser.add_argument("--inner-tol", type=float, default=1e-7,
                        help="fixed inner gradient tolerance")
    args = parser.parse_args()
    print_results(run_comparison(args.method, args.solver, args.tol, args.inner_tol))


if __name__ == "__main__":
    main()
