"""L17: barrier methods, quadratic penalty (QP), and ALM.

Common test: minimize 0.5*((x1-1)^2+(x2-2)^2), x1^2+x2-1<=0.
Basic Newton/Armijo subproblems use a fixed inner tolerance and warm starts.
"""

import argparse
from dataclasses import dataclass
from time import perf_counter

import numpy as np


COMMON_SEED = (1.0, 2.0)


# Common problem and KKT quantities


def objective(x):
    return float(0.5*((x[0]-1.0)**2 + (x[1]-2.0)**2))


def gradient(x):
    return np.asarray(x) - np.array([1.0, 2.0])


def constraint(x):
    return float(x[0]**2 + x[1] - 1.0)


def normal(x):
    return np.array([2.0*x[0], 1.0])


def kkt_residuals(x, multiplier):
    """Stationarity, primal feasibility, dual feasibility, complementarity."""
    h = constraint(x)
    return (float(np.linalg.norm(gradient(x)+multiplier*normal(x), np.inf)),
            max(h, 0.0), max(-multiplier, 0.0), abs(multiplier*h))


# Phase I: find a strictly interior point by a separate optimization


@dataclass
class Initialization:
    x: np.ndarray
    steps: int
    status: str


def find_strict_start(x0, margin=0.5, max_iter=100):
    """SD/Armijo on .5*max(h+margin,0)^2; stop at a strict certificate.

    Preserve already interior inputs. A failed search is not an infeasibility
    certificate for a general nonlinear problem.
    """
    x = np.asarray(x0, dtype=float).copy()
    if (x.shape != (2,) or not np.all(np.isfinite(x))
            or not np.isfinite(margin) or margin <= 0 or max_iter < 0):
        raise ValueError("require finite x0 in R^2, margin>0, max_iter>=0")
    if constraint(x) < 0:
        return Initialization(x, 0, "already_interior")
    for j in range(max_iter):
        violation = max(constraint(x)+margin, 0.0)
        grad = violation*normal(x)
        direction, alpha = -grad, 1.0
        for _ in range(60):
            trial = x+alpha*direction
            trial_violation = max(constraint(trial)+margin, 0.0)
            if (np.all(np.isfinite(trial)) and
                    0.5*trial_violation**2 <= 0.5*violation**2
                    - 1e-4*alpha*float(grad @ grad)):
                break
            alpha *= 0.5
        else:
            return Initialization(x, j, "line_search_failed")
        if np.array_equal(trial, x):
            return Initialization(x, j, "stagnation")
        x = trial
        if constraint(x) <= -margin/2:
            return Initialization(x, j+1, "strictly_feasible")
    return Initialization(x, max_iter, "initialization_limit")


# Unconstrained subproblems and Armijo line search


def subproblem_derivatives(x, parameter, method, reference=0.0):
    h, a = constraint(x), normal(x)
    if method == "barrier":
        if h >= 0:
            raise ValueError("the logarithmic barrier requires h(x)<0")
        multiplier, normal_weight = parameter/(-h), parameter/h**2
    elif method in ("qp", "alm"):
        shifted = (reference if method == "alm" else 0.0)+parameter*h
        multiplier = max(shifted, 0.0)
        # PHR and QP are C1, piecewise C2; choose the inactive side at the kink.
        normal_weight = parameter if shifted > 0 else 0.0
    else:
        raise ValueError("method must be barrier, qp or alm")
    hessian = (np.diag([1.0+2.0*multiplier, 1.0])
               + normal_weight*np.outer(a, a))
    return gradient(x)+multiplier*a, hessian


def objective_change(x, trial, parameter, method, reference=0.0):
    """Stable Armijo value difference for these quadratic f and h."""
    step = trial-x
    change = float(gradient(x) @ step + 0.5*(step @ step))
    h, h_change = constraint(x), float(normal(x) @ step + step[0]**2)
    if method == "barrier":
        if h >= 0 or constraint(trial) >= 0:
            return np.inf
        return change-parameter*np.log1p(h_change/h)
    offset = reference if method == "alm" else 0.0
    old, new = max(offset+parameter*h, 0.0), max(offset+parameter*(h+h_change), 0.0)
    return change+(new-old)*(new+old)/(2*parameter)


@dataclass
class InnerResult:
    x: np.ndarray
    steps: int
    residual: float
    status: str


def newton_subproblem(parameter, x0, method, inner_tol=1e-7, max_iter=100,
                      reference=0.0):
    """Newton/Armijo; barrier trials must pass the nonlinear domain check."""
    x = np.asarray(x0, dtype=float).copy()
    if method not in ("barrier", "qp", "alm"):
        raise ValueError("method must be barrier, qp or alm")
    if (x.shape != (2,) or not np.all(np.isfinite(x))
            or not np.all(np.isfinite([parameter, inner_tol, reference]))
            or reference < 0 or parameter <= 0 or inner_tol <= 0 or max_iter < 0):
        raise ValueError("require finite x0 in R^2, parameter,inner_tol>0, max_iter>=0")
    if method == "barrier" and constraint(x) >= 0:
        raise ValueError("a strictly interior starting point h(x0)<0 is required")
    status = "inner_limit"
    for j in range(max_iter+1):
        grad, hessian = subproblem_derivatives(x, parameter, method, reference)
        residual = float(np.linalg.norm(grad, np.inf))
        if not np.all(np.isfinite(grad)) or not np.all(np.isfinite(hessian)):
            status = "nonfinite"
            break
        if residual <= inner_tol:
            status = "converged"
            break
        if j == max_iter:
            break
        try:
            direction = np.linalg.solve(hessian, -grad)
        except np.linalg.LinAlgError:
            status = "singular_hessian"
            break
        slope = float(grad @ direction)
        if not np.isfinite(slope) or slope >= 0:
            status = "non_descent_direction"
            break
        alpha = 1.0
        linear_change = float(normal(x) @ direction)
        if method == "barrier" and linear_change > 0:
            alpha = min(alpha, 0.99*(-constraint(x))/linear_change)
        for _ in range(60):
            trial = x+alpha*direction
            if np.array_equal(trial, x):
                status = "stagnation"
                break
            # The linear boundary fraction alone is insufficient for nonlinear h.
            if np.all(np.isfinite(trial)) and (method != "barrier" or constraint(trial) < 0):
                change = objective_change(x, trial, parameter, method, reference)
                if np.isfinite(change) and change <= 1e-4*alpha*slope:
                    break
            alpha *= 0.5
        else:
            status = "line_search_failed"
        if status != "inner_limit":
            break
        x = trial
    return InnerResult(x, j, residual, status)


@dataclass
class Run:
    method: str
    x: np.ndarray
    multiplier: float
    history: list
    elapsed: float
    status: str
    initialization: object = None


def _outer(method, x0, parameter0, factor, tol, inner_tol, max_outer):
    if (not np.all(np.isfinite([parameter0, factor, tol, inner_tol]))
            or parameter0 <= 0 or tol <= 0 or inner_tol <= 0 or max_outer < 1
            or (method == "barrier" and not 0 < factor < 1)
            or (method == "qp" and factor <= 1)):
        raise ValueError("require finite valid parameters and max_outer>=1")
    start = perf_counter()
    x, parameter = np.asarray(x0, dtype=float).copy(), float(parameter0)
    history, status, multiplier = [], "outer_limit", 0.0
    for k in range(max_outer):
        reference = multiplier if method == "alm" else 0.0
        inner = newton_subproblem(parameter, x, method, inner_tol, reference=reference)
        x = inner.x
        multiplier = (parameter/(-constraint(x)) if method == "barrier"
                      else max(reference+parameter*constraint(x), 0.0))
        stat, feas, dual, comp = kkt_residuals(x, multiplier)
        row = dict(k=k, parameter=parameter, reference=reference, x=x.copy(),
                   multiplier=multiplier,
                   stationarity=stat, feasibility=feas, dual_feasibility=dual,
                   complementarity=comp, kkt=max(stat, feas, dual, comp),
                   inner_steps=inner.steps, inner_tolerance=inner_tol,
                   inner_residual=inner.residual, inner_status=inner.status)
        history.append(row)
        if inner.status != "converged":
            status = inner.status
            break
        if row["kkt"] <= tol:
            status = "converged"
            break
        parameter *= factor
    return Run(method, x, multiplier, history, perf_counter()-start, status)


def barrier_method(x0=COMMON_SEED, mu0=1.0, reduction=0.2, tol=1e-6,
                   inner_tol=1e-7, max_outer=20):
    """Phase I if needed; warm starts and fixed inner accuracy and mu factor."""
    initialization = find_strict_start(x0)
    if initialization.status not in ("already_interior", "strictly_feasible"):
        return Run("barrier", initialization.x, 0.0, [], 0.0,
                   "initialization_failed", initialization)
    run = _outer("barrier", initialization.x, mu0, reduction, tol, inner_tol, max_outer)
    run.initialization = initialization
    return run


def quadratic_penalty(x0=COMMON_SEED, rho0=1.0, growth=5.0, tol=1e-6,
                      inner_tol=1e-7, max_outer=20):
    """Classical QP: fixed rho growth and nu=rho*max(h(x),0)."""
    return _outer("qp", x0, rho0, growth, tol, inner_tol, max_outer)


def augmented_lagrangian(x0=COMMON_SEED, rho=5.0, tol=1e-6,
                         inner_tol=1e-7, max_outer=30):
    """PHR ALM: fixed rho, fixed inner accuracy, unconditional projected update.

    The subproblem is f+([nu_ref+rho*h]_+^2-nu_ref^2)/(2*rho).
    After each solve set nu=[nu_ref+rho*h(x)]_+, starting from nu=0.
    """
    return _outer("alm", x0, rho, 1.0, tol, inner_tol, max_outer)


def run_comparison(method="all", tol=1e-6, inner_tol=1e-7):
    """Use the same Phase I output as the starting point for all methods."""
    drivers = {"barrier": barrier_method, "qp": quadratic_penalty,
               "alm": augmented_lagrangian}
    if method not in (*drivers, "all"):
        raise ValueError("method must be barrier, qp, alm or all")
    initialization = find_strict_start(COMMON_SEED)
    if constraint(initialization.x) >= 0:
        raise RuntimeError("Phase I did not produce a strictly interior common start")
    names = tuple(drivers) if method == "all" else (method,)
    runs = [drivers[name](initialization.x, tol=tol, inner_tol=inner_tol)
            for name in names]
    for run in runs:
        run.initialization = initialization
    return runs


def print_results(runs):
    initialization = next((run.initialization for run in runs
                           if run.initialization is not None), None)
    if initialization is not None:
        print(f"Phase I: {initialization.steps} SD steps, status={initialization.status}, "
              f"x0={np.array2string(initialization.x, precision=9)}, "
              f"h(x0)={constraint(initialization.x):.9f}")
    print(" method outer Newton    parameter       r_stat       r_feas       r_dual       r_comp        r_KKT status")
    for run in runs:
        if not run.history:
            print(f"{run.method}: {run.status}")
            continue
        last = run.history[-1]
        steps = sum(row["inner_steps"] for row in run.history)
        print(f"{run.method:>7} {len(run.history):5d} {steps:6d} "
              f"{last['parameter']:12.5g} {last['stationarity']:12.3e} "
              f"{last['feasibility']:12.3e} {last['dual_feasibility']:12.3e} "
              f"{last['complementarity']:12.3e} "
              f"{last['kkt']:12.3e} {run.status}")
        print(f"  x={np.array2string(run.x, precision=9)}, nu={run.multiplier:.9f}, "
              f"seconds={run.elapsed:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("barrier", "qp", "alm", "all"), default="all")
    parser.add_argument("--tol", type=float, default=1e-6, help="outer KKT tolerance")
    parser.add_argument("--inner-tol", type=float, default=1e-7,
                        help="fixed inner gradient tolerance")
    args = parser.parse_args()
    print_results(run_comparison(args.method, args.tol, args.inner_tol))


if __name__ == "__main__":
    main()
