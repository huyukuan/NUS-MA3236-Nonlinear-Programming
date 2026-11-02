"""L20: projected dual supergradient ascent with fixed or diminishing steps.

Common test: minimize .5*max(abs(x1)-1, 0)**2 + .5*x2**2,
subject to x1=.25 and x2>=1. All Lagrangian subproblems are solved exactly.
The script compares step rules, prints diagnostics, and writes no files.
"""

import argparse
from dataclasses import dataclass

import numpy as np


COMMON_START = (1.0, 0.0)
OPTIMAL_VALUE = 0.5  # Used only for reporting, never for an algorithmic step.


# Common problem and exact Lagrangian oracle


def objective(x):
    return float(0.5 * max(abs(x[0]) - 1.0, 0.0)**2 + 0.5*x[1]**2)


def residual(x):
    """Signed residuals (g, h), with g=0 and h<=0 required."""
    return np.array([x[0] - 0.25, 1.0 - x[1]])


def dual_value(w):
    lam, nu = w
    return float(-abs(lam) - 0.5*lam**2 - 0.25*lam + nu - 0.5*nu**2)


def lagrangian_oracle(w):
    """Return a global minimizer, dual value, and residual supergradient.

    At lambda=0 every x1 in [-1,1] minimizes the inner problem.
    We consistently choose -1; no minimum-norm supergradient is computed.
    """
    lam, nu = w
    x1 = -np.sign(lam)*(1.0 + abs(lam)) if lam != 0.0 else -1.0
    x = np.array([x1, nu])
    return x, dual_value(w), residual(x)


def project_multipliers(w):
    """Projection onto W=R x R_+: leave lambda free and clip nu."""
    return np.array([w[0], max(w[1], 0.0)])


@dataclass
class Run:
    method: str
    history: list
    best_w: np.ndarray
    average_w: np.ndarray
    average_x: np.ndarray
    next_w: np.ndarray
    step_sum: float


def projected_supergradient(oracle, project, w0, step_rule, iterations=2000,
                           method="projected supergradient"):
    """Basic projected update, with exactly `iterations` oracle calls.

    History contains the evaluated points w^0,...,w^(K-1). The final
    update w^K is `next_w`; it has not been passed to the oracle.
    Averages use alpha_k for k=0,...,K-1. Small steps are not a stop test.
    The caller must ensure finite attained inner minima at every iterate.
    """
    w = np.asarray(w0, dtype=float).copy()
    if (iterations < 1 or int(iterations) != iterations
            or w.ndim != 1 or not np.all(np.isfinite(w))
            or not np.allclose(project(w), w, rtol=0, atol=0)):
        raise ValueError("require a finite feasible w0 and positive integer iterations")
    history, best_value, best_w = [], -np.inf, w.copy()
    weighted_w, weighted_x, step_sum = np.zeros_like(w), None, 0.0
    for k in range(iterations):
        alpha = float(step_rule(k))
        x, value, xi = oracle(w)
        x, xi = np.asarray(x, dtype=float), np.asarray(xi, dtype=float)
        if (alpha <= 0 or not np.isfinite(alpha) or not np.isfinite(value)
                or xi.shape != w.shape or not np.all(np.isfinite(xi))
                or not np.all(np.isfinite(x))):
            raise ValueError("require a positive finite step and a finite oracle result")
        history.append(dict(k=k, w=w.copy(), x=x.copy(), value=float(value),
                            supergradient=xi.copy(), alpha=alpha))
        if value > best_value:
            best_value, best_w = float(value), w.copy()
        if weighted_x is None:
            weighted_x = np.zeros_like(x)
        weighted_w += alpha*w
        weighted_x += alpha*x
        step_sum += alpha
        w = np.asarray(project(w + alpha*xi), dtype=float)
        if not np.all(np.isfinite(w)):
            raise ValueError("nonfinite multiplier update")
    return Run(method, history, best_w, weighted_w/step_sum,
               weighted_x/step_sum, w, step_sum)


def run_comparison(method="all", iterations=2000, constant_step=0.2,
                   scale=0.5, power=0.75, w0=COMMON_START):
    """Compare the same basic algorithm with prescribed step schedules."""
    if (method not in ("all", "constant", "diminishing")
            or not np.all(np.isfinite([constant_step, scale, power]))
            or constant_step <= 0 or scale <= 0 or not 0 < power <= 1):
        raise ValueError("require positive steps and 0<power<=1")
    runs = []
    if method in ("all", "constant"):
        runs.append(projected_supergradient(
            lagrangian_oracle, project_multipliers, w0,
            lambda k: constant_step, iterations, f"constant {constant_step:g}"))
    if method in ("all", "diminishing"):
        for q in ((0.5, 0.75, 1.0) if method == "all" else (power,)):
            runs.append(projected_supergradient(
                lagrangian_oracle, project_multipliers, w0,
                lambda k, q=q: scale/(k+1)**q, iterations, f"power {q:g}"))
    return runs


def diagnostics(run):
    """Known optimum is used only to measure errors on the common test."""
    r_average = residual(run.average_x)
    return dict(last_gap=OPTIMAL_VALUE-run.history[-1]["value"],
                best_gap=OPTIMAL_VALUE-dual_value(run.best_w),
                average_gap=OPTIMAL_VALUE-dual_value(run.average_w),
                average_equality=abs(r_average[0]),
                average_inequality=max(r_average[1], 0.0))


def print_results(runs):
    print("Common test: x*=(0.25,1), w*=(0,1), p*=d*=0.5")
    print("Dual gaps use last evaluated, best, and alpha-weighted-average multipliers.")
    print("Primal residuals use the alpha-weighted-average inner solutions.")
    print("method          calls      last_gap      best_gap       avg_gap"
          "     avg_|g|     avg_[h]+")
    for run in runs:
        d = diagnostics(run)
        print(f"{run.method:15} {len(run.history):5d} "
              f"{d['last_gap']:13.4e} {d['best_gap']:13.4e} "
              f"{d['average_gap']:13.4e} {d['average_equality']:11.4e} "
              f"{d['average_inequality']:11.4e}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("all", "constant", "diminishing"),
                        default="all")
    parser.add_argument("--iterations", type=int, default=2000,
                        help="number of exact inner solves")
    parser.add_argument("--constant-step", type=float, default=0.2)
    parser.add_argument("--scale", type=float, default=0.5,
                        help="coefficient a in alpha_k=a/(k+1)^q")
    parser.add_argument("--power", type=float, default=0.75,
                        help="q when --method diminishing; all compares .5,.75,1")
    args = parser.parse_args()
    print_results(run_comparison(args.method, args.iterations,
                                 args.constant_step, args.scale, args.power))


if __name__ == "__main__":
    main()
