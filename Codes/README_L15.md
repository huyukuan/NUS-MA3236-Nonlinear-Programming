# L15: Quadratic penalty and augmented Lagrangian method

## Requirements

Python 3.9 or newer and NumPy.

```bash
python3 -m pip install numpy
```

## Files and functions

File: `l15_penalty_alm.py`.

| Function | Purpose |
|---|---|
| `quadratic_penalty` | QP with a fixed geometric penalty growth factor. |
| `augmented_lagrangian` | ALM with a fixed penalty and an unconditional multiplier update. |
| `augmented_model` | Subproblem objective, gradient, and Hessian. |
| `solve_subproblem`, `armijo_backtracking` | Warm-started SD or safeguarded Newton with Armijo backtracking. |
| `kkt_residuals` | Stationarity and equality-feasibility residuals in the infinity norm. |
| `quadratic_test_problem` | Common quadratic test and its derivatives. |
| `run_comparison`, `print_results` | Run the selected methods and print the comparison. |

`EqualityProblem` supplies `f`, `grad_f`, `hess_f`, `g`, and `jac_g`;
constraint gradients are the columns of `jac_g`. Supply `hess_g` for nonlinear
constraints; omitting it declares affine constraints.
Each algorithm returns a `Run` with the final point, multipliers, status,
elapsed time, and outer history (points, penalties, residuals, and inner counts).

- QP uses `lambda = rho*g(x)` and multiplies `rho` by the same `growth`
  after each nonterminal outer iteration.
- ALM updates `lambda <- lambda + rho*g(x)` after every finite inner result
  and keeps `rho` fixed.
- Both use the same `inner_tol` throughout. Inner solves continue to the
  gradient tolerance; numerical failures stop the run with a failure status.
  `max_outer` limits outer solves. For quadratic models, Armijo uses the exact
  quadratic value change to avoid cancellation near a minimizer.

## How to run

Run commands from this directory. Compare QP and ALM using Newton inner solves:

```bash
python3 l15_penalty_alm.py
```

Select one method, use SD, or compare all four combinations:

```bash
python3 l15_penalty_alm.py --method qp
python3 l15_penalty_alm.py --method alm --solver sd
python3 l15_penalty_alm.py --solver all
```

QP/SD at the default accuracy takes substantially longer than the other runs.
Change the outer and fixed inner tolerances with `--tol` and `--inner-tol`.
The script prints numerical results and writes no files.

## Reference results

Common test: `f(x)=0.5[(x1-2)^2+4(x2-1)^2]`, `g(x)=x1+x2-1`.
Start: `x0=(2,1)`, `lambda0=0`. Solution: `x*=(0.4,0.6)`, `lambda*=1.6`.
Outer KKT tolerance: `1e-6`; fixed inner gradient tolerance: `1e-7`.
QP uses `rho0=1`, `growth=5`; ALM uses the fixed `rho=10`.

| Method / subsolver | Outer solves | Inner steps | Final rho | Final KKT violation |
|---|---:|---:|---:|---:|
| QP / Newton | 10 | 10 | 1953125 | 8.192e-7 |
| ALM / Newton | 6 | 6 | 10 | 3.304e-7 |
| QP / SD | 10 | 3836166 | 1953125 | 8.192e-7 |
| ALM / SD | 6 | 277 | 10 | 3.393e-7 |

A Newton step includes a linear solve, so inner-step counts alone do not
measure total computational cost. Timings depend on the machine.
