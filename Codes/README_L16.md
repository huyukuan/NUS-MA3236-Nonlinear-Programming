# L16: Lagrange--Newton and damped equality SQP

## Requirements and usage

Python 3.9 or newer and NumPy. Run from this directory:

```bash
python3 -m pip install numpy
python3 l16_lagrange_newton_sqp.py
python3 l16_lagrange_newton_sqp.py --method sqp
```

The script prints a comparison and accepted step lengths; it writes no files.
Use `--tol` for the outer KKT tolerance and `--inner-tol` for the fixed
normalized linear-system residual tolerance. SQP also accepts a fixed
`--rho` and a fixed nonnegative `--shift`.

## Implementation

| Function | Purpose |
|---|---|
| `objective`, `gradient`, `constraint`, `normal`, `lagrangian_hessian` | Common test and its exact derivatives. |
| `lagrange_newton_step`, `lagrange_newton` | Solve for `(d, delta_lambda)` and take full primal and multiplier steps. |
| `equality_sqp_step`, `damped_sqp` | Solve for `(d, lambda_hat)` and damp both updates. |
| `merit_backtracking` | Armijo for `f + rho*abs(g)`, with `sigma=1e-4` and halving from one. |
| `run_comparison`, `print_results` | Run the selected methods and print results. |

## Common test

`f(x)=0.5[(x1-1)^2+(x2-2)^2]`, `g(x)=x1^2+x2-1=0`.
Start: `x0=(0.5,0.75)`, `lambda0=0`. The solution is approximately
`x*=(0.312908409,0.902088327)`, `lambda*=1.097911673`.
Defaults: outer tolerance `1e-6`, fixed inner tolerance `1e-10`,
SQP merit penalty `rho=2`, and `shift=0`.

| Method | KKT solves | Accepted alpha | Stationarity | Feasibility |
|---|---:|---|---:|---:|
| LN | 4 | 1, 1, 1, 1 | 7.322e-7 | 8.160e-8 |
| Damped SQP | 3 | 0.5, 1, 1 | 7.082e-8 | 5.670e-8 |

With `shift=0`, the full SQP update equals LN; the line search changes the
paths. These counts depend on the problem and start. For example,
`--method sqp --rho 0.1` reports a non-descent direction without changing rho.
