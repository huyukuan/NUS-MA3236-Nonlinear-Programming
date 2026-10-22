# L17: Barrier methods, quadratic penalty, and ALM

## Requirements

Python 3.9 or newer and NumPy.

```bash
python3 -m pip install numpy
```

## Files and functions

File: `l17_barrier_methods.py`. The problem and its derivatives are defined in
this file; the implementations use its single nonlinear inequality.

| Function | Purpose |
|---|---|
| `barrier_method` | Phase I, fixed inner accuracy, and geometric barrier reduction. |
| `quadratic_penalty` | QP with fixed geometric penalty growth. |
| `augmented_lagrangian` | ALM with a fixed penalty and unconditional projected multiplier updates. |
| `find_strict_start` | Phase I by SD/Armijo to find a strictly interior point. |
| `newton_subproblem` | Newton/Armijo with an actual domain check for barrier trials. |
| `subproblem_derivatives`, `objective_change` | Derivatives and stable Armijo value differences. |
| `kkt_residuals` | Stationarity, primal feasibility, dual feasibility, and complementarity. |
| `run_comparison`, `print_results` | Run the selected methods and print the comparison. |

Each algorithm returns a `Run` with the final point, multiplier, status,
elapsed time, and outer history (points, parameters, residuals, and inner
counts). Phase I is recorded separately; the Newton counts and timings cover
only the main algorithm.

- Phase I minimizes `0.5*max(h+0.5,0)^2` by SD/Armijo and stops once
  `h<=-0.25`. Already interior input is preserved. Failure does not certify
  infeasibility of a general nonlinear problem.
- Barrier uses `f-mu*log(-h)` and `nu=mu/(-h)`, starting at `mu=1` and
  multiplying it by `0.2` after each nonterminal outer iteration.
- QP uses `f+(rho/2)*max(h,0)^2` and `nu=rho*max(h,0)`, starting at `rho=1`
  and multiplying it by `5` after each nonterminal outer iteration.
- ALM uses `f+(max(nu_ref+rho*h,0)^2-nu_ref^2)/(2*rho)`, with fixed `rho=5`
  and `nu0=0`. Every inner solve is followed by `nu=max(nu_ref+rho*h,0)`.
- All methods warm-start Newton/Armijo with the same fixed `inner_tol`,
  Armijo constant `1e-4`, and halving. Barrier caps the first trial by the
  linearized boundary fraction `0.99`, then checks actual `h<0` before any
  logarithm. The test has positive definite Newton matrices; at a QP/ALM
  kink, the inactive-side Hessian is used. Stable differences and `log1p`
  avoid subtracting nearby objective values.
- Stop when `max(r_stat,r_feas,r_dual,r_comp)<=tol`, where
  `r_stat=||grad(f)+nu*grad(h)||_inf`, `r_feas=max(h,0)`,
  `r_dual=max(-nu,0)`, and `r_comp=|nu*h|`. Failures and iteration limits
  are reported explicitly; there are no adaptive parameter rules.

## How to run

Run commands from this directory:

```bash
python3 l17_barrier_methods.py
python3 l17_barrier_methods.py --method barrier
python3 l17_barrier_methods.py --method qp
python3 l17_barrier_methods.py --method alm
```

Use `--tol` for the outer KKT tolerance and `--inner-tol` for the fixed inner
gradient tolerance. The script prints results and creates no plots or files.

## Reference results

Common test: `f(x)=0.5[(x1-1)^2+(x2-2)^2]`, `h(x)=x1^2+x2-1<=0`.
Phase I takes **2 SD steps** from `(1,2)` to the shared start
`x0=(0.21875,0.4375)`, where `h(x0)=-0.5146484375`.
Solution: `x*≈(0.312908409,0.902088327)`, `nu*≈1.097911673`.
Outer KKT tolerance: `1e-6`; fixed inner gradient tolerance: `1e-7`.

| Quantity | Barrier | QP | ALM |
|---|---:|---:|---:|
| Outer solves / Newton steps | 10 / 82 | 10 / 33 | 8 / 19 |
| Final parameter | mu=5.12e-7 | rho=1953125 | rho=5 |
| Stationarity | 9.511e-12 | 2.614e-10 | 6.340e-12 |
| Primal feasibility | 0 | 5.621e-7 | 3.307e-7 |
| Dual feasibility | 0 | 0 | 0 |
| Complementarity | 5.120e-7 | 6.172e-7 | 3.631e-7 |
| KKT violation | 5.120e-7 | 6.172e-7 | 3.631e-7 |

Phase I is excluded from these Newton counts. Barrier stays strictly feasible;
QP and ALM approach the boundary from outside on this test. QP/ALM also accept
infeasible starts when called directly. Counts depend on the problem and
start; timings depend on the machine.
