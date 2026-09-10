# L09: Multivariate Newton methods

## Requirements

Python 3.9+ and NumPy. Matplotlib is needed only for plots.

```bash
python3 -m pip install numpy matplotlib
```

## Files and functions

File: `l09_newton_methods.py`.

| Function | Purpose |
|---|---|
| `objective`, `gradient`, `hessian` | Common quartic and its derivatives. |
| `pure_newton` | Newton with full steps. |
| `shifted_newton`, `shifted_direction` | Full steps using a positive-definite shifted matrix. |
| `newton_armijo`, `descent_shifted_direction` | Accept a descending raw/shifted direction, then apply Armijo backtracking; positive definiteness is not required. |
| `run_comparison`, `print_results` | Run all three methods; print the three-method comparison. |
| `run_saddle_example`, `run_shifting_benefit`, `run_divergence_example` | Double-well and scalar examples. |
| `save_*`, `print_iteration_tables` | Save figures; print detailed iteration data. |

The three algorithms return a `Result` containing iterates, objective values,
gradient norms, step lengths, shifts, and evaluation/solve counts.

## How to run

Run commands from this directory. All three algorithms and the double-well example:

```bash
python3 l09_newton_methods.py
```

Run one algorithm and print its final point, update count, and gradient norm:

```bash
python3 -c "import l09_newton_methods as m; r=m.pure_newton(m.COMMON_START); print(r.points[-1], r.updates, r.grad_norms[-1])"
python3 -c "import l09_newton_methods as m; r=m.shifted_newton(m.COMMON_START); print(r.points[-1], r.updates, r.grad_norms[-1])"
python3 -c "import l09_newton_methods as m; r=m.newton_armijo(m.COMMON_START); print(r.points[-1], r.updates, r.grad_norms[-1])"
```

Include the second starting point, scalar example, and detailed tables:

```bash
python3 l09_newton_methods.py --compare-start 0 0.15 --divergence-demo --details
```

Optional figures (see `--help` for other figure options):

```bash
python3 l09_newton_methods.py --plot l09_comparison.pdf --trajectory-plot l09_trajectories.pdf --start-comparison-plot l09_starts.pdf
```

These figures are saved in the current working directory; relative paths are
resolved from there. Without figure options, no files are written. Existing
output files are overwritten; figures use the fixed lecture examples.

## Reference results

Common quartic: `f(x1,x2)=(x1²+x2-1)²+(x1+x2²-1)²`.
Gradient tolerance: `1e-8`. Entries below count updates.

| Starting point | Pure Newton | Shifted Newton | Newton–Armijo |
|---|---:|---:|---:|
| (1.23, 0.51) | 16 | 8 | 6 |
| (0, 0.15) | 6 | 9 | 9 |

The first start leads all methods to `(1,0)`. The second leads respectively to
`(-1.618034,-1.618034)`, `(0,1)`, and `(0.618034,0.618034)`; all attain `f≈0`.
For the second start, Newton–Armijo first accepts `tau=1.6`, `alpha=0.25`;
its 15 linear solves include rejected non-descending candidates.

- Double well, start `(0.4,-0.3)`: pure Newton reaches the saddle in 2 updates;
  shifted Newton reaches `(1,1)/sqrt(2)` in 8.
- Scalar `sqrt(1+x²)`, start `2`: pure Newton diverges; Newton–Armijo reaches
  approximately zero in 4 updates.
