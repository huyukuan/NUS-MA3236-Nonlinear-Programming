# L10: Steepest descent

## Requirements

Python 3.9 or newer and NumPy are required. Matplotlib is required only for
the experimental figures.

```bash
python3 -m pip install numpy matplotlib
```

## Files and functions

`L10_steepest_descent.py` contains the algorithms, reproducible examples,
numerical summaries, and experimental plots for L10.

| Function | Purpose |
|---|---|
| `steepest_descent`, `armijo_backtracking` | General line-search steepest descent with Armijo backtracking as the default. |
| `make_quadratic_exact_line_search`, `quadratic_steepest_descent` | Exact line search and the symmetric PD quadratic specialization. |
| `preconditioned_quadratic_trajectory` | Exact-line-search preconditioned SD for a quadratic. |
| `objective`, `gradient`, `Q`, `c` | Hand-worked quadratic test problem. |
| `quartic_objective`, `quartic_gradient` | Smooth nonquadratic Armijo test problem. |
| `quadratic_family`, `quadratic_relative_gap_trajectory` | Rotated quadratics used in the conditioning experiments. |
| `preconditioning_example` | Pure, Jacobi-preconditioned, and ideal preconditioned SD comparison. |
| `flat_phi`, `flat_phi_gradient`, `flat_sublinear_history` | Smooth convex example with step `1/L` and sublinear convergence. |
| `save_figure`, `save_all_figures` | Save one or all retained experimental figures. |

The optional `line_search` callback has signature
`(objective, x, direction, value, gradient) -> alpha`. A history row is
`(iteration, point, gradient_norm, step)`; the terminal row has `step=None`.

| Figure key | Output file | Experiment |
|---|---|---|
| `moderate` | `l10_quadratic_moderate.pdf` | Exact SD for curvature ratio `5`. |
| `flat` | `l10_quadratic_flat.pdf` | Exact SD for curvature ratio `50`. |
| `rates` | `l10_condition_rates.pdf` | Relative-gap histories for both curvature ratios. |
| `preconditioners` | `l10_preconditioner_trajectories.pdf` | Pure, Jacobi-preconditioned, and ideal trajectories. |
| `sublinear` | `l10_sublinear_rate.pdf` | Gap decay and one-step ratios for the flat convex example. |

## How to run

Run one numerical example or the complete numerical summary:

```bash
python3 L10_steepest_descent.py --example armijo
python3 L10_steepest_descent.py --example quadratic-exact
python3 L10_steepest_descent.py --example all
```

Generate one retained experimental figure or all five:

```bash
python3 L10_steepest_descent.py --figure flat
python3 L10_steepest_descent.py --figure all
```

Figures are written to the current working directory by default. Pass
`--output-dir PATH` to use another directory.

## Reference results

| Example | Default settings | Expected output |
|---|---|---|
| General nonquadratic Armijo SD | Initial point `(2,-1)`; gradient tolerance `1e-6` | 4 updates; accepted steps `(0.25,1,1,1)`; final point approximately `(0,1)`. |
| Convex quadratic | Initial point `(0,0)`; gradient tolerance `1e-6` | 41 updates; final point approximately `(0.99999905, 0.99999952)`; gradient norm `9.54e-7`; minimizer `(1,1)`. |
| Rotated quadratic, curvature ratio `5` | `R^T x0=(4,4/5)`; relative gap at most `1e-6` | Exact SD takes 18 updates. |
| Rotated quadratic, curvature ratio `50` | `R^T x0=(4,4/50)`; relative gap at most `1e-6` | Exact SD takes 173 updates. |
| Preconditioning comparison | Curvature ratio `50`, rotation `10` degrees, exact line search, relative gap at most `1e-6` | Pure SD: 173 updates; Jacobi-preconditioned SD: 19 updates with effective condition number `7.48365`; ideal `B=Q`: 1 update. |
| `varphi(x)=exp(-x^2)+x^2-1` | `x0=1`; `L=2.892521`; step `1/L=0.345719` | At `k=10000`, gap `2.612129e-9`, one-step gap ratio `0.999800108`, and `k^2` times the gap `0.261213`. |
