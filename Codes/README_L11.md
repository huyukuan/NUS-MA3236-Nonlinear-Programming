# L11: Conjugate directions and conjugate gradients

## Requirements

Python 3.9 or newer and NumPy are required. Matplotlib is required only for
the experiment figures.

```bash
python3 -m pip install numpy matplotlib
```

## Files and functions

`L11_conjugate_gradient.py` contains the algorithms, test problems,
numerical-result helpers, and four experiment figures.

| Function | Purpose |
|---|---|
| `quadratic_gradient`, `exact_line_step` | Basic operations for a PD quadratic. |
| `steepest_descent` | Exact-line-search steepest descent. |
| `conjugate_direction` | Exact searches along a supplied conjugate basis. |
| `conjugate_gradient` | Linear CG using `g = Qx-c`. |
| `preconditioned_conjugate_gradient` | CG or PCG with a fixed symmetric PD preconditioner solve. |
| `common_example_results` | Run selected methods on the common two-dimensional problem. |
| `pcg_experiment_results` | Return CG and Jacobi-PCG histories and condition numbers. |
| `finite_precision_experiment_results` | Return Hilbert conjugacy defects and scaled-Poisson restart histories. |
| `save_figure`, `save_all_figures` | Save one or all experiment figures. |

| Figure key | Output file | Experiment |
|---|---|---|
| `bases` | `l11_cd_two_bases.pdf` | Two conjugate bases versus exact-search steepest descent. |
| `comparison` | `l11_method_comparison.pdf` | Trajectories and relative objective gaps for all common-example methods. |
| `pcg` | `l11_pcg_comparison.pdf` | CG versus Jacobi PCG. |
| `restart` | `l11_cg_restart_comparison.pdf` | Finite-precision conjugacy loss and residual-refresh restarts. |

## How to run

Run every numerical example, or select one algorithm or experiment:

```bash
python3 L11_conjugate_gradient.py --algorithm all
python3 L11_conjugate_gradient.py --algorithm cg
python3 L11_conjugate_gradient.py --algorithm pcg
```

Generate one experiment figure or all four:

```bash
python3 L11_conjugate_gradient.py --figure restart
python3 L11_conjugate_gradient.py --figure all
```

Figures are written to the current working directory by default. Use
`--output-dir PATH` to choose another directory.

## Reference results

The common problem is `Q = [[5,-4],[-4,5]]`, `c = (0,0)`, and
`x0 = (1,0.8)`. Its eigenvalues are `1` and `9`, so its condition number is
`9`.

| Method | Gap ratio after one update | Updates | Final relative gap |
|---|---:|---:|---:|
| Exact SD | `0.64` | `31` | `9.807971e-7` |
| Conjugate directions A | `0.10` | `2` | numerical zero |
| Conjugate directions B | `0.80` | `2` | numerical zero |
| CG | `0.64` | `2` | numerical zero |

The steepest-descent target is a relative objective gap at most `1e-6`.
The conjugate-direction runs and CG reach the minimizer in two updates; CG and
steepest descent share the first update.

| Scaled 100-dimensional problem | Condition number | Updates to `1e-8` | Final relative `Q`-norm error |
|---|---:|---:|---:|
| CG | `220.9435` | `88` | `9.5330e-9` |
| Jacobi PCG | `2.9981` | `14` | `4.3426e-9` |

For the float32 4-by-4 Hilbert problem, the normalized conjugacy defects for
new-direction indices `1`, `2`, and `3` are `5.5954e-6`, `2.3914e-2`, and
`0.9996`.

| Scaled-Poisson float32 run | Matrix-vector-product budget | Refresh costs | Final relative `Q`-norm error |
|---|---:|---|---:|
| Ordinary CG | `100` | none | `7.8784e-8` |
| Restart every 30 updates | `100` | `31, 62, 93` | `1.0393e-9` |

The restart run is about `75.8` times more accurate at the same counted cost;
each residual refresh is included in the budget.
