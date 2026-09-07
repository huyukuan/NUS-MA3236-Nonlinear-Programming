# L08: One-dimensional optimization

## Requirements

Python 3.9+, NumPy, and Matplotlib.

```bash
python3 -m pip install numpy matplotlib
```

## Files and functions

File: `l08_one_dim_methods.py`.

| Function | Purpose |
|---|---|
| `objective`, `derivative`, `second_derivative` | Test function and its derivatives. |
| `bisection_stationary` | Bisection applied to the stationarity equation. |
| `newton_stationary` | Newton's method for the stationarity equation. |
| `golden_section` | Derivative-free golden-section search. |
| `run_demo`, `save_comparison_figure` | Run all three methods; save their comparison figure. |

Each algorithm returns a `Result` with `estimate`, `iterations`, `history`, and `evaluations`.

## How to run

Run commands from this directory. All three algorithms:

```bash
python3 l08_one_dim_methods.py
```

This prints the comparison and saves `l08_method_comparison.pdf` in the current
working directory. Use `--figure PATH` to choose another name or directory.

Run one algorithm (one command per method):

```bash
python3 -c "import l08_one_dim_methods as m; print(m.bisection_stationary(m.derivative, -1, 1, 1e-4))"
python3 -c "import l08_one_dim_methods as m; print(m.newton_stationary(m.derivative, m.second_derivative, 0, 1e-8))"
python3 -c "import l08_one_dim_methods as m; print(m.golden_section(m.objective, -1, 1, 1e-4))"
```

## Reference results

For `f(x)=exp(x)+x²-2x`, the reference minimizer is `0.314923057845`.
Bisection and golden section start on `[-1,1]` with location tolerance `1e-4`;
Newton starts at `0` with derivative tolerance `1e-8`.

| Method | Estimate | Reductions / updates | Evaluations |
|---|---:|---:|---|
| Bisection | 0.314880371 | 14 | 16 first derivatives |
| Newton | 0.314923059 | 3 | 4 first derivatives, 3 second derivatives |
| Golden section | 0.314935576 | 20 | 22 function values |
