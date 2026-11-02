# L20: Projected dual supergradient ascent

## Requirements and usage

Python 3.9 or newer and NumPy. Run from this directory:

```bash
python3 -m pip install numpy
python3 l20_dual_subgradient.py
python3 l20_dual_subgradient.py --method diminishing --power 0.75
python3 l20_dual_subgradient.py --method constant --constant-step 0.1
```

The script prints a comparison and writes no files. All runs use the same
initial multipliers and number of exact inner solves. Change that number
with `--iterations`; `--scale` changes the coefficient of a diminishing
step. There is no line search or adaptive rule.

## Implementation

| Function | Purpose |
|---|---|
| `objective`, `residual`, `dual_value` | Common primal problem and explicit dual value. |
| `lagrangian_oracle` | Return one exact global inner minimizer and its residual supergradient. |
| `project_multipliers` | Leave the equality multiplier free and clip the inequality multiplier at zero. |
| `projected_supergradient` | Basic update `w <- project(w + alpha*xi)` with a prescribed step rule. |
| `run_comparison`, `diagnostics`, `print_results` | Run the comparison and report dual gaps and averaged primal feasibility. |

The driver accepts an oracle, a projection, an initial multiplier, and a
step schedule. It assumes a finite attained inner infimum at every iterate.
Each `Run` contains the evaluated history, best multiplier, step-weighted
multiplier and primal averages, and `next_w`. With `K` oracle calls, the
history contains `w^0,...,w^(K-1)`; `next_w` is the unevaluated update `w^K`.
Both averages use weights `alpha_k`, for `k=0,...,K-1`.

## Common test

Minimize `f(x)=0.5*max(|x1|-1,0)^2+0.5*x2^2`, subject to
`g(x)=x1-0.25=0` and `h(x)=1-x2<=0`.
The objective is convex and continuously differentiable, but the dual

`theta(lambda,nu)=-|lambda|-lambda^2/2-0.25*lambda+nu-nu^2/2`

has a kink at `lambda=0`. The solutions are `x*=(0.25,1)` and
`w*=(0,1)`, with `p*=d*=0.5`. At `lambda=0`, the oracle always chooses
`x1=-1` from the interval of inner minimizers; it does not compute a
minimum-norm supergradient.

Defaults: `w0=(1,0)`, `K=2000`; compare constant `alpha_k=0.2` with
`alpha_k=0.5/(k+1)^q`, for `q=0.5,0.75,1`.
The known optimal value is used only for diagnostics, never in an update
or stopping decision. Small multiplier steps alone are not a stop test.

| Step rule | Last dual gap | Best dual gap | Averaged dual gap | Averaged primal `abs(g)` | Averaged primal `[h]+` |
|---|---:|---:|---:|---:|---:|
| Constant 0.2 | 3.8389e-2 | 2.0376e-2 | 3.7902e-2 | 3.0643e-3 | 2.5000e-3 |
| Power 0.5 | 2.5230e-3 | 3.4976e-6 | 1.9500e-3 | 2.2614e-2 | 2.2729e-2 |
| Power 0.75 | 5.1405e-4 | 7.6556e-8 | 5.0039e-2 | 8.5751e-2 | 8.5800e-2 |
| Power 1 | 1.9598e-4 | 1.0229e-4 | 1.7488e-1 | 2.4454e-1 | 2.4146e-1 |

The averaged dual gap is `d*-theta(average_w)`, not the average of the
individual gaps. These are finite-run results: constant steps can leave
persistent dual oscillations even when their primal averages happen to
have small residuals. Faster step decay can also retain more influence
from the initial iterates in the weighted averages.

For `q=0.5`, the last evaluated multiplier is approximately
`(-0.00335645,1)` and its weighted average is `(0.00135262,0.97727106)`.
The raw inner `x1` values still switch near `-1` and `1`; approaching an
optimal multiplier does not make each selected primal point feasible.
