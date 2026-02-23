# Derivative-Based Optimization (MATLAB)
Newton-type methods for large-scale nonlinear optimization with exact derivatives, finite-difference approximations, and Hessian-vector products.

This repository implements and benchmarks:
- **Modified Newton** with diagonal shifting (Cholesky-based) for robustness on non-convex problems.
- **Truncated Newton** with **Conjugate Gradient (CG)** using **Hessian-vector products (HVP)** for scalability.
- **Finite-difference (FD)** gradient / Hessian / HVP approximations under different step-size strategies.

The experiments are run on a test function (default: `penalty_1`) across multiple dimensions, including large-scale settings.

---

## What’s inside

### Core optimization methods
- `newton_modified.m`  
  Modified Newton with diagonal shift to enforce positive definiteness.  
  - Dense Hessian case: Cholesky + diagonal shift loop  
  - Large-scale case: supports a **structured Hessian representation** via a struct (`Sherman-Morrison`) to avoid materializing huge matrices.

- `newton_truncated.m`  
  Truncated Newton using **CG** and **Armijo backtracking**, with **Hessian-vector products** `HesVP(x, d)`.

### Test functions
- `penalty_1.m`  
  Returns `[f, g, H]` with:
  - exact function value and gradient
  - Hessian either as a sparse matrix (small n) or as a struct for large n:
    - `H.type = 'Sherman-Morrison'`
    - `H.d, H.u, H.sigma` such that `H ≈ d*I + sigma*(u*u')`

- `penalty_1_hesvp.m`  
  Exact Hessian-vector product for `penalty_1`.

- `broyd_tridia.m` and `broyd_tridia_hesvp.m`  
  Generalized Broyden tridiagonal function (available in code, currently commented out in the main scripts).

### Finite-difference utilities
- `findiff_grad.m`  
  Finite-difference gradient approximation:
  - forward (`'fw'`)
  - central (`'c'`)

- `findiff_hess.m`  
  FD Hessian approximation via forward differences on the gradient.

- `findiff_hess_second_order.m`  
  Second-order FD Hessian using 2-point diagonal and 4-point mixed partials.

- `findiff_hesvp.m`  
  FD Hessian-vector product computed from gradient differences along direction `v`.

### Output helpers
- `parse_output.m`  
  Prints summary of each run (final objective, gradient norm, backtracking count, etc.)

- `calc_conv_rate.m`  
  Estimates an empirical convergence order from the last part of the iterates sequence.

---

## Experiment scripts (wrappers)

The project is organized around three wrapper scripts corresponding to three derivative-availability settings:

### 1) Exact derivatives
**File:** `main_exact.m`

- Gradient: exact (`penalty_1`)
- Hessian: exact (`penalty_1`)
- HVP: exact (`penalty_1_hesvp`)
- Runs both:
  - Modified Newton
  - Truncated Newton (CG + exact HVP)

### 2) Mixed mode (Exact gradient + FD Hessian)
**File:** `main_hessian_fd.m`

- Gradient: exact (`penalty_1`)
- Hessian: finite-difference (`findiff_hess`)
- HVP for TN: exact HVP (`penalty_1_hesvp`)
- Tests multiple FD step sizes and strategies:
  - `h ∈ {1e-4, 1e-8, 1e-12}`
  - `Constant` vs `Relative` (scaled by `|x_i|`)

### 3) Full finite differences
**File:** `main_full_fd.m`

- Gradient: finite-difference (`findiff_grad`, central)
- Hessian: finite-difference (`findiff_hess_second_order`)
- HVP: finite-difference (`findiff_hesvp`)
- Tests multiple FD step sizes and strategies:
  - `h ∈ {1e-4, 1e-8, 1e-12}`
  - `Constant` vs `Relative`

---

## How to run

Open MATLAB in the repository folder and run one of:

```matlab
main_exact
main_hessian_fd
main_full_fd
```

Each script runs experiments for multiple dimensions (default):

```matlab
n = [2 1e3 1e4 1e5];
```

For each `n`, it evaluates:

- A standard starting point  
  ```matlab
  x0 = ones(n,1);   % for penalty_1
  ```

- 5 random perturbations around the standard starting point

---

## Notes on scalability

For very large `n`, explicitly forming and storing an `n × n` Hessian is infeasible.

This repository supports large-scale runs via:

- **Hessian-vector products** (Truncated Newton)

- **Structured Hessian representation for `penalty_1`:**

  ```matlab
  H = d*I + sigma*(u*u');   % rank-1 update
  ```

  handled inside `newton_modified.m` using a Sherman–Morrison-style solve after ensuring positive definiteness.

---

## Parameters

Both methods use Armijo backtracking with:

- `c1` (Armijo parameter)
- `rho` (step reduction)
- `btmax` (maximum backtracking steps)

### Modified Newton also uses:

- `diag_shift`
- `dsmax`  
  (for the diagonal shifting loop)

### Truncated Newton uses:

- `cgmax` (maximum inner CG iterations)

- A forcing term:
  ```matlab
  eta_k = min(0.5, sqrt(norm(g_k)));
  ```
  used for truncation.

The exact values are defined inside each `main_*.m` wrapper.

---

## Expected outputs

For each run, `parse_output.m` prints:

- Initial point type (standard or random)
- First/last components of `x_k`
- Convergence status and iterations
- Empirical convergence rate estimate
- Final `f(x_k)` and `||g(x_k)||`
- Total backtracking steps
- Average CG iterations per outer step (TN only)
- Execution time

---

## Switching test functions

Both `main_exact.m` and `main_hessian_fd.m` include commented code blocks to switch to the generalized Broyden tridiagonal function.

To switch:

1. Uncomment the `broyd_tridia` blocks  
2. Comment out the `penalty_1` blocks  
3. Update the starting point accordingly (currently present in comments)

---

## Author

Taha Atabay Çetin  
MSc Data Science & Engineering — Politecnico di Torino

