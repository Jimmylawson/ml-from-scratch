# Linear Regression From Scratch (CS229 Style) - Review Notes

## Goal
Build linear regression from scratch using NumPy (no sklearn model training), understand both optimization and closed-form solution, and evaluate correctly on unseen data.

---

## Dataset + Pipeline

### Data
- California Housing
- Features: 8 numeric features
- Target: median house value

### Preprocessing Steps (in order)
1. Shuffle data with fixed seed
2. Split into train/test (80/20)
3. Standardize features using **train** mean/std only
4. Add bias column of ones
5. Train with Gradient Descent (GD)
6. Train with Normal Equation (NE)
7. Compare metrics and convergence

---

## Why Each Step Matters

### Shuffle + Seed
- Shuffling prevents train/test bias from ordered data.
- Seed makes experiments reproducible.

### Train/Test Split
- Train set learns parameters.
- Test set estimates generalization on unseen data.

### Standardization
For each feature:
`x_std = (x - μ) / σ`
- Compute \(\mu, \sigma\) from `X_train` only.
- Apply same \(\mu, \sigma\) to train and test.
- Avoid leakage from test distribution.
- Handle zero-variance feature safely:
`sigma[sigma == 0] = 1.0`

### Bias Term
Add `x0 = 1` to every row so model can learn intercept `θ0`:
`ŷ = θ0 + θ1*x1 + ... + θn*xn`

---

## Shape Conventions (important)

Using 1D target/parameter vectors:
- `X ∈ R^(m×n)`
- `θ ∈ R^n`
- `y ∈ R^m`
- `ŷ = Xθ ∈ R^m`

This shape convention made matrix math clean and avoided many broadcasting bugs.

---

## Core Math Implemented

### Hypothesis
`hθ(x) = θᵀx`
Vectorized:
`ŷ = Xθ`

### Cost Function (for GD optimization)
`J(θ) = (1/(2m)) * Σ(i=1..m) (ŷ(i) - y(i))²`

### Gradient
`∇θ J(θ) = (1/m) * Xᵀ(Xθ - y)`

### Batch GD Update
`θ := θ - α ∇θJ(θ)`

### Normal Equation (closed form)
`θ = (XᵀX)^(-1)Xᵀy`
Implemented numerically as:
`θ = solve(XᵀX, Xᵀy)`
(`np.linalg.solve` preferred over explicit inverse)

---

## Metrics Implemented

### MSE
`MSE = (1/m) * Σ(i=1..m) (ŷ(i) - y(i))²`

### RMSE
`RMSE = √MSE`

### R²
`R² = 1 - (SSE / SST)`
where
`SSE = Σ(y - ŷ)²`
`SST = Σ(y - ȳ)²`

Interpretation learned:
- `R² ≈ 0.60` means model explains ~60% of target variance.
- Remaining variance can be noise, missing features, or nonlinearity.

---

## Results (Current Implementation)

### Gradient Descent
- Train MSE: `0.531378`
- Test MSE: `0.530633`
- Train RMSE: `0.728957`
- Test RMSE: `0.728446`
- Train `R²`: `0.599823`
- Test `R²`: `0.605798`

### Normal Equation
- Train MSE: `0.524368`
- Test MSE: `0.524912`
- Train RMSE: `0.724133`
- Test RMSE: `0.724508`
- Train `R²`: `0.605102`
- Test `R²`: `0.610048`

Conclusion:
- GD and NE are consistent.
- NE is slightly better here.
- Train/test are very close, indicating stable generalization.

---

## Convergence Study (GD -> NE)

Compared GD at different iteration counts to NE using:
`||θ_GD - θ_NE||₂ = √(Σ_j (θ_GD,j - θ_NE,j)²)`

Observed values:
- 100 iters: gap `1.484687`
- 500 iters: gap `0.923622`
- 2000 iters: gap `0.330523`
- 10000 iters: gap `0.006038`

Meaning:
- As iterations increase, GD approaches NE solution.
- Convergence behavior is correct.

Plot saved:
- `gd_convergence.png` (MSE vs iterations)

---

## Functions Built (`model.py`)
- `predict(X, theta)`
- `error(pred, y)`
- `compute_cost(X, y, theta)`
- `compute_gradient(X, y, theta)`
- `fit_gradient_descent(X, y, theta, alpha, num_iters, log_every=None)`
- `normal_equation(X, y)`
- `mse(y, y_pred)`
- `rmse(mse_value)`
- `r2(y_true, y_pred)`

---

## Common Bugs I Hit (and fixed)
- Printing giant arrays each iteration (`print(err)` inside cost loop).
- Shape mismatch from using tuple return instead of unpacking:
  `theta_gd, _ = fit_gradient_descent(...)`
- Wrong `R²` formula (used MSE ratio instead of SSE/SST).
- Accidentally mixing GD predictions and NE predictions in metric calculation.

---

## What I Learned (Journey Summary)
- Why shuffling and seeding are not optional.
- Why standardization changes optimization geometry.
- Why bias term must be explicit in matrix form.
- Difference between training objective `J(θ)` and reporting metric MSE.
- How to verify convergence mathematically, not just visually.
- How to interpret RMSE and `R²` together.

---

## Next (CS229 order)
- Move to Logistic Regression from scratch:
  - sigmoid
  - logistic hypothesis
  - cross-entropy cost
  - gradient descent for classification
