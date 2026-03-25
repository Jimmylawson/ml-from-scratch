# Logistic Regression From Scratch (CS229 Style) - Revision Notes

## Goal
Build binary logistic regression from scratch with NumPy, train using gradient descent, and evaluate with classification metrics (not just accuracy).

---

## Dataset + Pipeline

### Dataset
- Breast Cancer dataset (`load_breast_cancer`)
- Binary target: `0/1`

### Full Pipeline (what I implemented)
1. Load `X, y`
2. Shuffle with fixed seed
3. Train/test split (80/20)
4. Standardize features using train stats only
5. Add bias column of ones
6. Initialize `theta = 0`
7. Train with logistic GD
8. Evaluate:
- accuracy
- confusion matrix
- precision/recall/F1
- threshold analysis
- convergence curve
- learning curve

---

## Why Each Step Matters

### Shuffle + Seed
- Shuffle makes split fair (avoids order bias).
- Seed makes results reproducible.

### Standardization
`x_std = (x - μ) / σ`
- Speeds up and stabilizes gradient descent.
- Use `μ, σ` from train only.
- Apply same stats to test.
- Zero-variance guard: `sigma[sigma == 0] = 1.0`

### Bias Term
Add `x0 = 1` to each row:
`p(y=1|x) = sigmoid(θ0 + θ1*x1 + ... + θn*xn)`

---

## Shape Convention
- `X`: `(m, n)`
- `theta`: `(n,)`
- `y`: `(m,)`
- `z = X @ theta`: `(m,)`
- `y_hat = sigmoid(z)`: `(m,)`

---

## Core Math Implemented

### Linear part
`z = Xθ`

### Sigmoid
`sigmoid(z) = 1 / (1 + exp(-z))`

### Logistic hypothesis
`y_hat = p(y=1|x;θ) = sigmoid(Xθ)`

### Logistic cost (cross-entropy)
`J(θ) = -(1/m) * Σ [ y*log(y_hat) + (1-y)*log(1-y_hat) ]`

Numerical stability used:
- `eps = 1e-15`
- `y_hat = clip(y_hat, eps, 1-eps)`

### Gradient
`∇θ J(θ) = (1/m) * Xᵀ(y_hat - y)`

### GD update
`θ := θ - α * ∇θJ(θ)`

---

## Functions Built (`model.py`)
- `predict(X, theta)` (linear score)
- `sigmoid(X, theta)`
- `logistic_cost_function(X, theta, y)`
- `logistic_gradient(X, y, theta)`
- `fit_gradient_descent(X, y, theta, num_iters, alpha)`
- `pred_prob(X, theta)`
- `predict_class(X, theta, threshold=0.5)`
- `accuracy(X, y, theta)`
- `precision(TP, FP)`
- `recall(TP, FN)`
- `f1_score(TP, FP, FN)`

---

## Debugging Lessons (important)

### 1) Scalar cost vs vector cost
Bug I hit:
- Cost became an array (huge print) with negative entries.

Cause:
- Wrong parentheses in cost; only one term was inside `np.sum(...)`.

Fix:
- Put both log terms inside one `np.sum(...)`.
- Return one scalar cost per iteration.

### 2) `predict_class` signature mismatch
Bug I hit:
- `accuracy()` failed when `predict_class` required threshold arg.

Fix:
- `predict_class(X, theta, threshold=0.5)` default arg.

### 3) Metric divide-by-zero
Fix in precision/recall/F1:
- Return `0.0` if denominator is zero.

---

## Evaluation Metrics (what they mean)

### Accuracy
`accuracy = correct / total`

### Confusion matrix counts
- `TN`: predicted 0, true 0
- `FP`: predicted 1, true 0
- `FN`: predicted 0, true 1
- `TP`: predicted 1, true 1

### Precision
`precision = TP / (TP + FP)`
"Of predicted positives, how many were truly positive?"

### Recall
`recall = TP / (TP + FN)`
"Of actual positives, how many did we catch?"

### F1
`F1 = 2 * (precision * recall) / (precision + recall)`
Balance between precision and recall.

---

## My Key Results

### Training behavior
- First cost: `0.6729916376642551`
- Last cost: `0.08984339085375558`
- Improvement: `0.5831482468104995`

Meaning:
- Cost decreased strongly.
- Optimization is working.

### Accuracy
- Train accuracy: `0.9824`
- Test accuracy: `1.0`

### Confusion matrix at threshold 0.5
- `TN=42, FP=0`
- `FN=0, TP=72`

### Threshold analysis (precision/recall tradeoff)
- Threshold `0.3`: `Precision=0.96`, `Recall=1.0`, `F1=0.9796`
- Threshold `0.5`: `Precision=1.0`, `Recall=1.0`, `F1=1.0`
- Threshold `0.7`: `Precision=1.0`, `Recall=0.9861`, `F1=0.9930`

Interpretation:
- Lower threshold -> more positives predicted -> usually higher recall, lower precision.
- Higher threshold -> stricter positives -> usually higher precision, lower recall.
- On this split, `0.5` is best overall.

---

## Plots Completed

### 1) Convergence plot
- `cost_history` vs iterations
- Smooth decreasing curve = stable GD

Saved file:
- `logistic_gd_convergence.png`

### 2) Learning curve
- Train error and test error vs training set size
- Error metric used: `1 - accuracy`

Reading rule:
- Underfitting: both errors high
- Overfitting: train low, test high
- Good fit: both low and close

My result:
- Both train/test errors low and close -> good generalization on this split

Saved file:
- `logistic_learning_curve.png`

---

## Quick Checklist (Done)
- [x] Logistic cost + gradient from scratch
- [x] GD training loop
- [x] Accuracy
- [x] Confusion matrix
- [x] Precision/Recall/F1
- [x] Threshold analysis
- [x] Convergence plot
- [x] Learning curve

---

## Next (if continuing logistic)
1. Add L2-regularized logistic regression
2. Add validation split for tuning (`alpha`, `num_iters`, `lambda`)
3. Add ROC curve + AUC
4. Final short report with assumptions + limitations
