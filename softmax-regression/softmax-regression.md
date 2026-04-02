# Softmax Regression From Scratch (CS229 Style) - Revision Notes

## Goal
Build multiclass logistic regression (softmax regression) from scratch using NumPy and evaluate per-class behavior, not just one overall number.

---

## Dataset + Pipeline

### Dataset
- `load_digits` (10 classes: digits 0-9)
- Features: flattened pixel values

### Steps implemented
1. Load `X, y`
2. Shuffle with fixed seed (`42`)
3. Train/test split (`80/20`)
4. Standardize using train mean/std only
5. Add bias column of ones
6. Initialize parameter matrix `Theta` with shape `(n_features, n_classes)`
7. Train with gradient descent on softmax cross-entropy
8. Evaluate with:
- train/test accuracy
- confusion matrix
- per-class accuracy

---

## Why Softmax (not sigmoid)
- Sigmoid logistic regression is for binary class (`0/1`).
- Softmax handles multiclass (`K > 2`) by outputting a probability for each class.

---

## Shape Setup (very important)
- `X`: `(m, n)`
- `Theta`: `(n, K)`
- `logits = X @ Theta`: `(m, K)`
- `probs = softmax(logits)`: `(m, K)`
- `y`: `(m,)` containing class indices in `{0,...,K-1}`

`m` = number of samples  
`n` = number of features (including bias after hstack)  
`K` = number of classes

---

## Core Math Implemented

### 1) Linear scores
`logits = XTheta`

### 2) Softmax probabilities
For sample `i`, class `k`:

`p(i,k) = exp(z(i,k)) / sum_j exp(z(i,j))`

Vectorized stability trick used:
- subtract row max before `exp`:
`logits = logits - max(logits, axis=1, keepdims=True)`

Why:
- prevents overflow in `exp`
- does not change softmax probabilities

### 3) Cross-entropy loss (indexed labels)
`J(Theta) = -(1/m) * sum_i log p(i, y_i)`

Equivalent one-hot form:
`-(1/m) * sum_i sum_k y(i,k) log p(i,k)`

Meaning of this indexing line in code:
`correct_class_probs = probs[np.arange(m), y]`
- `probs` has shape `(m, K)` (one row per sample, one column per class).
- `np.arange(m)` gives row indices `[0, 1, 2, ..., m-1]`.
- `y` gives the true class column for each row.
- Together, it selects `probs[i, y[i]]` for every sample `i`.
- So `correct_class_probs` is the vector of model probabilities assigned to each sample's true class.

Numerical stability:
- clip probabilities:
`probs = clip(probs, eps, 1-eps)`

### 4) Gradient
With one-hot labels:

`grad = (1/m) * X^T (probs - y_onehot)`

where:
- `y_onehot = eye(K)[y]` with shape `(m, K)`
- gradient shape `(n, K)` matches `Theta`

### 5) GD update
`Theta := Theta - alpha * grad`

---

## Functions Built (`model.py`)
- `predict(X, theta)` -> logits (`X @ theta`)
- `softmax(X, theta)`
- `cross_entropy_loss(X, y, theta)`
- `softmax_gradient(X, y, theta)`
- `fit_softmax_gd(X, y, theta, alpha, num_iters, log_every=None)`
- `predict_class(X, theta)` using `argmax(axis=1)`
- `accuracy(X, y, theta)`

---

## Key Bugs I Hit (and fixed)

### Bug 1: `axis 1 is out of bounds`
Cause:
- `Theta` initialized as 1D vector instead of matrix.

Fix:
- `theta = np.zeros((X_train.shape[1], num_classes))`

### Bug 2: `Cannot interpret '10' as a data type`
Cause:
- wrong `np.zeros` call:
`np.zeros(X_train.shape[1], num_classes)`

Fix:
- pass tuple:
`np.zeros((X_train.shape[1], num_classes))`

### Bug 3: Confusion matrix all counts in first column
Cause:
- model was not actually trained (`fit` call commented), so `Theta` stayed all zeros.

Fix:
- run training before predicting:
`theta, cost_history = fit_softmax_gd(...)`

---

## What Confusion Matrix Means (multiclass)
- Matrix shape: `(K, K)` = `(10, 10)`
- Rows = true class
- Columns = predicted class

Interpretation:
- Strong diagonal -> good model
- Off-diagonal -> specific class confusions

Your matrix is mostly diagonal with small off-diagonal entries, so the model is strong.

---

## Per-Class Accuracy
For class `c`:

`acc_c = cm[c,c] / sum_j cm[c,j]`

Meaning:
- Among true class-`c` samples, what fraction were correctly predicted?

Why useful:
- reveals hard classes (even if overall accuracy is high)
- identifies where model confuses similar digits

---

## Training/Convergence Notes
- Loss decreased steadily across iterations (good convergence behavior).
- Final train/test accuracy indicates strong performance with mild expected train-test gap.

---

## Practical Checklist (Done)
- [x] Softmax forward pass
- [x] Numerically stable softmax
- [x] Cross-entropy loss
- [x] One-hot gradient
- [x] Gradient descent trainer
- [x] Accuracy
- [x] Confusion matrix
- [x] Per-class accuracy

---

## Next Improvements (optional)
1. Re-enable `log_every` printing in training to monitor loss live.
2. Plot loss curve from `cost_history`.
3. Add L2 regularization on `Theta` (exclude bias row).
4. Add top confusion pairs report from confusion matrix.
---


## Probability and Interpretation
- 90-100%: Very confident, likely correct
- 70-89%: Confident, usually correct
- 50-69%: Moderate confidence, could be wrong
- Below 50%: Low confidence, likely wrong
---

## Accuracy
Why accuracy matters more:
- Probability: How confident model is per prediction
- Accuracy: How often model is actually correct
- Goal: High accuracy, not just high confidence