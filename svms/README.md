# Support Vector Machines Revision Notes

This folder implements a linear soft-margin SVM from scratch.

The goal is to learn the core SVM idea:

- build a separating boundary
- prefer a wide margin
- allow some mistakes with soft margin
- train using hinge loss and gradient descent

## 1. Binary Labels

SVM is usually written with labels:

```text
y class:  1
-y class: -1
```

That is why the dataset labels are converted from `0, 1` to `-1, 1`:

```python
y_svm = np.where(y == 0, -1, 1)
```

This is still binary classification. It is not multiclass.

The reason SVM uses `-1` and `1` is because the margin formula becomes clean:

$$
y_i(w^T x_i + b)
$$

If the prediction is correct, this value is positive. If it is wrong, this value is negative.

## 2. Prediction Score

The linear SVM first computes a raw score:

$$
f(x) = w^T x + b
$$

Code:

```python
def prediction_score(X, w, b):
    return X @ w + b
```

Meaning:

- `w` is the weight vector, like `theta` in linear/logistic regression
- `b` is the bias/intercept
- `X @ w` computes the linear score for each row in `X`

For 2D data:

```text
w = [w_1, w_2]
x = [x_1, x_2]
```

So:

$$
w^T x + b
=
w_1x_1 + w_2x_2 + b
$$

## 3. Decision Boundary

The SVM decision boundary is where the score equals zero:

$$
w^T x + b = 0
$$

For 2D plotting:

```text
w_1 x_1 + w_2 x_2 + b = 0
```

Solve for `x_2`:

```text
x_2 = -(w_1 x_1 + b) / w_2
```

Code:

```python
decision_boundary = -(w[0] * x1_vals + b) / w[1]
```

Why solve for `x_2`?

Because `matplotlib` plots x-axis values against y-axis values. We generate many `x_1` values, then calculate the matching `x_2` values.

## 4. Class Prediction

After computing the score:

```text
score = w^T x + b
```

Prediction rule:

```text
if score >= 0: predict 1
if score < 0:  predict -1
```

Code:

```python
def predict_class(X, w, b):
    scores = prediction_score(X, w, b)
    return np.where(scores >= 0, 1, -1)
```

The model predicts using `X`, not `y`.

- `X_test` is the input features
- `y_test` is the answer key used only for evaluation

Correct evaluation:

```python
y_test_pred = predict_class(X_test, w, b)
test_acc = np.mean(y_test_pred == y_test)
```

Do not do this:

```python
test_acc = np.mean(y_test == y_test)
```

That always gives `1.0` because every label equals itself.

## 5. Functional Margin

For one example:

$$
margin_i = y_i(w^T x_i + b)
$$

Code:

```python
margins = y * prediction_score(X, w, b)
```

Interpretation:

- `margin >= 1`: correct and safely outside the margin
- `0 < margin < 1`: correct, but too close to the boundary
- `margin <= 0`: wrong side of the boundary

Examples:

```text
y = 1,  score = 3   -> margin = 3     correct and confident
y = -1, score = -3  -> margin = 3     correct and confident
y = 1,  score = 0.3 -> margin = 0.3   correct but too close
y = 1,  score = -2  -> margin = -2    wrong
```

## 6. Margin Lines

The central decision boundary is:

$$
w^T x + b = 0
$$

The two margin lines are:

$$
w^T x + b = 1
$$

$$
w^T x + b = -1
$$

For 2D plotting:

```python
margin_positive = -(w[0] * x1_vals + b - 1) / w[1]
margin_negative = -(w[0] * x1_vals + b + 1) / w[1]
```

The distance between the two margin lines is:

```text
2 / ||w||
```

Small `||w||` means a wider margin.

## 7. Hard Margin SVM

Hard margin SVM assumes data is perfectly linearly separable.

Constraint:

$$
y_i(w^T x_i + b) \ge 1
$$

Objective:

```text
minimize 1/2 ||w||^2
```

Meaning:

- classify every training example correctly
- maximize the margin
- no violations allowed

This is useful for theory, but too strict for real noisy data.

## 8. Soft Margin SVM

Soft margin SVM allows some examples to violate the margin.

Andrew Ng's notes write this with slack variables:

```text
minimize 1/2 ||w||^2 + C sum_i xi_i
```

subject to:

$$
y_i(w^T x_i + b) \ge 1 - \xi_i
$$

$$
\xi_i \ge 0
$$

The slack variable `xi_i` measures how much example `i` violates the margin.

Instead of explicitly optimizing `xi_i`, your code uses hinge loss.

## 9. Hinge Loss

For one example:

$$
loss_i
=
\max\left(0,\ 1 - y_i(w^T x_i + b)\right)
$$

Code:

```python
losses = np.maximum(0, 1 - margins)
```

Interpretation:

- if `margin >= 1`, loss is `0`
- if `margin < 1`, loss is positive

Examples:

```text
margin = 3    -> max(0, 1 - 3) = 0
margin = 1    -> max(0, 1 - 1) = 0
margin = 0.3  -> max(0, 1 - 0.3) = 0.7
margin = -2   -> max(0, 1 - (-2)) = 3
```

So hinge loss punishes:

- wrong predictions
- correct predictions that are too close to the boundary

## 10. Full SVM Loss

Your loss function:

```python
def soft_margin_svm_loss(X, y, w, b, C):
    margins = y * prediction_score(X, w, b)
    losses = np.maximum(0, 1 - margins)
    return 0.5 * np.dot(w, w) + C * np.mean(losses)
```

Math:

$$
J(w,b)
= \frac{1}{2}\lVert w\rVert^2
+ C \cdot \frac{1}{m}\sum_{i=1}^{m}
\max\left(0,\ 1 - y_i(w^T x_i + b)\right)
$$

Two parts:

$$
\frac{1}{2}\lVert w\rVert^2
$$

This encourages a wide margin.

$$
C \cdot \frac{1}{m}\sum_{i=1}^{m}
\max\left(0,\ 1 - y_i(w^T x_i + b)\right)
$$

This punishes margin violations.

What each term means:

- $J(w,b)$ is the total loss/objective we want to minimize.
- $w$ is the weight vector that controls the direction of the boundary.
- $b$ is the bias/intercept that shifts the boundary.
- $\lVert w\rVert^2$ is the squared length of the weight vector.
- $C$ controls how strongly we punish margin violations.
- $m$ is the number of training examples.
- $y_i$ is the true label for example $i$, either $-1$ or $+1$.
- $x_i$ is the feature vector for example $i$.
- $w^T x_i + b$ is the raw prediction score.
- $\max(0,\ 1 - y_i(w^T x_i + b))$ is the hinge loss.

`C` controls the tradeoff:

- large `C`: punish mistakes more, narrower margin, more fitting
- small `C`: allow more violations, wider margin, more regularization

## 11. Why `np.dot(w, w)`?

The term:

```python
np.dot(w, w)
```

computes:

```text
w^T w = ||w||^2
```

Example:

```text
w = [3, 4]
np.dot(w, w) = 3*3 + 4*4 = 25
||w|| = sqrt(25) = 5
```

## 12. Gradient

Your gradient function:

```python
def soft_margin_svm_gradient(X, y, w, b, C):
    margins = y * prediction_score(X, w, b)
    violating = margins < 1

    dw = w.copy()
    db = 0.0

    if np.any(violating):
        dw -= C * np.mean(y[violating, None] * X[violating], axis=0)
        db -= C * np.mean(y[violating])

    return dw, db
```

The derivative of:

```text
1/2 ||w||^2
```

is:

```text
w
```

That is why:

```python
dw = w.copy()
```

The hinge loss only affects points with:

```text
margin < 1
```

That is why:

```python
violating = margins < 1
```

For violating points:

$$
\frac{\partial J}{\partial w}
= w - C \cdot \frac{1}{m_v}
\sum_{i:\ margin_i < 1} y_i x_i
$$

$$
\frac{\partial J}{\partial b}
= -C \cdot \frac{1}{m_v}
\sum_{i:\ margin_i < 1} y_i
$$

Here $m_v$ is the number of examples currently violating the margin.

Code:

```python
dw -= C * np.mean(y[violating, None] * X[violating], axis=0)
db -= C * np.mean(y[violating])
```

`y[violating, None]` turns labels into a column so NumPy can multiply each selected row in `X`.

Example:

```text
y[violating] = [-1, 1]
X[violating] = [[2, 3],
                [4, 5]]
```

Then:

```text
y[violating, None] = [[-1],
                      [ 1]]
```

and:

```text
y[violating, None] * X[violating]
= [[-2, -3],
   [ 4,  5]]
```

## 13. Fit Loop

Training means repeatedly updating `w` and `b`.

```python
def fit_svm(X, y, alpha, C, num_iters):
    w = np.zeros(X.shape[1])
    b = 0.0
    loss_history = []

    for i in range(num_iters):
        dw, db = soft_margin_svm_gradient(X, y, w, b, C)

        w = w - alpha * dw
        b = b - alpha * db

        loss = soft_margin_svm_loss(X, y, w, b, C)
        loss_history.append(loss)

    return w, b, loss_history
```

The update rule:

```text
w := w - alpha dw
b := b - alpha db
```

`alpha` is the learning rate.

## 14. Plotting

The plot shows:

- training points
- test points
- decision boundary
- margin lines

```python
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap="bwr", label="train")
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap="bwr", marker="x", label="test")
```

`X_train[:, 0]` is feature 1 on the x-axis.

`X_train[:, 1]` is feature 2 on the y-axis.

`c=y_train` colors points by class.

```python
plt.plot(x1_vals, decision_boundary, "k-", label="decision boundary")
plt.plot(x1_vals, margin_positive, "k--", label="margin +1")
plt.plot(x1_vals, margin_negative, "k--", label="margin -1")
```

`"k-"` means black solid line.

`"k--"` means black dashed line.

## 15. Big Picture

SVM is not trying only to classify correctly.

It is trying to classify correctly while keeping a large margin.

The key flow:

```text
score -> margin -> hinge loss -> gradient -> update
```

If you remember only one formula:

$$
J(w,b)
= \frac{1}{2}\lVert w\rVert^2
+ C \cdot \frac{1}{m}\sum_{i=1}^{m}
\max\left(0,\ 1 - y_i(w^T x_i + b)\right)
$$

That is your soft-margin linear SVM.
