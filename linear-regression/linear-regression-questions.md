# Linear Regression Quiz Review

This file collects the linear regression questions we reviewed, with corrected explanations and formulas for future revision.

---

## 1. What is the linear regression hypothesis?

### Question

What does this mean?

```text
h_theta(x) = theta^T x
```

### Answer

The hypothesis is the model's prediction function.

```text
h_theta(x) = theta^T x
```

Expanded:

```text
h_theta(x) = theta_0*x_0 + theta_1*x_1 + theta_2*x_2 + ... + theta_n*x_n
```

Where:

- `x` is the input feature vector for one training example.
- `theta` is the parameter/weight vector the model learns.
- `theta^T x` is the dot product between the parameters and features.
- `h_theta(x)` is the predicted output.

### Example

If:

```text
x = [1, 2104, 5]
theta = [50, 0.1, 10]
```

Then:

```text
h_theta(x) = 50*1 + 0.1*2104 + 10*5
           = 50 + 210.4 + 50
           = 310.4
```

So the model predicts `310.4`.

---

## 2. Why do we add the bias/intercept term?

### Question

Why do we add `x_0 = 1`?

### Answer

We add `x_0 = 1` so the model can learn an intercept/bias term.

Without the bias term:

```text
h_theta(x) = theta_1*x_1
```

The line must pass through the origin `(0, 0)`.

With the bias term:

```text
h_theta(x) = theta_0 + theta_1*x_1
```

The line can shift up or down.

### Meaning of "line passing through the origin"

A line passes through the origin if it goes through point `(0, 0)`.

Example:

```text
y = 3x
```

When `x = 0`, `y = 0`, so it passes through the origin.

But:

```text
y = 3x + 5
```

When `x = 0`, `y = 5`, so it does not pass through the origin.

### Why this matters

Most real datasets do not naturally have output `0` when all features are `0`.

The bias term gives the model flexibility.

---

## 3. Why do we square the error in linear regression?

### Question

Why does the cost function use squared error?

```text
(h_theta(x_i) - y_i)^2
```

### Answer

We square the error for two main reasons.

First, it prevents positive and negative errors from cancelling out.

Example:

```text
error 1 =  5
error 2 = -5
sum = 0
```

Without squaring, the model may look perfect even though it made mistakes.

Second, squaring penalizes large errors more strongly.

Example:

```text
error = 2  -> squared error = 4
error = 10 -> squared error = 100
```

So large mistakes become much more costly.

### Cost function

```text
J(theta) = (1 / 2m) * sum_i=1^m (h_theta(x_i) - y_i)^2
```

Meaning:

- `m` is the number of training examples.
- `h_theta(x_i)` is the prediction for example `i`.
- `y_i` is the true target for example `i`.
- The cost measures average squared prediction error.

---

## 4. Why do we use `1 / 2m` in the cost function?

### Question

Why is the cost function:

```text
J(theta) = (1 / 2m) * sum_i=1^m (h_theta(x_i) - y_i)^2
```

instead of:

```text
J(theta) = (1 / m) * sum_i=1^m (h_theta(x_i) - y_i)^2
```

### Answer

The `1 / m` averages the error over all training examples.

The `1 / 2` is mainly for calculus convenience.

When we take the derivative of a square:

```text
d/dz z^2 = 2z
```

The `1/2` cancels that `2`.

So:

```text
(1 / 2m) * 2 = 1 / m
```

This makes the gradient cleaner.

### Key point

The `1/2` does not change where the minimum is.

It only makes the derivative simpler.

---

## 5. What does this gradient term mean?

### Question

What does this term mean?

```text
(h_theta(x_i) - y_i) * x_j_i
```

### Answer

It means:

- `h_theta(x_i) - y_i` is the prediction error for training example `i`.
- `x_j_i` is feature `j` for that same training example.
- Multiplying them tells gradient descent how much feature `j` contributed to the error.

### Formula

```text
(h_theta(x^(i)) - y^(i)) * x_j^(i)
```

If the feature value is large, that example has a stronger effect on updating `theta_j`.

### Intuition

Gradient descent asks:

```text
How should theta_j change to reduce the prediction error?
```

This term is part of that answer.

---

## 6. What is batch gradient descent?

### Question

What is the batch gradient descent update for linear regression?

### Answer

Batch gradient descent uses all training examples to compute one update.

For each parameter `theta_j`:

```text
theta_j := theta_j - alpha * (1/m) * sum_i=1^m (h_theta(x_i) - y_i) * x_j_i
```

Where:

- `alpha` is the learning rate.
- `m` is the number of training examples.
- The summation uses every training example before updating.

### Why it is called "batch"

It is called batch gradient descent because each update uses the full batch of training data.

Comparison:

- Batch gradient descent: use all examples, then update.
- Stochastic gradient descent: use one example, then update.
- Mini-batch gradient descent: use a small group of examples, then update.

---

## 7. Why does gradient descent subtract the gradient?

### Question

Why is the update:

```text
theta_j := theta_j - alpha * dJ/dtheta_j
```

instead of adding?

### Answer

The gradient points in the direction where the cost increases fastest.

So:

```text
+ gradient
```

moves uphill and increases cost.

But:

```text
- gradient
```

moves downhill and decreases cost.

### Important correction

Gradient descent is not mainly trying to make the parameter values small.

It is trying to make the cost function small.

The parameter may increase or decrease depending on which direction lowers the cost.

### Example

If:

```text
theta = 2
gradient = positive
```

Then:

```text
theta := 2 - alpha * positive
```

`theta` decreases.

If:

```text
theta = 2
gradient = negative
```

Then:

```text
theta := 2 - alpha * negative
```

`theta` increases.

The goal is always to move toward lower cost.

---

## 8. What does the learning rate do?

### Question

What is the purpose of `alpha` in gradient descent?

### Answer

The learning rate controls how big each gradient descent step is.

```text
theta_j := theta_j - alpha * dJ/dtheta_j
```

If `alpha` is too small:

- Gradient descent moves slowly.
- Training may take many iterations.

If `alpha` is too large:

- Gradient descent may overshoot the minimum.
- Cost may oscillate or diverge.

### Intuition

The gradient gives direction.

The learning rate controls step size.

---

## 9. Why does feature scaling help gradient descent?

### Question

Why do we standardize features using mean and standard deviation?

### Answer

Feature scaling helps gradient descent converge faster.

Standardization:

```text
x_j := (x_j - mu_j) / sigma_j
```

Where:

- `mu_j` is the mean of feature `j`.
- `sigma_j` is the standard deviation of feature `j`.

### Why it helps

If one feature has very large values and another has small values, the cost surface can become stretched.

Example:

```text
house_size = 2000
bedrooms = 3
```

Without scaling, the large feature can dominate the gradient update.

With scaling, features are on similar scales, so gradient descent moves more directly toward the minimum.

### Important rule

Compute `mu` and `sigma` from the training set only.

Then use those same values to transform both train and test sets.

This prevents test-set leakage.

---

## 10. What is the normal equation?

### Question

What is the normal equation in linear regression?

### Answer

The normal equation is a closed-form solution for the best `theta`.

```text
theta = (X^T X)^(-1) X^T y
```

It finds the minimum directly instead of using many gradient descent steps.

### Meaning

It comes from taking the derivative of the cost function and setting it equal to zero:

```text
gradient = 0
```

At the minimum, the slope of the cost function is zero.

### Comparison with gradient descent

Gradient descent:

- Iterative.
- Needs learning rate.
- Benefits from feature scaling.
- Better for large feature sets.

Normal equation:

- Direct one-step solution.
- No learning rate.
- No iterations.
- Feature scaling is not required.
- Can be expensive when there are many features.

---

## 11. Why can the normal equation be expensive?

### Question

Why might the normal equation be slow for large datasets?

### Answer

The expensive part is computing the inverse of:

```text
X^T X
```

If there are `n` features, then `X^T X` is an `n x n` matrix.

Matrix inversion costs roughly:

```text
O(n^3)
```

So if the number of features is very large, the normal equation becomes expensive.

### Important distinction

The main issue is the number of features, not only the number of training examples.

Large `n` makes matrix inversion expensive.

---

## 12. What if `X^T X` is not invertible?

### Question

What does it mean if `X^T X` is not invertible, and what can cause it?

### Answer

If `X^T X` is not invertible, it means the matrix has no inverse.

Then this formula cannot be used directly:

```text
theta = (X^T X)^(-1) X^T y
```

Common causes:

- Redundant features.
- One feature is a linear combination of another.
- More features than training examples.

### Example of redundant features

If:

```text
x_2 = 2 * x_1
```

then `x_2` does not add new information. It is dependent on `x_1`.

### Fixes

Remove redundant features, or use the pseudo-inverse:

```python
theta = np.linalg.pinv(X) @ y
```

Later, regularization can also help:

```text
theta = (X^T X + lambda I)^(-1) X^T y
```

---

## 13. Why do we compare train error and test error?

### Question

Why do we compare training error and test error?

### Answer

Training error tells us how well the model fits the data it learned from.

Test error tells us how well the model generalizes to unseen data.

### Good fit

```text
train error: low
test error:  low and close to train error
```

This means the model learned a pattern that generalizes.

### Overfitting

```text
train error: low
test error:  much higher
```

This means the model learned the training data too specifically, including noise or accidental patterns.

### Underfitting

```text
train error: high
test error:  high
```

This means the model is too simple or not trained well enough to capture the real pattern.

### Examples

Likely overfitting:

```text
Train RMSE: 0.20
Test RMSE:  1.50
```

Likely underfitting:

```text
Train RMSE: 2.00
Test RMSE:  2.10
```

Good generalization:

```text
Train RMSE: 0.72
Test RMSE:  0.73
```

---

## 14. What is MSE?

### Question

What does MSE measure?

### Answer

MSE means Mean Squared Error.

```text
MSE = (1/m) * sum_i=1^m (y_hat_i - y_i)^2
```

Where:

- `y_hat_i` is the prediction.
- `y_i` is the true target.
- The error is squared.
- The average is taken across all examples.

### Meaning

MSE reports how wrong the model is on average using squared errors.

Because the errors are squared, large mistakes are penalized heavily.

### Important

MSE is usually used as an evaluation metric.

In Andrew Ng's linear regression derivation, the optimization cost often uses:

```text
J(theta) = (1 / 2m) * sum_i=1^m (h_theta(x_i) - y_i)^2
```

That is closely related to MSE, but with an extra `1/2` for derivative convenience.

---

## 15. What is RMSE?

### Question

What is RMSE, and how is it different from MSE?

### Answer

RMSE means Root Mean Squared Error.

```text
RMSE = sqrt(MSE)
```

If:

```text
MSE = 0.531378
```

Then:

```text
RMSE = sqrt(0.531378) = 0.728957
```

### Why RMSE is useful

MSE is in squared target units.

RMSE is back in the original target units.

So RMSE is easier to interpret as a typical prediction error.

### Correction

RMSE is not the square of MSE.

RMSE is the square root of MSE.

---

## 16. What is R-squared?

### Question

What does `R^2` measure?

### Answer

`R^2` compares the model's error to a simple baseline.

The baseline predicts the average target value for every example.

Formula:

```text
R^2 = 1 - (SSE / SST)
```

Where:

```text
SSE = sum_i=1^m (y_i - y_hat_i)^2
```

SSE is the model's squared error.

```text
SST = sum_i=1^m (y_i - y_bar)^2
```

SST is the baseline squared error.

### Interpretation

```text
R^2 = 1
```

Perfect prediction.

```text
R^2 = 0
```

Same performance as predicting the mean every time.

```text
R^2 < 0
```

Worse than predicting the mean.

```text
R^2 = 0.60
```

The model explains about 60% of the target variance.

The remaining 40% is unexplained by the model. It may come from noise, missing features, nonlinear structure, or limits in the data.

### Important correction

`R^2 = 0.60` does not mean the model is the same as the baseline.

`R^2 = 0` means the model is the same as the baseline.

---

## 17. Gradient descent vs normal equation

### Question

What is the difference between gradient descent and the normal equation?

### Answer

Gradient descent is an iterative optimization method.

It starts with an initial `theta`, computes gradients, and updates many times:

```text
theta := theta - alpha * gradient
```

The normal equation solves for `theta` directly:

```text
theta = (X^T X)^(-1) X^T y
```

### Gradient descent

Use when:

- The dataset has many features.
- Matrix inversion is too expensive.
- You want an iterative optimization method.

Needs:

- Learning rate.
- Number of iterations.
- Often benefits from feature scaling.

### Normal equation

Use when:

- The number of features is not too large.
- You want a direct solution.

Does not need:

- Learning rate.
- Iterations.
- Feature scaling.

---

## 18. Final linear regression checklist

You should be able to explain:

- What `x`, `theta`, and `h_theta(x)` mean.
- Why we add `x_0 = 1`.
- Why squared error is used.
- Why the cost has `1 / 2m`.
- How batch gradient descent updates parameters.
- Why gradient descent subtracts the gradient.
- What the learning rate controls.
- Why feature scaling helps convergence.
- What the normal equation does.
- Why the normal equation can be expensive.
- What to do if `X^T X` is not invertible.
- Why train/test comparison matters.
- Difference between MSE and RMSE.
- How to interpret `R^2`.

