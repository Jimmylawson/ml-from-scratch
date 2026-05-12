# Linear Regression Quiz Review

This file collects the linear regression questions we reviewed, with corrected explanations, clean formulas, examples, and implementation notes.

---

## 1. Linear Regression Hypothesis

### Question

What does the linear regression hypothesis mean?

$$
h_\theta(x) = \theta^T x
$$

### Answer

The hypothesis is the model's prediction function.

It takes the input features and combines them with the learned parameters.

Expanded:

$$
h_\theta(x) = \theta_0x_0 + \theta_1x_1 + \theta_2x_2 + \cdots + \theta_nx_n
$$

Where:

- \(x\) is the input feature vector.
- \(\theta\) is the parameter vector the model learns.
- \(\theta^T x\) is the dot product between parameters and features.
- \(h_\theta(x)\) is the predicted value.

### Example

If:

```text
x = [1, 2104, 5]
theta = [50, 0.1, 10]
```

Then:

$$
h_\theta(x) = 50(1) + 0.1(2104) + 10(5)
$$

$$
h_\theta(x) = 50 + 210.4 + 50 = 310.4
$$

So the model predicts `310.4`.

---

## 2. Bias / Intercept Term

### Question

Why do we add \(x_0 = 1\)?

### Answer

We add \(x_0 = 1\) so the model can learn an intercept term.

Without bias:

$$
h_\theta(x) = \theta_1x_1
$$

The line must pass through the origin \((0,0)\).

With bias:

$$
h_\theta(x) = \theta_0 + \theta_1x_1
$$

The line can shift up or down.

### Meaning of Passing Through the Origin

A line passes through the origin if it goes through \((0,0)\).

Example:

$$
y = 3x
$$

When \(x=0\), \(y=0\).

But:

$$
y = 3x + 5
$$

When \(x=0\), \(y=5\). So it does not pass through the origin.

### Why This Matters

Most real datasets do not naturally have output `0` when all features are `0`.

The bias term gives the model flexibility.

---

## 3. Squared Error

### Question

Why does linear regression square the error?

$$
(h_\theta(x^{(i)}) - y^{(i)})^2
$$

### Answer

We square the error for two main reasons.

First, it prevents positive and negative errors from cancelling out.

```text
error 1 =  5
error 2 = -5
sum     =  0
```

Without squaring, the model could look perfect even though it made mistakes.

Second, squaring penalizes large errors more strongly.

```text
error = 2  -> squared error = 4
error = 10 -> squared error = 100
```

So large mistakes become much more costly.

---

## 4. Cost Function

### Question

What is the linear regression cost function?

### Answer

Andrew Ng's linear regression cost is:

$$
J(\theta) = \frac{1}{2m}\sum_{i=1}^{m}\left(h_\theta(x^{(i)}) - y^{(i)}\right)^2
$$

Where:

- \(m\) is the number of training examples.
- \(h_\theta(x^{(i)})\) is the prediction for example \(i\).
- \(y^{(i)}\) is the true target for example \(i\).
- The cost measures average squared prediction error.

### Why \(1 / 2m\)?

The \(1/m\) averages over the training set.

The \(1/2\) is mainly for calculus convenience.

Because:

$$
\frac{d}{dz}z^2 = 2z
$$

The \(1/2\) cancels the `2` when differentiating.

$$
\frac{1}{2m} \cdot 2 = \frac{1}{m}
$$

The \(1/2\) does not change where the minimum is. It only makes the gradient cleaner.

---

## 5. Gradient Descent Update

### Question

What is batch gradient descent for linear regression?

### Answer

Gradient descent updates each parameter by moving opposite the gradient.

For parameter \(\theta_j\):

$$
\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j}J(\theta)
$$

For linear regression, the gradient is:

$$
\frac{\partial}{\partial \theta_j}J(\theta)
= \frac{1}{m}\sum_{i=1}^{m}\left(h_\theta(x^{(i)}) - y^{(i)}\right)x_j^{(i)}
$$

So the update becomes:

$$
\theta_j := \theta_j - \alpha \frac{1}{m}\sum_{i=1}^{m}\left(h_\theta(x^{(i)}) - y^{(i)}\right)x_j^{(i)}
$$

### Meaning of the Gradient Term

$$
\left(h_\theta(x^{(i)}) - y^{(i)}\right)x_j^{(i)}
$$

This says:

- \(h_\theta(x^{(i)}) - y^{(i)}\) is the prediction error.
- \(x_j^{(i)}\) tells how much feature \(j\) contributed.
- Multiplying them tells how feature \(j\) should influence the parameter update.

If the model is overpredicting and feature \(j\) is large, the gradient pushes \(\theta_j\) downward.

---

## 6. Learning Rate

### Question

What does the learning rate \(\alpha\) do?

### Answer

The learning rate controls the step size of gradient descent.

Update rule:

$$
\theta := \theta - \alpha \nabla_\theta J(\theta)
$$

If \(\alpha\) is too small:

```text
Gradient descent is slow.
```

If \(\alpha\) is too large:

```text
Gradient descent may overshoot the minimum or diverge.
```

Good learning rate:

```text
Cost decreases steadily over iterations.
```

---

## 7. Convergence and Divergence

### Question

What do convergence and divergence mean?

### Answer

Convergence means the optimization is moving toward a stable minimum.

In practice:

```text
cost goes down and eventually flattens
```

Divergence means the optimization is moving away from the minimum.

In practice:

```text
cost increases, explodes, or becomes NaN
```

### Why This Matters

Your convergence plot should show cost decreasing over time.

A good curve looks like:

```text
high cost -> lower cost -> flat
```

---

## 8. Feature Scaling

### Question

Why does feature scaling help gradient descent?

### Answer

Feature scaling makes features live on similar ranges.

Standardization:

$$
x_j := \frac{x_j - \mu_j}{\sigma_j}
$$

Where:

- \(\mu_j\) is the mean of feature \(j\).
- \(\sigma_j\) is the standard deviation of feature \(j\).

This helps because large-scale features will not dominate small-scale features.

Gradient descent usually converges faster when features are scaled.

### Important Case

If \(\sigma_j = 0\), that feature has the same value for every example.

In code, avoid division by zero:

```python
sigma[sigma == 0] = 1
```

---

## 9. Normal Equation

### Question

What is the normal equation?

### Answer

The normal equation solves linear regression directly without gradient descent.

$$
\theta = (X^TX)^{-1}X^Ty
$$

Because matrix inverse can fail or be unstable, a better NumPy version is:

$$
\theta = \text{pinv}(X^TX)X^Ty
$$

In code:

```python
theta = np.linalg.pinv(X.T @ X) @ X.T @ y
```

### Why Not Always Use It?

Computing the inverse is expensive for many features.

Roughly:

$$
O(n^3)
$$

where \(n\) is the number of features.

So:

- Normal equation is fine for smaller feature sets.
- Gradient descent is better for large datasets or many features.

---

## 10. MSE and RMSE

### Question

Why do we compute MSE and RMSE?

### Answer

MSE reports average squared prediction error.

$$
\text{MSE} = \frac{1}{m}\sum_{i=1}^{m}(\hat{y}^{(i)} - y^{(i)})^2
$$

RMSE is the square root of MSE.

$$
\text{RMSE} = \sqrt{\text{MSE}}
$$

MSE is useful for optimization and comparison.

RMSE is easier to interpret because it is in the same unit as the target.

Example:

```text
MSE = 0.53
RMSE = 0.73
```

This means the typical prediction error is roughly `0.73` target units.

---

## 11. R Squared

### Question

What does \(R^2\) mean?

### Answer

\(R^2\) compares your model against a simple baseline model that always predicts the mean of \(y\).

$$
R^2 = 1 - \frac{\text{SSE}}{\text{SST}}
$$

Where:

$$
\text{SSE} = \sum_{i=1}^{m}(y^{(i)} - \hat{y}^{(i)})^2
$$

$$
\text{SST} = \sum_{i=1}^{m}(y^{(i)} - \bar{y})^2
$$

Interpretation:

- \(R^2 = 1\): perfect predictions.
- \(R^2 = 0\): same as predicting the mean baseline.
- \(R^2 < 0\): worse than the mean baseline.
- \(R^2 \approx 0.60\): model explains about 60% of the target variance.

### Important Correction

\(R^2 = 0.60\) does not mean exactly 40% is noise.

It means 40% of the variance is not explained by this model. That can be noise, missing features, nonlinear structure, or model limitations.

---

## 12. Train vs Test Error

### Question

Why compare train and test metrics?

### Answer

Training metrics show how well the model fits data it learned from.

Test metrics show how well the model generalizes to unseen data.

Patterns:

```text
low train error, low test error   -> good fit
low train error, high test error  -> overfitting
high train error, high test error -> underfitting
```

### Overfitting

The model captures training data too specifically and does not generalize well.

### Underfitting

The model is too simple or poorly trained and does not capture the pattern.

---

## 13. GD vs Normal Equation Comparison

### Question

Why compare gradient descent parameters to normal equation parameters?

### Answer

The normal equation gives the direct optimum for linear regression.

Gradient descent should move close to that optimum if it converges well.

A useful comparison is:

$$
\lVert \theta_{GD} - \theta_{NE} \rVert_2
$$

Expanded:

$$
\lVert \theta_{GD} - \theta_{NE} \rVert_2
= \sqrt{\sum_{j=0}^{n}(\theta_{GD,j} - \theta_{NE,j})^2}
$$

If this value becomes small as iterations increase, gradient descent is approaching the normal-equation solution.

---

## 14. Final Linear Regression Checklist

You should be able to explain:

- What \(h_\theta(x)=\theta^Tx\) means.
- Why we add \(x_0=1\).
- Why squared error is used.
- Why the cost uses \(1/(2m)\).
- How gradient descent updates \(\theta\).
- What the learning rate does.
- What convergence and divergence mean.
- Why feature scaling helps.
- What the normal equation does.
- Why normal equation can be expensive.
- How MSE, RMSE, and \(R^2\) evaluate performance.
- How train/test metrics reveal overfitting and underfitting.
