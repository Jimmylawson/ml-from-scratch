# Gaussian Discriminant Analysis Revision Notes

This folder implements Gaussian Discriminant Analysis (GDA) from scratch.

Your code uses the Iris dataset and extends the binary GDA idea to multiclass classification.

## 1. What GDA Is

GDA is a generative learning algorithm.

Discriminative models learn:

$$
p(y\mid x)
$$

Generative models learn:

$$
p(x\mid y)\quad \text{and}\quad p(y)
$$

Then they use Bayes rule to predict:

$$
p(y\mid x)
=
\frac{p(x\mid y)p(y)}{p(x)}
$$

For prediction, $p(x)$ is the same for all classes, so we compare:

$$
\arg\max_y\ p(x\mid y)p(y)
$$

In log space:

$$
\arg\max_y
\left[
\log p(y) + \log p(x\mid y)
\right]
$$

That is exactly what your `predict_one` function does.

## 2. GDA Assumption

GDA assumes that the features are continuous and Gaussian within each class.

For binary GDA:

```text
y in {0, 1}
```

Class prior:

```text
y ~ Bernoulli(phi)
```

Class conditional distributions:

```text
x | y = 0 ~ N(mu_0, Sigma)
x | y = 1 ~ N(mu_1, Sigma)
```

The important assumption is that both classes share the same covariance matrix `Sigma`.

For multiclass GDA:

```text
y in {0, 1, ..., K-1}
```

Each class has its own mean:

```text
x | y = k ~ N(mu_k, Sigma)
```

but all classes still share the same covariance matrix:

```text
Sigma
```

## 3. Data Shape

For Iris:

```text
X shape: (150, 4)
y shape: (150,)
```

Meaning:

- 150 examples
- 4 features per example
- 3 classes: `0`, `1`, `2`

Each row of `X` is one flower:

```text
[sepal length, sepal width, petal length, petal width]
```

Each `y` value is the class label.

## 4. Train/Test Split

Your main file shuffles the data:

```python
rg = np.random.default_rng(42)
idx = rg.permutation(X.shape[0])
X = X[idx]
y = y[idx]
```

This keeps features and labels aligned.

Then:

```python
m_train = int(0.8 * m)
X_train = X[:m_train]
X_test = X[m_train:]
y_train = y[:m_train]
y_test = y[m_train:]
```

The model estimates parameters using training data, then tests on unseen data.

## 5. Class Prior

The prior is:

```text
p(y = k)
```

It answers:

```text
Before seeing the features, how common is class k?
```

Code:

```python
def class_prior(y_train):
    classes = np.unique(y_train)
    return np.array([np.mean(y_train == k) for k in classes]), classes
```

Math:

```text
phi_k = (number of training examples with y = k) / m
```

Code mapping:

```python
np.mean(y_train == k)
```

works because `y_train == k` creates booleans:

```text
True  -> 1
False -> 0
```

So the mean becomes the fraction of examples in class `k`.

Example:

```text
y_train = [0, 0, 1, 2, 2]
```

For class `2`:

```text
y_train == 2 -> [False, False, False, True, True]
mean -> 2/5 = 0.4
```

## 6. Mean Vector

Each class has a mean vector:

```text
mu_k = average of x examples where y = k
```

Code:

```python
def mean_vector(X_train, y_train, classes):
    return np.array([np.mean(X_train[y_train == k], axis=0) for k in classes])
```

The expression:

```python
X_train[y_train == k]
```

selects only the rows of `X_train` that belong to class `k`.

Example:

```text
X_train = [[5.1, 3.5],
           [4.9, 3.0],
           [6.2, 2.8]]

y_train = [0, 0, 1]
```

For class `0`:

```text
X_train[y_train == 0]
= [[5.1, 3.5],
   [4.9, 3.0]]
```

Then:

```python
np.mean(..., axis=0)
```

computes the average per feature/column:

```text
mu_0 = [5.0, 3.25]
```

`axis=0` means: go down the rows and compute one mean per column.

## 7. Multivariate Gaussian Density

Andrew Ng's notes use the multivariate normal density:

$$
p(x;\mu,\Sigma)
=
\frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}}
\exp\left(
-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)
\right)
$$

Where:

- $d$ is the number of features.
- $\mu$ is the mean vector for one class.
- $\Sigma$ is the covariance matrix.
- $|\Sigma|$ is the determinant of the covariance matrix.
- $\Sigma^{-1}$ is the inverse covariance matrix.
- $(x-\mu)$ is how far the point is from the class mean.
- $(x-\mu)^T\Sigma^{-1}(x-\mu)$ is the covariance-aware distance from the mean.

## 8. Shared Covariance Matrix

For GDA, the shared covariance matrix is:

$$
\Sigma
=
\frac{1}{m}\sum_{i=1}^{m}
\left(x_i - \mu_{y_i}\right)
\left(x_i - \mu_{y_i}\right)^T
$$

Meaning:

For every training example:

1. Find its class
2. Subtract that class mean
3. Compute the outer product
4. Add it to the covariance matrix
5. Divide by `m`

Code:

```python
def covariance_matrix(X_train, y_train, classes, mus):
    m = len(y_train)
    n_features = X_train.shape[1]
    sigma = np.zeros([n_features, n_features])

    for i in range(m):
        class_index = np.where(classes == y_train[i])[0][0]
        diff = X_train[i] - mus[class_index]
        diff = diff.reshape(-1, 1)
        sigma += diff @ diff.T

    return sigma / m
```

## 9. Why `diff.reshape(-1, 1)`?

Suppose:

```text
diff = [2, 3]
```

This has shape:

```text
(2,)
```

To compute the covariance contribution, we need:

```text
diff diff^T
```

So we reshape it into a column:

```text
[[2],
 [3]]
```

Then:

```text
diff @ diff.T
```

gives:

```text
[[2],
 [3]]
@
[[2, 3]]
=
[[4, 6],
 [6, 9]]
```

That matrix says how features vary together.

## 10. Gaussian Log Likelihood

Your code computes the log of the Gaussian density instead of the density directly:

```python
def gaussian_log_likelihood(x, mu, sigma):
    d = len(x)
    diff = x - mu
    inv_sigma = np.linalg.inv(sigma)
    det_sigma = np.linalg.det(sigma)

    return (
        -0.5 * diff.T @ inv_sigma @ diff
        -0.5 * np.log(det_sigma)
        -0.5 * d * np.log(2 * np.pi)
    )
```

This is the log version of:

$$
p(x;\mu,\Sigma)
=
\frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}}
\exp\left(
-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)
\right)
$$

Taking logs gives:

$$
\log p(x;\mu,\Sigma)
=
-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)
-\frac{1}{2}\log|\Sigma|
-\frac{d}{2}\log(2\pi)
$$

Code mapping:

```python
-0.5 * diff.T @ inv_sigma @ diff
```

is:

$$
-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)
$$

This measures how far `x` is from the class mean, using the covariance shape.

```python
-0.5 * np.log(det_sigma)
```

is:

$$
-\frac{1}{2}\log|\Sigma|
$$

```python
-0.5 * d * np.log(2 * np.pi)
```

is:

$$
-\frac{d}{2}\log(2\pi)
$$

## 11. Why Use Logs?

The original prediction uses:

$$
p(x\mid y=k)p(y=k)
$$

But Gaussian densities can be very small. Multiplying many small numbers can underflow numerically.

Logs convert products into sums:

$$
\log\left[p(x\mid y=k)p(y=k)\right]
=
\log p(x\mid y=k) + \log p(y=k)
$$

This is safer and easier to compare.

## 12. Prediction

Prediction compares each class score:

$$
score_k
=
\log p(y=k)
+
\log p(x\mid y=k)
$$

Then chooses the class with the largest score:

$$
\hat{y}
=
\arg\max_k\ score_k
$$

Code:

```python
def predict_one(x, prior, classes, mus, sigma):
    scores = []

    for k_idx, k in enumerate(classes):
        score = np.log(prior[k_idx]) + gaussian_log_likelihood(x, mus[k_idx], sigma)
        scores.append(score)

    return classes[np.argmax(scores)]
```

`np.argmax(scores)` returns the index of the largest score.

`classes[...]` converts that index back to the actual class label.

## 13. Binary vs Multiclass GDA

Andrew's notes first present binary GDA:

```text
y in {0, 1}
```

Your implementation is multiclass:

```text
y in {0, 1, 2}
```

The multiclass version is the same idea:

- estimate one prior per class
- estimate one mean vector per class
- estimate one shared covariance matrix
- compute one score per class
- pick the largest score

Multiclass GDA can also handle binary data. Binary is just the case where `K = 2`.

## 14. Accuracy

Code:

```python
y_pred = np.array([predict_one(x, priors, classes, mus, sigma) for x in X_test])
acc = np.mean(y_pred == y_test)
```

Meaning:

```text
accuracy = number of correct predictions / number of test examples
```

If accuracy is:

```text
1.0000
```

then every test example in that split was classified correctly.

This does not mean GDA is always perfect. It means Iris is a small, clean dataset and the chosen split was easy for this model.

## 15. Big Picture

GDA learns how each class generates data.

For every class, it asks:

```text
How likely is this x under this class Gaussian?
How common is this class overall?
```

Then it chooses:

```text
class with largest log p(y) + log p(x | y)
```

The most important formula:

$$
score_k
=
\log p(y=k)
+
\log \mathcal{N}(x;\mu_k,\Sigma)
$$

That is the heart of your GDA implementation.
