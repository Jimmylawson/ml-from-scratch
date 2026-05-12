# Softmax Regression Quiz Review

This file collects the softmax regression and GLM-related questions we reviewed, with corrected explanations, formulas, examples, and implementation notes.

---

## 1. What Is Softmax Regression?

### Question

How is softmax regression related to logistic regression?

### Answer

Softmax regression is logistic regression generalized from two classes to many classes.

Logistic regression:

$$
y \in \{0,1\}
$$

Softmax regression:

$$
y \in \{0,1,2,\ldots,K-1\}
$$

Example:

```text
digit classification: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
K = 10 classes
```

Memory:

```text
Logistic regression: one score -> sigmoid -> probability for class 1
Softmax regression:  K scores -> softmax -> probabilities over K classes
```

---

## 2. One Score Per Class

### Question

Why does softmax need one score per class?

### Answer

The model must compare all possible classes.

For digit classification, it asks:

```text
How strongly does this image look like class 0?
How strongly does this image look like class 1?
...
How strongly does this image look like class 9?
```

So it computes:

$$
z_0 = \theta_0^Tx
$$

$$
z_1 = \theta_1^Tx
$$

$$
\cdots
$$

$$
z_{K-1} = \theta_{K-1}^Tx
$$

Then softmax converts the scores into probabilities.

---

## 3. Parameter Matrix

### Question

Why does softmax use a parameter matrix \(\Theta\)?

### Answer

Each class has its own parameter vector.

For \(K\) classes:

```text
theta_0 -> parameters for class 0
theta_1 -> parameters for class 1
...
theta_K-1 -> parameters for class K-1
```

These vectors are stored in one matrix:

$$
\Theta \in \mathbb{R}^{n \times K}
$$

If:

$$
X \in \mathbb{R}^{m \times n}
$$

then:

$$
Z = X\Theta
$$

and:

$$
Z \in \mathbb{R}^{m \times K}
$$

Each row contains scores for one example across all classes.

---

## 4. Meaning of \(m\), \(n\), and \(K\)

### Question

What do examples, features, and classes mean?

### Answer

Use these meanings:

```text
m = number of examples
n = number of features
K = number of classes
```

Shapes:

$$
X.shape = (m,n)
$$

$$
\Theta.shape = (n,K)
$$

$$
scores.shape = (m,K)
$$

$$
probs.shape = (m,K)
$$

$$
y.shape = (m,)
$$

Example:

```text
4 examples
3 features
5 classes
```

Then:

```text
X.shape      = (4, 3)
Theta.shape  = (3, 5)
scores.shape = (4, 5)
probs.shape  = (4, 5)
y.shape      = (4,)
```

For `scores` and `probs`:

```text
rows = examples
columns = classes
```

So:

```text
probs[0, 2]
```

means probability that example `0` belongs to class `2`.

---

## 5. Softmax Function

### Question

How does softmax convert scores into probabilities?

### Answer

For one example and class \(k\):

$$
P(y=k \mid x) = \frac{e^{z_k}}{\sum_{j=1}^{K}e^{z_j}}
$$

where:

$$
z_k = \theta_k^Tx
$$

Softmax does two things:

- Makes every probability positive.
- Makes all class probabilities sum to `1`.

### Example

Suppose scores are:

```text
z = [3, 1, 0.2]
```

Softmax gives something like:

```text
probs = [0.84, 0.11, 0.05]
```

The model predicts class `0` because class `0` has the highest probability.

---

## 6. Softmax Normalization

### Question

Why do we divide by the sum of exponentials?

### Answer

The denominator normalizes the scores into a probability distribution.

$$
\sum_{k=1}^{K}P(y=k \mid x) = 1
$$

Without the denominator, exponentiated scores are positive, but they are not valid probabilities.

The denominator makes them comparable and forces the probabilities to sum to `1`.

---

## 7. Numerical Stability

### Question

Why do we subtract the maximum logit before applying `exp`?

### Answer

Large exponentials can overflow.

Example:

```python
np.exp(1000)
```

can become infinity.

So we compute:

$$
z'_k = z_k - \max_j z_j
$$

Then:

$$
\text{softmax}(z) = \text{softmax}(z')
$$

Subtracting the same constant from every score does not change the final probabilities.

In code:

```python
logits = logits - np.max(logits, axis=1, keepdims=True)
```

### Why `axis=1`?

For matrix logits with shape `(m, K)`, each row is one example and each column is one class.

`axis=1` means:

```text
for each example, find max across classes
```

### Why `keepdims=True`?

It keeps the result shape as `(m, 1)` so NumPy can subtract it from `(m, K)` correctly.

---

## 8. Cross-Entropy Loss

### Question

What loss does softmax regression use?

### Answer

For one example:

$$
\text{loss}^{(i)} = -\log P(y=y^{(i)} \mid x^{(i)})
$$

For all examples:

$$
J(\Theta) = -\frac{1}{m}\sum_{i=1}^{m}\log P(y=y^{(i)} \mid x^{(i)})
$$

Using probability notation:

$$
J(\Theta) = -\frac{1}{m}\sum_{i=1}^{m}\log p_{i,y^{(i)}}
$$

where \(p_{i,y^{(i)}}\) is the probability assigned to the correct class for example \(i\).

### Why Negative Log?

If the model gives high probability to the correct class, loss is small.

```text
correct probability = 0.99 -> -log(0.99) ≈ 0.01
correct probability = 0.10 -> -log(0.10) ≈ 2.30
correct probability = 0.01 -> -log(0.01) ≈ 4.61
```

So confident wrong predictions are heavily penalized.

---

## 9. Correct Class Probability Indexing

### Question

What does this line mean?

```python
correct_class_probs = probs[np.arange(m), y]
```

### Answer

It selects the probability assigned to the true class for each example.

Suppose:

```python
probs = [
    [0.80, 0.10, 0.10],
    [0.20, 0.70, 0.10],
    [0.05, 0.15, 0.80],
]
y = [0, 1, 2]
```

Then:

```python
np.arange(m) = [0, 1, 2]
```

So:

```python
probs[np.arange(m), y]
```

means:

```text
probs[0, 0] = 0.80
probs[1, 1] = 0.70
probs[2, 2] = 0.80
```

Result:

```text
[0.80, 0.70, 0.80]
```

Important: this does not sort probabilities. It picks the probability of the true class for each row.

---

## 10. One-Hot Labels

### Question

Why do we sometimes use one-hot labels?

### Answer

For \(K\) classes, one-hot labels represent the true class as a vector.

Example for \(K=3\):

```text
class 0 -> [1, 0, 0]
class 1 -> [0, 1, 0]
class 2 -> [0, 0, 1]
```

In NumPy:

```python
y_onehot = np.eye(K)[y]
```

If:

```python
y = [0, 2, 1]
```

Then:

```text
y_onehot = [
    [1, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
]
```

---

## 11. Softmax Gradient

### Question

What is the softmax gradient?

### Answer

The key error term is:

$$
\text{error} = \text{probs} - y_{onehot}
$$

Then the gradient is:

$$
\nabla_\Theta J(\Theta) = \frac{1}{m}X^T(\text{probs} - y_{onehot})
$$

In code:

```python
probs = softmax(X, theta)
y_onehot = np.eye(K)[y]
gradient = (1 / m) * X.T @ (probs - y_onehot)
```

### Meaning

For each example:

```text
predicted probability distribution - true one-hot distribution
```

This tells the model how to adjust each class parameter vector.

---

## 12. Gradient Descent for Softmax

### Question

How do we train softmax regression?

### Answer

Use gradient descent:

$$
\Theta := \Theta - \alpha \nabla_\Theta J(\Theta)
$$

Training loop:

```python
cost_history = []
for i in range(num_iters):
    gradient = softmax_gradient(X, y, theta)
    theta = theta - alpha * gradient
    cost = cross_entropy_loss(X, y, theta)
    cost_history.append(cost)
```

The cost history lets you plot convergence.

---

## 13. Prediction

### Question

What does `predict_class` do?

### Answer

It converts class probabilities into a discrete class prediction.

First compute probabilities:

$$
P(y=k \mid x)
$$

Then choose the class with the largest probability:

$$
\hat{y} = \arg\max_k P(y=k \mid x)
$$

In code:

```python
preds = np.argmax(probs, axis=1)
```

`axis=1` means choose the best class across columns for each example.

---

## 14. Accuracy

### Question

How do we evaluate softmax regression?

### Answer

Accuracy measures the fraction of predictions that match the true labels.

$$
\text{Accuracy} = \frac{1}{m}\sum_{i=1}^{m}1\{\hat{y}^{(i)} = y^{(i)}\}
$$

In code:

```python
accuracy = np.mean(y_pred == y_true)
```

Why this works:

```text
True  -> 1
False -> 0
```

So the mean of boolean correctness gives the fraction correct.

---

## 15. Confusion Matrix and Per-Class Accuracy

### Question

Why did we compute a confusion matrix and per-class accuracy?

### Answer

Overall accuracy tells the total performance.

But it can hide which classes are weak.

A confusion matrix shows true class vs predicted class.

For multiclass:

```text
rows    = true classes
columns = predicted classes
```

Per-class accuracy asks:

```text
For class k, out of all true class-k examples, how many were correctly predicted?
```

Formula:

$$
\text{Accuracy}_k = \frac{\text{correct predictions for class }k}{\text{total examples of class }k}
$$

Example from your output:

```text
class 8: 0.8788 (29/33)
```

Meaning:

```text
There were 33 true examples of class 8.
The model got 29 correct.
```

---

## 16. GLM Connection

### Question

How does softmax connect to GLMs?

### Answer

Linear regression, logistic regression, and softmax regression are connected through Generalized Linear Models.

They use:

- a probability distribution
- a linear natural parameter
- a link between the linear score and prediction

For GLMs, the natural parameter is usually made linear:

$$
\eta = \theta^Tx
$$

For logistic regression, Bernoulli leads to sigmoid:

$$
P(y=1\mid x) = \frac{1}{1+e^{-\theta^Tx}}
$$

For softmax regression, multinomial leads to softmax:

$$
P(y=k\mid x) = \frac{e^{\theta_k^Tx}}{\sum_{j=1}^{K}e^{\theta_j^Tx}}
$$

So softmax is the multiclass GLM version of logistic regression.

---

## 17. Final Softmax Checklist

You should be able to explain:

- Why softmax is multiclass logistic regression.
- Why softmax needs one score per class.
- Why \(\Theta\) is a matrix.
- Meaning of \(m\), \(n\), and \(K\).
- Why softmax probabilities sum to `1`.
- Why subtracting max logits prevents overflow.
- What `axis=1` means.
- What `keepdims=True` does.
- Why cross-entropy uses the correct-class probability.
- What `probs[np.arange(m), y]` does.
- What one-hot labels mean.
- Why the gradient is \(X^T(probs-y_{onehot})/m\).
- How `predict_class` uses argmax.
- How accuracy and per-class accuracy evaluate the model.
