# Logistic Regression Quiz Review

This file collects the logistic regression questions we reviewed, with corrected explanations, formulas, examples, and implementation notes.

---

## 1. Why Linear Regression Is Not Ideal for Classification

### Question

If \(y\in\{0,1\}\), why not use linear regression and threshold at `0.5`?

### Answer

Linear regression outputs unbounded values.

$$
h_\theta(x) = \theta^Tx
$$

This can produce:

```text
-3, 0.2, 1.7, 10
```

But binary classification needs something interpretable as a probability:

$$
0 \le P(y=1 \mid x) \le 1
$$

Logistic regression fixes this by passing the linear score through the sigmoid function:

$$
h_\theta(x) = g(\theta^Tx)
$$

where:

$$
g(z) = \frac{1}{1 + e^{-z}}
$$

Now the output is always between `0` and `1`.

---

## 2. Sigmoid Function

### Question

What does the sigmoid function do?

### Answer

The sigmoid converts any real number into a value between `0` and `1`.

$$
g(z) = \frac{1}{1 + e^{-z}}
$$

So:

$$
0 < g(z) < 1
$$

Examples:

```text
z = -10 -> sigmoid(z) ≈ 0.00005
z = 0   -> sigmoid(z) = 0.5
z = 10  -> sigmoid(z) ≈ 0.99995
```

### Important Correction

Sigmoid does not directly output class `0` or class `1`.

It outputs a probability.

Then a threshold converts probability into a class:

$$
\hat{y} =
\begin{cases}
1 & \text{if } h_\theta(x) \ge 0.5 \\
0 & \text{if } h_\theta(x) < 0.5
\end{cases}
$$

---

## 3. Logistic Hypothesis

### Question

What does the logistic hypothesis mean?

$$
h_\theta(x) = \frac{1}{1 + e^{-\theta^Tx}}
$$

### Answer

This gives the probability that the example belongs to class `1`.

$$
h_\theta(x) = P(y=1 \mid x;\theta)
$$

Meaning:

- \(x\) is the input feature vector.
- \(\theta\) is the learned parameter vector.
- \(\theta^Tx\) is the linear score, also called the logit.
- The sigmoid turns the score into a probability.

Example:

```text
h_theta(x) = 0.82
```

means:

```text
The model estimates an 82% probability that y = 1.
```

---

## 4. Logistic Regression Likelihood

### Question

Where does logistic regression cost come from?

### Answer

Logistic regression models a Bernoulli output.

$$
P(y=1 \mid x) = h_\theta(x)
$$

$$
P(y=0 \mid x) = 1 - h_\theta(x)
$$

For one example, both cases can be written as:

$$
P(y \mid x;\theta) = h_\theta(x)^y(1-h_\theta(x))^{1-y}
$$

If \(y=1\):

$$
P(y \mid x;\theta) = h_\theta(x)
$$

If \(y=0\):

$$
P(y \mid x;\theta) = 1-h_\theta(x)
$$

Training chooses \(\theta\) that makes the observed labels likely.

---

## 5. Cross-Entropy Cost

### Question

What is the logistic regression cost function?

### Answer

Instead of maximizing likelihood, we minimize negative log-likelihood.

$$
J(\theta) = -\frac{1}{m}\sum_{i=1}^{m}\left[y^{(i)}\log(h_\theta(x^{(i)})) + (1-y^{(i)})\log(1-h_\theta(x^{(i)}))\right]
$$

This is also called:

- cross-entropy loss
- log loss
- negative log-likelihood

### Why Two Parts?

The first part matters when \(y=1\):

$$
-y\log(h_\theta(x))
$$

The second part matters when \(y=0\):

$$
-(1-y)\log(1-h_\theta(x))
$$

Because \(y\) is either `0` or `1`, one side turns off.

### If \(y=1\)

$$
J = -\log(h_\theta(x))
$$

So if the model predicts a low probability for class `1`, the loss is large.

### If \(y=0\)

$$
J = -\log(1-h_\theta(x))
$$

So if the model predicts a high probability for class `1` when the truth is `0`, the loss is large.

---

## 6. Why Not Squared Error?

### Question

Why do we not use squared error for logistic regression?

### Answer

With sigmoid plus squared error, the objective can become non-convex and harder to optimize.

Cross-entropy is better because:

- It comes naturally from Bernoulli maximum likelihood.
- It gives a convex objective for logistic regression.
- It strongly penalizes confident wrong predictions.

Example:

If the true label is `1`, but the model predicts:

```text
h = 0.01
```

Then:

$$
-\log(0.01) = 4.605
$$

That is a large penalty because the model was confidently wrong.

---

## 7. Logistic Gradient Descent

### Question

How is logistic regression gradient descent similar to linear regression?

### Answer

The update has the same outer structure:

$$
\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j}J(\theta)
$$

For logistic regression:

$$
\frac{\partial}{\partial \theta_j}J(\theta)
= \frac{1}{m}\sum_{i=1}^{m}\left(h_\theta(x^{(i)}) - y^{(i)}\right)x_j^{(i)}
$$

So:

$$
\theta_j := \theta_j - \alpha \frac{1}{m}\sum_{i=1}^{m}\left(h_\theta(x^{(i)}) - y^{(i)}\right)x_j^{(i)}
$$

The formula looks like linear regression, but now:

$$
h_\theta(x) = g(\theta^Tx)
$$

So under the hood, the prediction uses sigmoid.

---

## 8. Prediction and Threshold

### Question

What does the threshold do?

### Answer

The model outputs a probability.

$$
h_\theta(x) = P(y=1 \mid x;\theta)
$$

A threshold converts that probability into a class.

Default threshold:

$$
\hat{y} =
\begin{cases}
1 & \text{if } h_\theta(x) \ge 0.5 \\
0 & \text{otherwise}
\end{cases}
$$

### Threshold Tradeoff

If threshold increases from `0.5` to `0.7`:

- The model becomes stricter about predicting class `1`.
- False positives usually decrease.
- False negatives may increase.
- Precision may improve.
- Recall may decrease.

Example from your project:

```text
Threshold 0.3: more class-1 predictions, higher recall, more false positives
Threshold 0.7: fewer class-1 predictions, fewer false positives, more false negatives
```

---

## 9. Confusion Matrix

### Question

What do TN, FP, FN, and TP mean?

### Answer

For binary classification:

```text
TN = true negative  = predicted 0, true 0
FP = false positive = predicted 1, true 0
FN = false negative = predicted 0, true 1
TP = true positive  = predicted 1, true 1
```

Confusion matrix:

```text
TN  FP
FN  TP
```

Your example:

```text
TN=42, FP=0
FN=0,  TP=72
```

Meaning:

- 42 ham examples correctly predicted as ham.
- 72 spam examples correctly predicted as spam.
- 0 ham examples wrongly predicted as spam.
- 0 spam examples wrongly predicted as ham.

---

## 10. Precision, Recall, and F1

### Question

What are metrics beyond accuracy?

### Answer

Accuracy:

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

Precision:

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

Precision answers:

```text
Out of everything predicted as spam, how many were actually spam?
```

Recall:

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

Recall answers:

```text
Out of all real spam messages, how many did we catch?
```

F1 score:

$$
F1 = \frac{2(\text{Precision})(\text{Recall})}{\text{Precision}+\text{Recall}}
$$

F1 balances precision and recall.

---

## 11. Accuracy Can Be Misleading

### Question

If a spam dataset has 95% ham and 5% spam, is 95% accuracy good?

### Answer

Not necessarily.

A model that always predicts `ham` would get 95% accuracy but catch zero spam.

That is why spam classification needs:

- confusion matrix
- precision
- recall
- F1 score

Accuracy alone can hide poor performance on the minority class.

---

## 12. Convergence Plot

### Question

What does a logistic convergence plot show?

### Answer

It plots cost over training iterations.

```python
plt.plot(range(len(cost_history)), cost_history)
```

Meaning:

- x-axis: iteration number
- y-axis: logistic cost

A good convergence plot shows:

```text
cost decreases and then flattens
```

If cost increases or becomes `NaN`, the learning rate may be too high or the implementation may be wrong.

---

## 13. Learning Curve

### Question

What does a learning curve show?

### Answer

A learning curve shows model error as training-set size increases.

Classification error:

$$
\text{Error} = 1 - \text{Accuracy}
$$

Usually you plot:

- train error vs training-set size
- test error vs training-set size

Interpretation:

```text
low train error, high test error  -> overfitting
high train error, high test error -> underfitting
low train error, low test error   -> good fit
```

Your logistic curve had low train/test error, so it looked like a good fit overall.

---

## 14. Final Logistic Regression Checklist

You should be able to explain:

- Why linear regression is not ideal for binary classification.
- What sigmoid does.
- Why sigmoid outputs probability, not class directly.
- What \(h_\theta(x)=P(y=1\mid x;\theta)\) means.
- How Bernoulli likelihood leads to cross-entropy.
- Why cross-entropy has two parts.
- Why squared error is not preferred.
- Why the gradient looks similar to linear regression.
- What threshold means.
- How threshold affects precision and recall.
- What TN, FP, FN, TP mean.
- Why accuracy can be misleading.
- How convergence plots and learning curves help diagnose training.
