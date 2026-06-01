# SVM Quiz Review

This file collects the SVM Chapter 6 review questions we went through, with corrected explanations, formulas, examples, and implementation connections.

---

## 1. Prediction Score and Class Prediction

### Question

What is the SVM prediction score?

$$
w^Tx + b
$$

And how does it become a class prediction?

$$
\hat{y} = \operatorname{sign}(w^Tx+b)
$$

### Answer

The quantity:

$$
w^Tx + b
$$

is the **prediction score**.

It tells which side of the decision boundary the point is on.

The predicted class is:

$$
\hat{y} = \operatorname{sign}(w^Tx+b)
$$

So:

- If \(w^Tx+b > 0\), predict \(+1\).
- If \(w^Tx+b < 0\), predict \(-1\).
- If \(w^Tx+b = 0\), the point is exactly on the decision boundary.

### Important Correction

A positive score does **not** automatically mean the prediction is correct.

It only means the model predicts \(+1\).

To know if the prediction is correct, compare with the true label \(y\):

$$
y(w^Tx+b) > 0
$$

If this is positive, the prediction is correct.

### Example

If the true label is \(+1\) and the score is `3`:

$$
y(w^Tx+b) = (+1)(3)=3>0
$$

Correct.

If the true label is \(-1\) and the score is `3`:

$$
y(w^Tx+b) = (-1)(3)=-3<0
$$

Wrong.

---

## 2. Decision Boundary

### Question

What is the decision boundary of an SVM?

$$
w^Tx+b=0
$$

### Answer

The decision boundary is the separator between the two predicted regions.

$$
w^Tx+b=0
$$

Points on this boundary are exactly between class \(+1\) and class \(-1\).

In 2D, the boundary is a line.

In 3D, it is a plane.

In higher dimensions, it is called a hyperplane.

### Separator

The **separator** is another name for the decision boundary.

```text
class -1        separator        class +1

   o o o    |  w^T x + b = 0  |     x x x
```

So when we say SVM finds the best separator, we mean:

```text
SVM finds the best decision boundary.
```

---

## 3. Decision Boundary vs Margin

### Question

Is the decision boundary the same as the margin?

### Answer

No.

The decision boundary is the middle separating line:

$$
w^Tx+b=0
$$

The margin lines are:

$$
w^Tx+b=1
$$

and:

$$
w^Tx+b=-1
$$

The margin is the empty space around the decision boundary.

So:

- Decision boundary: middle separator.
- Margin lines: the two parallel lines around it.
- Margin width: distance between those margin lines.

---

## 4. Maximum Margin Idea

### Question

What does it mean that SVM finds the separating boundary with the widest margin?

### Answer

SVM does not just want any line that separates the classes.

It wants the separating line that leaves the largest safety gap between the classes.

The margin lines are:

$$
w^Tx+b=1
$$

$$
w^Tx+b=-1
$$

The decision boundary is:

$$
w^Tx+b=0
$$

SVM chooses the boundary that makes the space between the margin lines as wide as possible.

Why this helps:

- A wider margin usually generalizes better.
- The classifier is less sensitive to small changes and noise.
- Points far from the boundary are classified with more confidence.
- Points close to the boundary are risky.

Memory:

```text
SVM = maximum margin classifier
```

---

## 5. Functional Margin

### Question

What does this quantity mean?

$$
y^{(i)}(w^Tx^{(i)}+b)
$$

Why multiply by \(y^{(i)}\)?

### Answer

This is the **functional margin**.

It combines:

- the true label \(y^{(i)}\)
- the model score \(w^Tx^{(i)}+b\)

SVM labels are usually:

$$
y^{(i)} \in \{-1,+1\}
$$

Multiplying by \(y^{(i)}\) makes correctness easy to check with one number.

If the true label is \(+1\):

$$
y^{(i)}(w^Tx^{(i)}+b) = w^Tx^{(i)}+b
$$

So it is correct when the score is positive.

If the true label is \(-1\):

$$
y^{(i)}(w^Tx^{(i)}+b) = -(w^Tx^{(i)}+b)
$$

So it is correct when the score is negative, because negative times negative becomes positive.

Meaning:

$$
y^{(i)}(w^Tx^{(i)}+b)>0
$$

means correctly classified.

$$
y^{(i)}(w^Tx^{(i)}+b)<0
$$

means incorrectly classified.

---

## 6. Correct Classification vs Margin Satisfaction

### Question

What is the difference between these two conditions?

$$
y^{(i)}(w^Tx^{(i)}+b)>0
$$

and:

$$
y^{(i)}(w^Tx^{(i)}+b) \ge 1
$$

### Answer

The first condition means the point is on the correct side of the decision boundary.

$$
y^{(i)}(w^Tx^{(i)}+b)>0
$$

So the example is correctly classified.

The second condition means the point is correctly classified **and** outside or on the margin.

$$
y^{(i)}(w^Tx^{(i)}+b) \ge 1
$$

Interpretation:

- \(>0\): correct side of boundary.
- \(=1\): exactly on the margin.
- \(>1\): correct and safely outside the margin.
- \(0 < \text{margin} < 1\): correct, but inside the margin.
- \(<0\): wrong side, misclassified.

---

## 7. Hinge Loss

### Question

What is hinge loss, and why does SVM use it?

$$
\max(0, 1-y^{(i)}(w^Tx^{(i)}+b))
$$

### Answer

Hinge loss measures **margin violation**.

Let:

$$
m_i = y^{(i)}(w^Tx^{(i)}+b)
$$

Then hinge loss is:

$$
\max(0, 1-m_i)
$$

### Cases

If:

$$
m_i \ge 1
$$

then:

$$
\max(0,1-m_i)=0
$$

No loss. The point is correctly classified and outside/on the margin.

If:

$$
0 < m_i < 1
$$

then the point is correctly classified, but inside the margin. There is loss.

If:

$$
m_i < 0
$$

then the point is misclassified. There is larger loss.

### Why SVM Uses Hinge Loss

Real-world data is messy and not always perfectly separable.

Hinge loss allows soft-margin SVM to tolerate errors by penalizing margin violations instead of requiring perfect separation.

Better wording:

```text
Hinge loss measures how much we violated the margin.
```

Not:

```text
how much we violated the error
```

---

## 8. Soft-Margin SVM Objective

### Question

Explain the soft-margin SVM objective:

$$
J(w,b) = \frac{1}{2}\lVert w\rVert^2 + C\cdot\frac{1}{m}\sum_{i=1}^{m}\max(0,1-y^{(i)}(w^Tx^{(i)}+b))
$$

### Answer

This objective has two main parts.

### Part 1: Margin / Regularization Term

$$
\frac{1}{2}\lVert w\rVert^2
$$

This tries to keep \(w\) small.

Why?

Because margin width is related to:

$$
\frac{1}{\lVert w\rVert}
$$

So a smaller \(\lVert w\rVert\) means a wider margin.

### Part 2: Average Hinge Loss

$$
\frac{1}{m}\sum_{i=1}^{m}\max(0,1-y^{(i)}(w^Tx^{(i)}+b))
$$

This measures the average margin violation across the training set.

It penalizes:

- points inside the margin
- misclassified points

### Part 3: The \(C\) Parameter

$$
C
$$

This controls the tradeoff between wide margin and fewer violations.

Large \(C\):

```text
penalizes violations heavily
stricter about mistakes
tries harder to classify training data correctly
can overfit more
```

Small \(C\):

```text
penalizes violations less
allows more margin violations
more flexible
can generalize better if data is noisy
```

---

## 9. Why Minimizing \(\lVert w\rVert^2\) Maximizes the Margin

### Question

Why does minimizing this term create a wider margin?

$$
\frac{1}{2}\lVert w\rVert^2
$$

### Answer

The geometric margin width is proportional to:

$$
\frac{1}{\lVert w\rVert}
$$

So:

- smaller \(\lVert w\rVert\) means larger margin
- larger \(\lVert w\rVert\) means smaller margin

More precisely, the distance between the two margin lines:

$$
w^Tx+b=1
$$

and:

$$
w^Tx+b=-1
$$

is:

$$
\frac{2}{\lVert w\rVert}
$$

So minimizing:

$$
\lVert w\rVert^2
$$

maximizes the margin width.

---

## 10. Hard-Margin vs Soft-Margin SVM

### Question

What is the difference between hard-margin SVM and soft-margin SVM?

### Answer

Hard-margin SVM assumes the data is perfectly linearly separable.

It requires:

$$
y^{(i)}(w^Tx^{(i)}+b) \ge 1
$$

for every training example.

That means:

```text
no misclassified points
no points inside the margin
```

Soft-margin SVM allows violations.

It uses hinge loss:

$$
\max(0,1-y^{(i)}(w^Tx^{(i)}+b))
$$

So points can be:

- correctly classified and outside the margin
- correctly classified but inside the margin
- misclassified

Soft margin is used more in practice because real datasets are noisy and not perfectly separable.

---

## 11. Support Vectors

### Question

What are support vectors?

Why are they important?

### Answer

Support vectors are **not** the decision boundary.

Support vectors are the training examples closest to the decision boundary.

They are the points that lie on or inside the margin.

For hard-margin SVM, support vectors usually satisfy:

$$
y^{(i)}(w^Tx^{(i)}+b)=1
$$

They are important because they determine the final boundary.

If you move a far-away point, the boundary usually does not change.

But if you move a support vector, the boundary can change.

Correct memory:

```text
Support vectors are the closest training points to the decision boundary.
They determine the margin and the final decision boundary.
```

---

## 12. Geometric Distance to the Boundary

### Question

What does this mean geometrically?

$$
\frac{w^Tx+b}{\lVert w\rVert}
$$

### Answer

This is the **signed distance** from point \(x\) to the decision boundary.

The decision boundary is:

$$
w^Tx+b=0
$$

So:

- Positive value: point is on the \(+1\) side.
- Negative value: point is on the \(-1\) side.
- Zero: point is exactly on the decision boundary.

The absolute value gives the actual distance:

$$
\left|\frac{w^Tx+b}{\lVert w\rVert}\right|
$$

For a labeled example, the geometric margin is:

$$
\gamma^{(i)} = y^{(i)}\frac{w^Tx^{(i)}+b}{\lVert w\rVert}
$$

This includes the true label so correctly classified points have positive margin.

Important correction:

```text
This is the signed distance from a point to the decision boundary,
not the distance from the margin to the decision boundary.
```

---

## 13. Margin Violations in Gradient Code

### Question

What does this condition mean?

$$
y^{(i)}(w^Tx^{(i)}+b) < 1
$$

### Answer

This means the point violates the SVM margin requirement.

It includes two cases:

1. The point is correctly classified but inside the margin.
2. The point is misclassified.

In code, this is usually:

```python
margins = y * prediction_score(X, w, b)
violating = margins < 1
```

Only violating points contribute to the hinge-loss gradient.

If:

$$
y^{(i)}(w^Tx^{(i)}+b) \ge 1
$$

then the point has zero hinge loss and does not push the boundary through the hinge term.

---

## 14. SVM Loss Function in Code

Your code used the prediction score:

```python
def prediction_score(X, w, b):
    return X @ w + b
```

This computes:

$$
Xw+b
$$

For one example:

$$
w^Tx+b
$$

The soft-margin SVM loss is:

```python
def soft_margin_svm_loss(X, y, w, b, C):
    margins = y * prediction_score(X, w, b)
    losses = np.maximum(0, 1 - margins)
    return 0.5 * np.dot(w, w) + C * np.mean(losses)
```

Math:

$$
J(w,b) = \frac{1}{2}w^Tw + C\cdot\frac{1}{m}\sum_{i=1}^{m}\max(0,1-y^{(i)}(w^Tx^{(i)}+b))
$$

Since:

$$
w^Tw = \lVert w\rVert^2
$$

this is the same as:

$$
J(w,b) = \frac{1}{2}\lVert w\rVert^2 + C\cdot\frac{1}{m}\sum_{i=1}^{m}\max(0,1-y^{(i)}(w^Tx^{(i)}+b))
$$

---

## 15. Dot Product Meaning

### Question

What does `np.dot(w, w)` mean?

### Answer

It computes:

$$
w^Tw
$$

If:

```text
w = [3, 4]
```

Then:

$$
w^Tw = 3(3) + 4(4) = 9 + 16 = 25
$$

So:

$$
\lVert w\rVert = \sqrt{25}=5
$$

and:

$$
\lVert w\rVert^2 = 25
$$

That is why:

```python
np.dot(w, w)
```

represents the squared length of the weight vector.

---

## 16. Linear Boundary Plot

For a 2D model:

$$
w^Tx+b = w_0x_1 + w_1x_2 + b
$$

The decision boundary is:

$$
w_0x_1+w_1x_2+b=0
$$

Solving for \(x_2\):

$$
x_2 = -\frac{w_0x_1+b}{w_1}
$$

That is why your plot used:

```python
decision_boundary = -(w[0] * x1_vals + b) / w[1]
```

The margin lines are:

$$
w^Tx+b=1
$$

and:

$$
w^Tx+b=-1
$$

Solving for \(x_2\):

$$
x_2 = \frac{1 - w_0x_1 - b}{w_1}
$$

$$
x_2 = \frac{-1 - w_0x_1 - b}{w_1}
$$

---

## 17. Final SVM Checklist

You should be able to explain:

- What \(w^Tx+b\) means.
- How \(\operatorname{sign}(w^Tx+b)\) gives the predicted class.
- Why positive score does not automatically mean correct prediction.
- What the decision boundary is.
- Difference between decision boundary and margin.
- Why SVM wants the widest margin.
- What the functional margin \(y^{(i)}(w^Tx^{(i)}+b)\) means.
- Difference between \(>0\) and \(\ge 1\).
- What hinge loss measures.
- What the soft-margin objective does.
- How \(C\) controls strictness vs flexibility.
- Why minimizing \(\lVert w\rVert^2\) increases margin width.
- Difference between hard-margin and soft-margin SVM.
- What support vectors are.
- What the signed distance formula means.
- How the SVM loss maps to your code.

