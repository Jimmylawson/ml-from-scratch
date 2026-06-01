# Chapter 7.1 Review: Supervised Learning with Nonlinear Models

This note reviews Section 7.1 before moving into neural networks. The section is a bridge: the model becomes more flexible, but the training workflow remains familiar.

---

## 1. Main Goal of Section 7.1

Earlier supervised-learning models often used a linear prediction function:

$$
h_\theta(x) = \theta^T x
$$

or a linear model applied to manually transformed features:

$$
h_\theta(x) = \theta^T \phi(x)
$$

Section 7.1 prepares us to learn more general models:

$$
h_\theta(x)
$$

These models may be nonlinear in both the input features and the parameters. Neural networks are the main example.

The important idea is:

$$
\text{the model changes, but the training workflow remains the same}
$$

The workflow is:

$$
\text{prediction}
\longrightarrow
\text{loss}
\longrightarrow
\text{gradient}
\longrightarrow
\text{parameter update}
\longrightarrow
\text{repeat}
$$

---

## 2. Review of the Linear Model

The linear model is:

$$
h_\theta(x) = \theta^T x
$$

Where:

- \(x\) is the input feature vector.
- \(\theta\) is the parameter vector learned during training.
- \(h_\theta(x)\) is the prediction.
- \(\theta^T x\) is the dot product between the parameters and the features.

If:

$$
x =
\begin{bmatrix}
x_1 \\
x_2 \\
\vdots \\
x_d
\end{bmatrix}
\qquad
\theta =
\begin{bmatrix}
\theta_1 \\
\theta_2 \\
\vdots \\
\theta_d
\end{bmatrix}
$$

then:

$$
\theta^T x
=
\theta_1x_1
+
\theta_2x_2
+
\cdots
+
\theta_dx_d
$$

### Example

If:

$$
x =
\begin{bmatrix}
2 \\
4
\end{bmatrix}
\qquad
\theta =
\begin{bmatrix}
3 \\
5
\end{bmatrix}
$$

then:

$$
h_\theta(x)
=
\theta^T x
=
3(2) + 5(4)
=
26
$$

The model outputs one prediction:

$$
26
$$

---

## 3. Feature Maps

A feature map transforms the original input into a new representation:

$$
x \longrightarrow \phi(x)
$$

The model becomes:

$$
h_\theta(x) = \theta^T\phi(x)
$$

For example, if:

$$
x =
\begin{bmatrix}
x_1 \\
x_2
\end{bmatrix}
$$

we could define:

$$
\phi(x) =
\begin{bmatrix}
x_1 \\
x_2 \\
x_1^2 \\
x_1x_2 \\
x_2^2
\end{bmatrix}
$$

Then:

$$
h_\theta(x)
=
\theta_1x_1
+
\theta_2x_2
+
\theta_3x_1^2
+
\theta_4x_1x_2
+
\theta_5x_2^2
$$

This can produce a curved relationship with the original features:

$$
x
$$

However, the model is still linear in its parameters:

$$
\theta_1,\theta_2,\ldots,\theta_5
$$

Each parameter is only multiplied by a feature and added to the result.

### Connection to Kernels

Feature maps are more general than kernels.

A kernel can calculate similarities in a transformed feature space without explicitly constructing a potentially large feature vector:

$$
\phi(x)
$$

The kernel trick is useful when manually constructing the transformed feature space would be computationally expensive.

---

## 4. Why Neural Networks Are Different

A simple neural-network neuron can use:

$$
h_\theta(x)
=
\operatorname{ReLU}(w^Tx+b)
$$

where:

$$
\operatorname{ReLU}(t) = \max(t,0)
$$

The first step is a linear score:

$$
w^Tx+b
$$

The ReLU activation then applies a nonlinear transformation:

$$
\max(w^Tx+b,0)
$$

ReLU creates a bend or kink in the function. This is one way neural networks move beyond purely linear models.

Section 7.1 does not yet explain neural-network layers in depth. It prepares the optimization machinery required to train them.

---

## 5. Training Examples and Notation

Suppose the training dataset contains:

$$
n
$$

examples:

$$
\left\{
\left(x^{(i)}, y^{(i)}\right)
\right\}_{i=1}^{n}
$$

For the \(i\)-th example:

- \(x^{(i)}\) is the input feature vector.
- \(y^{(i)}\) is the true target.
- \(h_\theta(x^{(i)})\) is the model's prediction.

The superscript:

$$
(i)
$$

means the example number. It is not an exponent.

---

## 6. Loss for One Training Example

For one example, Section 7.1 defines the squared loss:

$$
J^{(i)}(\theta)
=
\frac{1}{2}
\left(
h_\theta(x^{(i)})
-
y^{(i)}
\right)^2
$$

Where:

- \(J^{(i)}(\theta)\) is the loss for one training example.
- \(h_\theta(x^{(i)})\) is the prediction.
- \(y^{(i)}\) is the actual target.
- The difference is the prediction error.

### Why Square the Error?

The squared error:

$$
\left(
h_\theta(x^{(i)})
-
y^{(i)}
\right)^2
$$

is useful because:

1. Positive and negative errors do not cancel each other.
2. Large mistakes receive a larger penalty.
3. The squared function is differentiable, which helps optimization.

### Why Multiply by One Half?

The factor:

$$
\frac{1}{2}
$$

is used for calculus convenience.

When differentiating a squared term:

$$
\frac{d}{dz}z^2 = 2z
$$

the:

$$
\frac{1}{2}
$$

cancels the:

$$
2
$$

This makes the gradient expression cleaner. It does not change the location of the minimum.

---

## 7. Average Cost Across the Dataset

The loss for one example is:

$$
J^{(i)}(\theta)
$$

The average cost across the entire training dataset is:

$$
J(\theta)
=
\frac{1}{n}
\sum_{i=1}^{n}
J^{(i)}(\theta)
$$

Where:

- \(n\) is the number of training examples.
- The summation adds the losses for all training examples.
- The factor \(1/n\) calculates the average.

The model should perform well across the dataset, not only on one example.

### Important Distinction

Single-example loss:

$$
J^{(i)}(\theta)
$$

Average dataset cost:

$$
J(\theta)
$$

Do not reverse these two definitions.

---

## 8. Gradient Descent

Gradient descent updates the parameters using:

$$
\theta
:=
\theta
-
\alpha
\nabla_\theta J(\theta)
$$

Where:

- \(\theta\) is the parameter vector.
- \(\alpha\) is the learning rate.
- \(\nabla_\theta J(\theta)\) is the gradient of the cost with respect to the parameters.

The gradient points in the direction of the steepest increase in cost.

We subtract it because we want the cost to decrease:

$$
\text{new parameters}
=
\text{old parameters}
-
\text{step toward increasing cost}
$$

### Learning Rate

The learning rate:

$$
\alpha
$$

controls the size of each parameter update. It does not control the loss directly.

- If \(\alpha\) is too small, training is slow.
- If \(\alpha\) is too large, training may overshoot the minimum or diverge.

---

## 9. Batch Gradient Descent

Batch gradient descent calculates the average gradient using every training example before updating the parameters:

$$
\theta
:=
\theta
-
\frac{\alpha}{n}
\sum_{i=1}^{n}
\nabla_\theta J^{(i)}(\theta)
$$

### Process

1. Use all \(n\) training examples.
2. Calculate each example's gradient.
3. Average the gradients.
4. Perform one parameter update.

Batch gradient descent produces stable updates. However, each update can be expensive when the dataset is large.

---

## 10. Stochastic Gradient Descent

Stochastic gradient descent, commonly abbreviated as SGD, uses one randomly selected training example for each parameter update:

$$
\theta
:=
\theta
-
\alpha
\nabla_\theta J^{(j)}(\theta)
$$

Where:

$$
j
$$

is the index of the selected training example.

### Process

1. Select one example.
2. Calculate its gradient.
3. Update the parameters immediately.
4. Repeat.

SGD can update the model quickly, but its cost curve is noisier. One randomly selected example may contain noise or may not represent the full dataset well.

---

## 11. Mini-Batch Stochastic Gradient Descent

Mini-batch SGD uses a small group of training examples before each parameter update:

$$
\theta
:=
\theta
-
\frac{\alpha}{B}
\sum_{k=1}^{B}
\nabla_\theta J^{(j_k)}(\theta)
$$

Where:

- \(B\) is the batch size.
- \(j_k\) is the index of one selected training example in the batch.
- The gradients from the batch are averaged before updating the parameters.

### Example

If:

$$
B = 32
$$

the model:

1. Processes 32 examples.
2. Calculates their gradients.
3. Averages the gradients.
4. Updates the parameters once.

Mini-batch SGD is commonly used to train neural networks because it:

- Is computationally efficient.
- Uses parallel matrix operations effectively.
- Produces more stable updates than processing only one example.
- Requires less work per update than full batch gradient descent.

---

## 12. Epochs and Number of Updates

An epoch is one complete pass through the training dataset.

If:

$$
n = 1000
$$

and:

$$
B = 100
$$

then the number of mini-batches in one epoch is:

$$
\frac{n}{B}
=
\frac{1000}{100}
=
10
$$

Therefore:

$$
1\text{ epoch}
=
10\text{ mini-batches}
$$

If training runs for:

$$
5
$$

epochs, then:

$$
5 \times 10 = 50
$$

parameter updates are performed.

### General Formula

Assuming the dataset divides evenly by the batch size:

$$
\text{updates per epoch}
=
\frac{\text{number of training examples}}
{\text{batch size}}
$$

---

## 13. Comparison of Optimization Methods

| Method | Examples used before one update | Main advantage | Main tradeoff |
| --- | ---: | --- | --- |
| Batch gradient descent | Entire dataset | Stable gradient estimate | Expensive update for large datasets |
| Stochastic gradient descent | One example | Frequent, inexpensive updates | Noisy update direction |
| Mini-batch SGD | Small group of examples | Efficient and reasonably stable | Batch size must be chosen |

---

## 14. Common Mistakes to Avoid

### Mistake 1: Confusing Parameters and Targets

The parameters are:

$$
\theta
$$

The true target for example \(i\) is:

$$
y^{(i)}
$$

### Mistake 2: Reversing Single-Example Loss and Dataset Cost

Single-example loss:

$$
J^{(i)}(\theta)
$$

Average dataset cost:

$$
J(\theta)
$$

### Mistake 3: Confusing a Mini-Batch with an Epoch

A mini-batch is a small subset of training examples.

An epoch is one complete pass through the full training dataset.

### Mistake 4: Saying the Learning Rate Controls the Loss

The learning rate:

$$
\alpha
$$

controls the update step size. It influences training behavior, but it does not directly define the loss.

### Mistake 5: Saying a Feature Map Is Only for Kernels

A feature map:

$$
\phi(x)
$$

is any transformation of the input features.

A kernel is a technique that can avoid explicitly computing a large feature map.

---

## 15. Final Mental Model

Section 7.1 prepares us for neural networks by showing that the familiar optimization workflow still applies:

$$
\boxed{
\text{prediction}
\longrightarrow
\text{loss}
\longrightarrow
\text{gradient}
\longrightarrow
\text{parameter update}
}
$$

The model may become nonlinear and more complex, but we still:

1. Make predictions.
2. Measure the error.
3. Calculate gradients.
4. Update parameters.
5. Repeat over mini-batches and epochs.

---

## 16. Self-Test Questions

1. What is the role of \(x\), \(\theta\), and \(\theta^Tx\)?
2. Why can \(\theta^T\phi(x)\) be nonlinear in the original input but linear in the parameters?
3. What is the difference between \(J^{(i)}(\theta)\) and \(J(\theta)\)?
4. Why do we square the prediction error?
5. What does \(\alpha\) control?
6. Why do we subtract the gradient?
7. How many examples does batch gradient descent use before one update?
8. Why is the SGD cost curve usually noisier?
9. What does \(B\) represent in mini-batch SGD?
10. What is an epoch?
11. If \(n=2000\), \(B=100\), and training runs for 3 epochs, how many parameter updates occur?
12. Why is mini-batch SGD commonly used for neural-network training?

