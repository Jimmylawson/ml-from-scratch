# Chapter 7.1 Review: Supervised Learning With Nonlinear Models

This section is the bridge from the models you already built into neural networks.

Before Chapter 7, most models were linear in the parameters. Chapter 7 starts asking what happens when the model itself becomes nonlinear.

## 1. Supervised Learning Setup

In supervised learning, we have training examples:

$$
\{(x^{(i)}, y^{(i)})\}_{i=1}^{n}
$$

Each example has:

- `x`: input features
- `y`: target/output
- `h_theta(x)`: model prediction
- `theta`: parameters we learn

The goal is to learn parameters that make predictions close to the targets.

## 2. Linear Models From Earlier Chapters

In linear regression, the hypothesis was:

$$
h_\theta(x)=\theta^T x
$$

Meaning:

- `x` is the input feature vector.
- `theta` is the parameter vector.
- `theta^T x` is the dot product.
- The output is a weighted sum of the input features.

This model is linear because the prediction is a straight weighted combination of the inputs.

## 3. Feature Maps Still Keep the Model Linear in Parameters

In kernels and feature engineering, we also saw:

$$
h_\theta(x)=\theta^T\phi(x)
$$

Here, `phi(x)` is a transformed feature vector.

Example:

$$
\phi(x)=
\begin{bmatrix}
1 \\
x \\
x^2
\end{bmatrix}
$$

Then:

$$
h_\theta(x)=\theta_0+\theta_1x+\theta_2x^2
$$

This can be nonlinear in `x`, but it is still linear in `theta`.

That means the parameters still appear as a weighted sum.

## 4. What Changes in Chapter 7

Neural networks are different because they can be nonlinear in both:

- the input `x`
- the parameters `theta`

A simple nonlinear model is:

$$
h_\theta(x)=\operatorname{ReLU}(w^Tx+b)
$$

where:

$$
\operatorname{ReLU}(t)=\max(t,0)
$$

The linear part is:

$$
w^Tx+b
$$

The nonlinear part is:

$$
\max(t,0)
$$

This nonlinear activation is what lets neural networks learn more complex patterns than a single linear model.

## 5. Single-Example Loss

For one training example, Andrew defines the squared loss as:

$$
J^{(i)}(\theta)=\frac{1}{2}\left(h_\theta(x^{(i)})-y^{(i)}\right)^2
$$

Meaning:

- `h_theta(x^(i))` is the model prediction for example `i`.
- `y^(i)` is the true target.
- The difference is the prediction error.
- Squaring makes errors positive and penalizes large mistakes more.
- The `1/2` makes derivatives cleaner.

If:

$$
h_\theta(x^{(i)})=10
$$

and:

$$
y^{(i)}=7
$$

then:

$$
J^{(i)}(\theta)=\frac{1}{2}(10-7)^2
$$

$$
J^{(i)}(\theta)=\frac{9}{2}=4.5
$$

## 6. Dataset Cost

For the full training set, the cost is the average of all single-example losses:

$$
J(\theta)=\frac{1}{n}\sum_{i=1}^{n}J^{(i)}(\theta)
$$

Substitute the single-example loss:

$$
J(\theta)=\frac{1}{n}\sum_{i=1}^{n}\frac{1}{2}\left(h_\theta(x^{(i)})-y^{(i)}\right)^2
$$

Equivalent form:

$$
J(\theta)=\frac{1}{2n}\sum_{i=1}^{n}\left(h_\theta(x^{(i)})-y^{(i)}\right)^2
$$

This is still mean squared error style loss, but the model `h_theta` may now be nonlinear.

## 7. Gradient Descent

Gradient descent updates parameters by moving opposite the gradient:

$$
\theta := \theta - \alpha \nabla_\theta J(\theta)
$$

Meaning:

- `alpha` is the learning rate.
- `nabla_theta J(theta)` tells the direction of steepest increase.
- We subtract because we want to reduce the loss.

The learning rate controls the step size:

- Too small: training is slow.
- Too large: training can diverge.
- Reasonable value: training converges toward a lower cost.

## 8. Batch Gradient Descent

Batch gradient descent uses the entire training set before each update:

$$
\theta := \theta - \alpha \nabla_\theta J(\theta)
$$

Since:

$$
J(\theta)=\frac{1}{n}\sum_{i=1}^{n}J^{(i)}(\theta)
$$

The gradient is:

$$
\nabla_\theta J(\theta)=\frac{1}{n}\sum_{i=1}^{n}\nabla_\theta J^{(i)}(\theta)
$$

So batch GD averages the gradient over all examples before updating.

## 9. Stochastic Gradient Descent

SGD samples one training example and updates using that example only:

$$
\theta := \theta - \alpha \nabla_\theta J^{(j)}(\theta)
$$

where `j` is a randomly sampled training example index.

SGD is noisier, but often faster per update.

## 10. Mini-Batch SGD

Mini-batch SGD samples a small batch of `B` examples:

$$
\{j_1,j_2,\ldots,j_B\}
$$

Then updates using the average gradient over that mini-batch:

$$
\theta := \theta - \frac{\alpha}{B}\sum_{k=1}^{B}\nabla_\theta J^{(j_k)}(\theta)
$$

This is the most common approach in deep learning.

It is a balance between:

- Batch GD: stable but slow
- SGD: fast but noisy
- Mini-batch: efficient and stable enough

## 11. Epoch vs Iteration

An iteration is one parameter update.

An epoch is one full pass through the training set.

Example:

- Training examples: 100
- Batch size: 5

Number of mini-batches in one epoch:

$$
\frac{100}{5}=20
$$

So one epoch has 20 updates.

## 12. Main Takeaway

Chapter 7.1 says:

- We are moving from linear models to nonlinear models.
- The loss function idea stays familiar.
- Gradient descent still trains the model.
- Mini-batch SGD becomes important because neural networks are usually trained on many examples.

The important shift is not the cost function.

The important shift is the model:

$$
h_\theta(x)
$$

can now be nonlinear and much more expressive.
