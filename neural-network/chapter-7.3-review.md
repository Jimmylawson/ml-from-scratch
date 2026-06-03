# Chapter 7.3 Review: Backpropagation

Backpropagation is the algorithm used to compute gradients efficiently in neural networks.

Forward propagation computes predictions.

Backpropagation computes how much each parameter contributed to the loss.

Gradient descent then uses those gradients to update the parameters.

## 1. The Simple One-Neuron Setup

For one neuron:

$$
z=wx+b
$$

$$
a=g(z)
$$

$$
J=\frac{1}{2}(a-y)^2
$$

Meaning:

- `x`: input
- `w`: weight
- `b`: bias
- `z`: raw score
- `a`: activation/output
- `y`: true target
- `J`: loss

## 2. What Backpropagation Wants

We want gradients:

$$
\frac{\partial J}{\partial w}
$$

and:

$$
\frac{\partial J}{\partial b}
$$

These tell us how changing `w` or `b` changes the loss.

## 3. Chain Rule for the Weight

Since:

$$
w \rightarrow z \rightarrow a \rightarrow J
$$

we use the chain rule:

$$
\frac{\partial J}{\partial w}
=
\frac{\partial J}{\partial a}
\frac{\partial a}{\partial z}
\frac{\partial z}{\partial w}
$$

Each piece is simple.

First:

$$
\frac{\partial J}{\partial a}=a-y
$$

Second:

$$
\frac{\partial a}{\partial z}=g'(z)
$$

Third:

$$
\frac{\partial z}{\partial w}=x
$$

So:

$$
\frac{\partial J}{\partial w}=(a-y)g'(z)x
$$

## 4. Chain Rule for the Bias

Since:

$$
b \rightarrow z \rightarrow a \rightarrow J
$$

we use:

$$
\frac{\partial J}{\partial b}
=
\frac{\partial J}{\partial a}
\frac{\partial a}{\partial z}
\frac{\partial z}{\partial b}
$$

The pieces are:

$$
\frac{\partial J}{\partial a}=a-y
$$

$$
\frac{\partial a}{\partial z}=g'(z)
$$

$$
\frac{\partial z}{\partial b}=1
$$

So:

$$
\frac{\partial J}{\partial b}=(a-y)g'(z)
$$

## 5. ReLU Derivative

For ReLU:

$$
g(z)=\max(z,0)
$$

The derivative is:

$$
g'(z)=
\begin{cases}
1, & z>0 \\
0, & z<0
\end{cases}
$$

So if `z` is negative, the neuron is inactive and its gradient becomes zero.

That means:

$$
\frac{\partial J}{\partial w}=0
$$

and:

$$
\frac{\partial J}{\partial b}=0
$$

for that example.

## 6. Numerical Example

Let:

$$
x=4
$$

$$
w=3
$$

$$
b=-1
$$

$$
y=6
$$

Forward pass:

$$
z=wx+b
$$

$$
z=3(4)-1=11
$$

ReLU activation:

$$
a=\operatorname{ReLU}(11)=11
$$

Loss:

$$
J=\frac{1}{2}(a-y)^2
$$

$$
J=\frac{1}{2}(11-6)^2
$$

$$
J=\frac{25}{2}=12.5
$$

Since `z > 0`, ReLU derivative is 1:

$$
g'(z)=1
$$

Weight gradient:

$$
\frac{\partial J}{\partial w}=(a-y)g'(z)x
$$

$$
\frac{\partial J}{\partial w}=(11-6)(1)(4)=20
$$

Bias gradient:

$$
\frac{\partial J}{\partial b}=(a-y)g'(z)
$$

$$
\frac{\partial J}{\partial b}=(11-6)(1)=5
$$

## 7. Gradient Descent Update

After backprop computes gradients, gradient descent updates the parameters:

$$
w := w - \alpha\frac{\partial J}{\partial w}
$$

$$
b := b - \alpha\frac{\partial J}{\partial b}
$$

Backprop computes the gradients.

Gradient descent uses the gradients.

## 8. Backpropagation in Layers

For layer `l`, define the error term:

$$
\delta^{[l]}=\frac{\partial J}{\partial z^{[l]}}
$$

This means: how much the raw score of layer `l` affects the loss.

For a hidden layer:

$$
\delta^{[l]}=
\left((W^{[l+1]})^T\delta^{[l+1]}\right)
\odot g'^{[l]}(z^{[l]})
$$

Meaning:

- `delta^[l+1]` is the error from the next layer.
- `(W^[l+1])^T` moves that error backward.
- `g'(z^[l])` accounts for the activation derivative.
- `odot` means elementwise multiplication.

## 9. Gradients for a Layer

Once we have `delta^[l]`, the gradients are:

$$
\frac{\partial J}{\partial W^{[l]}}
=
\delta^{[l]}(a^{[l-1]})^T
$$

$$
\frac{\partial J}{\partial b^{[l]}}
=
\delta^{[l]}
$$

Why this makes sense:

- The error term tells how wrong the layer output was.
- The previous activation tells what input caused that error.
- Multiplying them gives the weight gradient.

## 10. Why We Cache Values From Forward Pass

During forward propagation we compute:

$$
z^{[l]}
$$

and:

$$
a^{[l]}
$$

Backprop needs these values later.

For example, to compute:

$$
g'^{[l]}(z^{[l]})
$$

we need the old `z^[l]` from the forward pass.

That is why neural network code usually stores a cache during forward propagation.

## 11. Main Takeaway

Chapter 7.3 says:

- Forward pass computes predictions.
- Loss measures prediction error.
- Backprop uses the chain rule to compute gradients.
- Gradient descent updates weights and biases.
- The key object is the error term `delta`.
- Backprop moves from output layer back toward earlier layers.
