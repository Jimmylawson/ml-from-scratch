# Chapter 7.3 Review: Backpropagation

This note reviews Section 7.3. It explains how a neural network calculates gradients and how those gradients are used to update weights and biases.

---

## 1. Main Goal of Backpropagation

A neural network learns through four steps:

$$
\boxed{
\text{forward pass}
\longrightarrow
\text{loss}
\longrightarrow
\text{backpropagation}
\longrightarrow
\text{gradient descent update}
}
$$

During the forward pass, the network calculates a prediction:

$$
\hat{y}
$$

The loss function measures the difference between the prediction and the target:

$$
J
$$

Backpropagation calculates how much each parameter contributed to the loss:

$$
\frac{\partial J}{\partial W^{[l]}}
\qquad
\text{and}
\qquad
\frac{\partial J}{\partial b^{[l]}}
$$

Gradient descent then uses these gradients to update the parameters:

$$
W^{[l]}
:=
W^{[l]}
-
\alpha
\frac{\partial J}{\partial W^{[l]}}
$$

$$
b^{[l]}
:=
b^{[l]}
-
\alpha
\frac{\partial J}{\partial b^{[l]}}
$$

Important distinction:

- **Backpropagation** calculates gradients.
- **Gradient descent** uses gradients to update parameters.

---

## 2. Review: Forward Propagation

For one neuron:

$$
z=wx+b
$$

$$
a=g(z)
$$

Where:

- \(x\) is the input.
- \(w\) is the weight.
- \(b\) is the bias.
- \(z\) is the raw score, also called the pre-activation.
- \(g\) is the activation function.
- \(a\) is the activated output.

If the activation is ReLU:

$$
a
=
\operatorname{ReLU}(z)
=
\max(z,0)
$$

For a regression example, the loss can be:

$$
J
=
\frac{1}{2}(a-y)^2
$$

The dependency path is:

$$
w
\longrightarrow
z
\longrightarrow
a
\longrightarrow
J
$$

The weight does not affect the loss directly. It affects the raw score, which affects the activation, which affects the loss.

---

## 3. Why We Need the Chain Rule

We want to calculate:

$$
\frac{\partial J}{\partial w}
$$

Because the weight affects the loss indirectly, we multiply the local derivatives along the dependency path:

$$
\boxed{
\frac{\partial J}{\partial w}
=
\frac{\partial J}{\partial a}
\cdot
\frac{\partial a}{\partial z}
\cdot
\frac{\partial z}{\partial w}
}
$$

For the bias:

$$
\boxed{
\frac{\partial J}{\partial b}
=
\frac{\partial J}{\partial a}
\cdot
\frac{\partial a}{\partial z}
\cdot
\frac{\partial z}{\partial b}
}
$$

The chain rule answers:

> If a parameter changes slightly, how will that change travel through the network and affect the final loss?

---

## 4. Local Derivatives for One Neuron

Start with:

$$
J
=
\frac{1}{2}(a-y)^2
$$

Differentiate with respect to the activation:

$$
\boxed{
\frac{\partial J}{\partial a}
=
a-y
}
$$

Now use:

$$
z=wx+b
$$

Differentiate with respect to the weight:

$$
\boxed{
\frac{\partial z}{\partial w}
=
x
}
$$

Differentiate with respect to the bias:

$$
\boxed{
\frac{\partial z}{\partial b}
=
1
}
$$

The derivative of the activation function is:

$$
\boxed{
\frac{\partial a}{\partial z}
=
g'(z)
}
$$

Combine the pieces:

$$
\boxed{
\frac{\partial J}{\partial w}
=
(a-y)g'(z)x
}
$$

$$
\boxed{
\frac{\partial J}{\partial b}
=
(a-y)g'(z)
}
$$

---

## 5. ReLU Derivative

ReLU is:

$$
\operatorname{ReLU}(z)
=
\max(z,0)
$$

Its derivative is:

$$
\boxed{
\operatorname{ReLU}'(z)
=
\begin{cases}
1, & z>0 \\
0, & z<0
\end{cases}
}
$$

At:

$$
z=0
$$

the derivative is not uniquely defined. In implementations, it is common to use:

$$
\operatorname{ReLU}'(0)=0
$$

Interpretation:

- If \(z>0\), the neuron is active and the gradient can pass backward.
- If \(z<0\), the neuron is inactive and the gradient is blocked.

---

## 6. Worked Example: Active ReLU Neuron

Suppose:

$$
x=4
\qquad
w=2
\qquad
b=-3
\qquad
y=6
$$

### Forward pass

$$
z
=
wx+b
$$

$$
z
=
(2)(4)-3
=
5
$$

Apply ReLU:

$$
a
=
\operatorname{ReLU}(5)
=
5
$$

Calculate the loss:

$$
J
=
\frac{1}{2}(a-y)^2
$$

$$
J
=
\frac{1}{2}(5-6)^2
=
\frac{1}{2}
$$

### Backward pass

Because:

$$
z>0
$$

we have:

$$
g'(z)=1
$$

The weight gradient is:

$$
\frac{\partial J}{\partial w}
=
(a-y)g'(z)x
$$

$$
\frac{\partial J}{\partial w}
=
(5-6)(1)(4)
=
-4
$$

The bias gradient is:

$$
\frac{\partial J}{\partial b}
=
(a-y)g'(z)
$$

$$
\frac{\partial J}{\partial b}
=
(5-6)(1)
=
-1
$$

### Gradient descent update

Suppose:

$$
\alpha=0.1
$$

Update the weight:

$$
w
:=
w
-
\alpha
\frac{\partial J}{\partial w}
$$

$$
w
:=
2
-
(0.1)(-4)
=
2.4
$$

Update the bias:

$$
b
:=
b
-
\alpha
\frac{\partial J}{\partial b}
$$

$$
b
:=
-3
-
(0.1)(-1)
=
-2.9
$$

The parameters changed in a direction that attempts to reduce the loss.

---

## 7. Worked Example: Inactive ReLU Neuron

Suppose:

$$
x=2
\qquad
w=-1
\qquad
b=-3
\qquad
y=5
$$

Calculate the raw score:

$$
z
=
wx+b
$$

$$
z
=
(-1)(2)-3
=
-5
$$

Apply ReLU:

$$
a
=
\operatorname{ReLU}(-5)
=
0
$$

The loss is:

$$
J
=
\frac{1}{2}(0-5)^2
=
\frac{25}{2}
$$

However:

$$
g'(z)
=
0
$$

Therefore:

$$
\frac{\partial J}{\partial w}
=
(0-5)(0)(2)
=
0
$$

$$
\frac{\partial J}{\partial b}
=
(0-5)(0)
=
0
$$

The neuron has a large loss, but gradient descent cannot update it for this example because ReLU blocks the gradient.

This is related to the **dying ReLU** problem.

---

## 8. Why Backpropagation Moves Backward

Consider a two-layer network:

$$
z^{[1]}
=
W^{[1]}x+b^{[1]}
$$

$$
a^{[1]}
=
g\left(z^{[1]}\right)
$$

$$
z^{[2]}
=
W^{[2]}a^{[1]}+b^{[2]}
$$

$$
a^{[2]}
=
\hat{y}
$$

$$
J
=
\frac{1}{2}(\hat{y}-y)^2
$$

The first-layer weights affect the loss through this path:

$$
W^{[1]}
\longrightarrow
z^{[1]}
\longrightarrow
a^{[1]}
\longrightarrow
z^{[2]}
\longrightarrow
a^{[2]}
\longrightarrow
J
$$

Therefore, the first-layer gradient must account for the complete path:

$$
\frac{\partial J}{\partial W^{[1]}}
=
\frac{\partial J}{\partial a^{[2]}}
\cdot
\frac{\partial a^{[2]}}{\partial z^{[2]}}
\cdot
\frac{\partial z^{[2]}}{\partial a^{[1]}}
\cdot
\frac{\partial a^{[1]}}{\partial z^{[1]}}
\cdot
\frac{\partial z^{[1]}}{\partial W^{[1]}}
$$

Backpropagation begins at the loss and moves backward through each dependency.

---

## 9. Intermediate Values Must Be Stored

During the forward pass, the network stores:

$$
z^{[l]}
$$

and:

$$
a^{[l]}
$$

There are two reasons:

### Reason 1: Continue the forward pass

The current activation becomes the input to the next layer:

$$
a^{[l]}
\longrightarrow
z^{[l+1]}
$$

### Reason 2: Calculate gradients during backpropagation

The activation derivative needs:

$$
z^{[l]}
$$

The weight gradient needs:

$$
a^{[l-1]}
$$

Reusing stored values is more efficient than recalculating them.

---

## 10. Error Signal Notation

Writing the complete chain-rule expression repeatedly becomes difficult as networks get deeper.

Define:

$$
\boxed{
\delta^{[l]}
=
\frac{\partial J}{\partial z^{[l]}}
}
$$

This is the **error signal** for layer \(l\).

It measures:

> How much would the loss change if the raw score of this layer changed slightly?

Once the error signal is known, the parameter gradients become:

$$
\boxed{
\frac{\partial J}{\partial W^{[l]}}
=
\delta^{[l]}
\left(a^{[l-1]}\right)^T
}
$$

$$
\boxed{
\frac{\partial J}{\partial b^{[l]}}
=
\delta^{[l]}
}
$$

---

## 11. Hidden-Layer Error Signal

For a hidden layer:

$$
\boxed{
\delta^{[l]}
=
\left(W^{[l+1]}\right)^T
\delta^{[l+1]}
\odot
g'\left(z^{[l]}\right)
}
$$

Where:

$$
\odot
$$

means element-by-element multiplication.

### Meaning of each part

The next-layer error signal:

$$
\delta^{[l+1]}
$$

contains the effect of the next layer's raw scores on the loss.

The transposed next-layer weights:

$$
\left(W^{[l+1]}\right)^T
$$

send the next-layer error backward to the current layer.

The activation derivative:

$$
g'\left(z^{[l]}\right)
$$

accounts for how the current layer's activation changed with respect to its raw score.

For ReLU, this derivative controls whether each neuron allows a gradient to pass backward.

---

## 12. Why the Weight Gradient Uses the Previous Activation

The weight gradient is:

$$
\frac{\partial J}{\partial W^{[l]}}
=
\delta^{[l]}
\left(a^{[l-1]}\right)^T
$$

The raw score is:

$$
z^{[l]}
=
W^{[l]}a^{[l-1]}+b^{[l]}
$$

Each weight multiplies an activation from the previous layer. Therefore, the effect of a weight depends on the input value that flowed through that connection.

For a single scalar neuron:

$$
\frac{\partial J}{\partial w}
=
\delta x
$$

For a full layer, the same idea becomes an outer product:

$$
\frac{\partial J}{\partial W^{[l]}}
=
\delta^{[l]}
\left(a^{[l-1]}\right)^T
$$

---

## 13. Backpropagation Summary for One Example

### Forward pass

$$
a^{[0]}=x
$$

For each layer:

$$
z^{[l]}
=
W^{[l]}a^{[l-1]}+b^{[l]}
$$

$$
a^{[l]}
=
g^{[l]}\left(z^{[l]}\right)
$$

### Backward pass

Start from the output layer:

$$
\delta^{[L]}
=
\frac{\partial J}{\partial a^{[L]}}
\odot
\left(g^{[L]}\right)'
\left(z^{[L]}\right)
$$

Then move backward through hidden layers:

$$
\delta^{[l]}
=
\left(W^{[l+1]}\right)^T
\delta^{[l+1]}
\odot
\left(g^{[l]}\right)'
\left(z^{[l]}\right)
$$

Calculate parameter gradients:

$$
\frac{\partial J}{\partial W^{[l]}}
=
\delta^{[l]}
\left(a^{[l-1]}\right)^T
$$

$$
\frac{\partial J}{\partial b^{[l]}}
=
\delta^{[l]}
$$

Update parameters:

$$
W^{[l]}
:=
W^{[l]}
-
\alpha
\frac{\partial J}{\partial W^{[l]}}
$$

$$
b^{[l]}
:=
b^{[l]}
-
\alpha
\frac{\partial J}{\partial b^{[l]}}
$$

---

## 14. Common Mistakes

### Mistake 1: Saying that backpropagation updates parameters

Correction:

- Backpropagation calculates gradients.
- Gradient descent updates parameters.

### Mistake 2: Calling the loss function backpropagation

Correction:

$$
J
$$

is the loss. Backpropagation begins from the loss and calculates gradients.

### Mistake 3: Forgetting the activation derivative

Wrong:

$$
\delta^{[l]}
=
\left(W^{[l+1]}\right)^T
\delta^{[l+1]}
$$

Correct:

$$
\delta^{[l]}
=
\left(W^{[l+1]}\right)^T
\delta^{[l+1]}
\odot
g'\left(z^{[l]}\right)
$$

### Mistake 4: Confusing the output with the error signal

The activation is:

$$
a^{[l]}
$$

The error signal is:

$$
\delta^{[l]}
=
\frac{\partial J}{\partial z^{[l]}}
$$

They have different meanings.

### Mistake 5: Ignoring the sign of the bias

If:

$$
b=-3
$$

then:

$$
wx+b
=
wx-3
$$

not:

$$
wx+3
$$

---

## 15. Mental Model

Use this mental model:

### Forward propagation

$$
\text{inputs}
\longrightarrow
\text{predictions}
$$

### Loss

$$
\text{predictions}
\longrightarrow
\text{error measurement}
$$

### Backpropagation

$$
\text{loss}
\longrightarrow
\text{gradients for every parameter}
$$

### Gradient descent

$$
\text{gradients}
\longrightarrow
\text{updated parameters}
$$

---

## 16. Self-Test Questions

1. What is the difference between forward propagation and backpropagation?
2. What is the difference between backpropagation and gradient descent?
3. Why do we need the chain rule?
4. What does this measure?

$$
\frac{\partial J}{\partial w}
$$

5. What does this measure?

$$
\delta^{[l]}
=
\frac{\partial J}{\partial z^{[l]}}
$$

6. Why does ReLU block the gradient when:

$$
z<0
$$

7. Why are intermediate values stored during the forward pass?
8. Why does the hidden-layer error signal use the transpose:

$$
\left(W^{[l+1]}\right)^T
$$

9. Why does the weight gradient use:

$$
\left(a^{[l-1]}\right)^T
$$

10. Explain the complete training loop in your own words.

---

## 17. Final Summary

Backpropagation is an efficient application of the chain rule.

It starts from the loss, moves backward through the network, and calculates the gradients needed by gradient descent.

The core formulas are:

$$
\boxed{
\delta^{[l]}
=
\left(W^{[l+1]}\right)^T
\delta^{[l+1]}
\odot
g'\left(z^{[l]}\right)
}
$$

$$
\boxed{
\frac{\partial J}{\partial W^{[l]}}
=
\delta^{[l]}
\left(a^{[l-1]}\right)^T
}
$$

$$
\boxed{
\frac{\partial J}{\partial b^{[l]}}
=
\delta^{[l]}
}
$$

