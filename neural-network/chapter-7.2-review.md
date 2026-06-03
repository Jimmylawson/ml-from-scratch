# Chapter 7.2 Review: Neural Networks

This section introduces the basic building block of neural networks: a neuron.

A neuron takes inputs, computes a weighted sum, adds a bias, and applies a nonlinear activation function.

## 1. Single Neuron

For one input vector `x`, a single neuron first computes a raw score:

$$
z=w^Tx+b
$$

Then it applies an activation function:

$$
a=g(z)
$$

So the full neuron is:

$$
a=g(w^Tx+b)
$$

Meaning:

- `x`: input features
- `w`: weight vector
- `b`: bias
- `z`: raw score, also called pre-activation
- `g`: activation function
- `a`: activation/output of the neuron

## 2. ReLU Activation

Andrew introduces ReLU:

$$
\operatorname{ReLU}(t)=\max(t,0)
$$

Piecewise:

$$
\operatorname{ReLU}(t)=
\begin{cases}
t, & t>0 \\
0, & t\le 0
\end{cases}
$$

So if:

$$
t=5
$$

then:

$$
\operatorname{ReLU}(5)=5
$$

If:

$$
t=-3
$$

then:

$$
\operatorname{ReLU}(-3)=0
$$

ReLU creates a kink in the function. That kink is what introduces nonlinearity.

## 3. Single-Neuron Model

A one-neuron model can be written as:

$$
h_\theta(x)=\operatorname{ReLU}(w^Tx+b)
$$

where:

$$
\theta=(w,b)
$$

This means the parameters are the weight vector and the bias.

## 4. Why Activation Functions Matter

If we stack only linear layers, the result is still linear.

Suppose we have two linear layers:

$$
z^{[1]}=W^{[1]}x+b^{[1]}
$$

$$
z^{[2]}=W^{[2]}z^{[1]}+b^{[2]}
$$

Substitute the first equation into the second:

$$
z^{[2]}=W^{[2]}(W^{[1]}x+b^{[1]})+b^{[2]}
$$

Distribute:

$$
z^{[2]}=(W^{[2]}W^{[1]})x+(W^{[2]}b^{[1]}+b^{[2]})
$$

This is still just:

$$
z^{[2]}=Wx+b
$$

So without nonlinear activation functions, multiple layers collapse into one linear layer.

That is why neural networks need activation functions like ReLU.

## 5. Layer Notation

For layer `l`, the raw score is:

$$
z^{[l]}=W^{[l]}a^{[l-1]}+b^{[l]}
$$

The activation is:

$$
a^{[l]}=g^{[l]}(z^{[l]})
$$

The input layer is:

$$
a^{[0]}=x
$$

Meaning:

- `a^[0]` is the original input vector.
- `z^[l]` is the pre-activation for layer `l`.
- `a^[l]` is the activation output of layer `l`.
- `W^[l]` maps activations from layer `l-1` into layer `l`.
- `b^[l]` shifts the raw scores in layer `l`.

## 6. Shape Rules for One Example

Let:

- `n^[l-1]`: number of units in the previous layer
- `n^[l]`: number of units in the current layer

Then:

$$
W^{[l]}\in\mathbb{R}^{n^{[l]}\times n^{[l-1]}}
$$

$$
b^{[l]}\in\mathbb{R}^{n^{[l]}\times 1}
$$

$$
z^{[l]}\in\mathbb{R}^{n^{[l]}\times 1}
$$

$$
a^{[l]}\in\mathbb{R}^{n^{[l]}\times 1}
$$

The reason is matrix multiplication:

$$
W^{[l]}a^{[l-1]}+b^{[l]}
$$

Shape example:

$$
(3,2)(2,1)+(3,1)=(3,1)
$$

Meaning:

- Weight matrix has 3 rows and 2 columns.
- Input activation has 2 rows and 1 column.
- Output has 3 rows and 1 column.
- Bias must also be 3 rows and 1 column.

## 7. Example With Concrete Shapes

Suppose:

- Input has 4 features.
- Hidden layer has 5 neurons.
- Output layer has 3 neurons.

Then:

$$
a^{[0]}=x\in\mathbb{R}^{4\times 1}
$$

Layer 1:

$$
W^{[1]}\in\mathbb{R}^{5\times 4}
$$

$$
b^{[1]}\in\mathbb{R}^{5\times 1}
$$

$$
z^{[1]}\in\mathbb{R}^{5\times 1}
$$

$$
a^{[1]}\in\mathbb{R}^{5\times 1}
$$

Layer 2:

$$
W^{[2]}\in\mathbb{R}^{3\times 5}
$$

$$
b^{[2]}\in\mathbb{R}^{3\times 1}
$$

$$
z^{[2]}\in\mathbb{R}^{3\times 1}
$$

$$
a^{[2]}\in\mathbb{R}^{3\times 1}
$$

## 8. Hidden Layers

A hidden layer is an intermediate representation.

It is called hidden because we do not directly observe what it should be. The model learns it.

The hidden layer transforms the input into a new representation that the next layer can use.

## 9. Forward Pass

A forward pass means moving from input to prediction.

For a neural network:

$$
a^{[0]}=x
$$

$$
z^{[1]}=W^{[1]}a^{[0]}+b^{[1]}
$$

$$
a^{[1]}=g^{[1]}(z^{[1]})
$$

$$
z^{[2]}=W^{[2]}a^{[1]}+b^{[2]}
$$

$$
a^{[2]}=g^{[2]}(z^{[2]})
$$

The final activation is the prediction.

## 10. Main Takeaway

Chapter 7.2 says:

- A neuron computes a raw score and applies an activation.
- ReLU makes the model nonlinear.
- Without activations, stacked linear layers collapse into one linear model.
- Shapes matter because every layer is matrix multiplication.
- Forward propagation is the process of computing predictions from input to output.
