# Chapter 7 Full Review: Neural Networks, Shapes, Forward Prop, and Backprop

This is the main review sheet for Chapter 7.

The goal is not just to memorize formulas. The goal is to know what every formula is doing, what each shape means, and how the vectorized code matches the math.

## 1. The Big Picture

A neural network does three main things:

1. Forward propagation: make predictions.
2. Backpropagation: compute gradients.
3. Gradient descent: update weights and biases.

The loop is:

```text
initialize parameters
repeat:
    forward propagation
    compute loss
    backpropagation
    update parameters
```

For your MNIST project:

```text
input image -> hidden layer -> output layer
```

Example architecture:

```text
784 -> 128 -> 10
```

Meaning:

```text
784 input features = pixels in one 28x28 image
128 hidden neurons = chosen by you
10 output neurons = digit classes 0 through 9
```

The 60000 training images are not neurons. They are examples.

## 2. Single Neuron

A single neuron first computes:

$$
z = w^T x + b
$$

Then applies an activation:

$$
a = g(z)
$$

Meaning:

```text
x = input features
w = weights
b = bias
z = pre-activation/raw score
g = activation function
a = activation/output
```

The activation function makes the model nonlinear.

Without nonlinear activations, many layers collapse into one linear layer.

## 3. Layer Formula

For layer `l`:

$$
z^{[l]} = W^{[l]}a^{[l-1]} + b^{[l]}
$$

$$
a^{[l]} = g^{[l]}(z^{[l]})
$$

The input layer is:

$$
a^{[0]} = x
$$

Meaning:

```text
a^[l-1] = input from previous layer
W^[l] = weights for current layer
b^[l] = bias for current layer
z^[l] = pre-activation for current layer
a^[l] = activation/output of current layer
```

This notation works for any number of layers:

```text
l = 1, 2, 3, ..., L
```

So yes, this is the general multi-layer notation.

## 4. Why Examples Are Columns

The formula:

$$
z^{[l]} = W^{[l]}a^{[l-1]} + b^{[l]}
$$

uses:

$$
W a
$$

For one MNIST image:

```text
one image = 784 pixel values
```

So one input example is treated as a column vector:

$$
x \in \mathbb{R}^{784 \times 1}
$$

If the first hidden layer has 128 neurons:

$$
W^{[1]} \in \mathbb{R}^{128 \times 784}
$$

Then:

$$
W^{[1]}x = (128,784)(784,1) = (128,1)
$$

That gives 128 hidden-neuron outputs for one image.

For many examples, we stack examples as columns:

$$
X \in \mathbb{R}^{784 \times m}
$$

For MNIST:

```text
X_train shape = (784, 60000)
```

Meaning:

```text
784 rows = pixels/features
60000 columns = images/examples
```

This is why we transpose after reshaping.

Before transpose:

```python
X_train.shape == (60000, 784)
```

Meaning:

```text
60000 rows = images
784 columns = pixels
```

After transpose:

```python
X_train = X_train.T
X_train.shape == (784, 60000)
```

Meaning:

```text
784 rows = pixels
60000 columns = images
```

Now the formula works:

$$
Z^{[1]} = W^{[1]}X + b^{[1]}
$$

Shape:

$$
(128,784)(784,60000) + (128,1) = (128,60000)
$$

## 5. MNIST Data Shapes

Raw Keras MNIST gives:

```text
X_train: (60000, 28, 28)
y_train: (60000,)
X_test:  (10000, 28, 28)
y_test:  (10000,)
```

Meaning:

```text
60000 training images
10000 test images
each image is 28 by 28 pixels
each label is one digit from 0 to 9
```

Flatten images:

```python
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
```

because:

```text
28 * 28 = 784
```

Normalize:

```python
X_train = X_train / 255.0
X_test = X_test / 255.0
```

Pixel values originally go from:

```text
0 to 255
```

After dividing by 255:

```text
0.0 to 1.0
```

This makes training more stable because the inputs are smaller.

Then transpose:

```python
X_train = X_train.T
X_test = X_test.T
```

Final shapes:

```text
X_train: (784, 60000)
X_test:  (784, 10000)
```

## 6. What One-Hot Encoding Does

Original labels look like:

```python
y_train[0] == 5
```

That is good for humans, but the output layer has 10 neurons.

The network output for one image has 10 numbers:

```text
one score/probability for digit 0
one score/probability for digit 1
...
one score/probability for digit 9
```

So label `5` becomes:

```text
[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
```

The `1` is at index 5.

Function:

```python
def one_hot(y, num_classes=10):
    one_hot_y = np.zeros((num_classes, y.size))
    one_hot_y[y, np.arange(y.size)] = 1
    return one_hot_y
```

Line by line:

```python
one_hot_y = np.zeros((num_classes, y.size))
```

Creates a zero matrix.

For MNIST:

```text
num_classes = 10
y.size = 60000
```

So:

```text
one_hot_y shape = (10, 60000)
```

Then:

```python
one_hot_y[y, np.arange(y.size)] = 1
```

puts a `1` in the correct class row for each example column.

Example:

```python
y = np.array([5, 0, 4])
np.arange(y.size) == [0, 1, 2]
```

Then NumPy does:

```python
one_hot_y[5, 0] = 1
one_hot_y[0, 1] = 1
one_hot_y[4, 2] = 1
```

Meaning:

```text
column 0 = digit 5
column 1 = digit 0
column 2 = digit 4
```

Final label shape:

```text
Y_train: (10, 60000)
Y_test:  (10, 10000)
```

Important nuance:

```text
y_train = original labels, shape (60000,)
Y_train = one-hot labels, shape (10, 60000)
```

Use `Y_train` for training loss/backprop.

Use `y_train` for accuracy comparison.

## 7. Parameter Shapes

Rule:

$$
W^{[l]} \in \mathbb{R}^{n^{[l]} \times n^{[l-1]}}
$$

$$
b^{[l]} \in \mathbb{R}^{n^{[l]} \times 1}
$$

Meaning:

```text
W rows = neurons in current layer
W columns = neurons/features from previous layer
b rows = neurons in current layer
```

For:

```text
784 -> 128 -> 10
```

Layer 1:

```text
W1: (128, 784)
b1: (128, 1)
```

Layer 2:

```text
W2: (10, 128)
b2: (10, 1)
```

Why not `W1 = (128, 60000)`?

Because 60000 is the number of training examples, not the number of input features.

Each hidden neuron looks at one image at a time.

One image has 784 pixels, so each hidden neuron needs 784 weights.

The same `W1` is reused for every image.

## 8. Forward Propagation for Two Layers

For your project:

$$
Z^{[1]} = W^{[1]}X + b^{[1]}
$$

$$
A^{[1]} = ReLU(Z^{[1]})
$$

$$
Z^{[2]} = W^{[2]}A^{[1]} + b^{[2]}
$$

$$
A^{[2]} = softmax(Z^{[2]})
$$

Code:

```python
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = RELU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2
```

Shape check:

```text
X:  (784, m)
W1: (128, 784)
b1: (128, 1)
Z1: (128, m)
A1: (128, m)

W2: (10, 128)
b2: (10, 1)
Z2: (10, m)
A2: (10, m)
```

Softmax changes values, not shape.

So if:

```text
Z2 = (10, 60000)
```

then:

```text
A2 = (10, 60000)
```

## 9. ReLU and Softmax

ReLU:

$$
ReLU(z)=\max(0,z)
$$

Code:

```python
def RELU(z):
    return np.maximum(0, z)
```

Use ReLU in hidden layers.

Softmax:

$$
softmax(z_i)=\frac{e^{z_i}}{\sum_j e^{z_j}}
$$

Code:

```python
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)
```

Use softmax in the final output layer for multi-class classification.

Why not sigmoid for MNIST output?

Sigmoid treats each output independently.

Softmax makes the 10 classes compete and outputs probabilities that sum to 1.

MNIST has exactly one correct digit per image, so softmax is the right output activation.

## 10. Why Softmax Subtracts the Max

This line:

```python
exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
```

is for numerical stability.

`np.exp(1000)` is too large and can overflow.

Subtracting the maximum value in each column keeps the largest value at 0:

```text
[2, 1, 4] - 4 = [-2, -3, 0]
```

Then exponentials are safe:

```text
exp([-2, -3, 0])
```

The softmax result is mathematically equivalent.

`axis=0` means column by column.

`keepdims=True` keeps the result shaped like:

```text
(1, m)
```

instead of:

```text
(m,)
```

This makes broadcasting clear when subtracting from:

```text
z shape = (10, m)
```

## 11. Broadcasting

Broadcasting means NumPy automatically stretches a smaller array to work with a bigger one.

Example:

```text
Z1: (128, 60000)
b1: (128, 1)
```

When you do:

```python
Z1 = W1 @ X + b1
```

NumPy broadcasts `b1` across all 60000 columns.

Conceptually:

```text
same bias vector added to every example
```

Bias does not have one separate value per example.

Bias has one value per neuron.

## 12. Loss and Cost

Loss is error for one example.

Cost is average loss over many examples.

For multi-class classification with softmax:

$$
L^{(i)} = -\sum_{k=1}^{10} y_k^{(i)}\log(a_k^{(i)})
$$

Since `Y` is one-hot, this just selects the log probability of the correct class.

Cost:

$$
J = \frac{1}{m}\sum_{i=1}^{m}L^{(i)}
$$

This is why gradients have:

$$
\frac{1}{m}
$$

The gradients are averaged over the current batch.

If using full-batch training:

```text
m = 60000
```

If using mini-batches:

```text
m = batch size, like 64
```

## 13. The Error Term Delta

The key object in backprop is:

$$
\delta^{[l]} = \frac{\partial J}{\partial z^{[l]}}
$$

In code, this is usually written as:

```text
dZ^[l]
```

Meaning:

```text
dZ^[l] tells how much the loss changes when Z^[l] changes
```

It is the layer's error term.

Important shape rule:

```text
dZ^[l] has the same shape as Z^[l]
```

So:

```text
Z2:  (10, m)
A2:  (10, m)
dZ2: (10, m)

Z1:  (128, m)
A1:  (128, m)
dZ1: (128, m)
```

## 14. Output Layer Backprop

For softmax output with cross-entropy loss:

$$
dZ^{[2]} = A^{[2]} - Y
$$

Code:

```python
dZ2 = A2 - Y
```

Shape:

```text
A2:  (10, m)
Y:   (10, m)
dZ2: (10, m)
```

This works because both prediction and true label are 10-by-m matrices.

Each column is one example's output error.

## 15. Weight Gradient Formula

General formula:

$$
dW^{[l]}=\frac{1}{m}dZ^{[l]}(A^{[l-1]})^T
$$

For layer 2:

$$
dW^{[2]}=\frac{1}{m}dZ^{[2]}(A^{[1]})^T
$$

Code:

```python
dW2 = (1 / m) * dZ2 @ A1.T
```

Shape:

```text
dZ2:  (10, m)
A1.T: (m, 128)
dW2:  (10, 128)
```

This matches:

```text
W2: (10, 128)
```

Why `A1.T`?

Because `W2` connects hidden-layer activations to output neurons.

Each weight gradient asks:

```text
output error * hidden activation that caused it
```

The transpose makes matrix multiplication produce the same shape as `W2`.

For layer 1:

$$
dW^{[1]}=\frac{1}{m}dZ^{[1]}X^T
$$

because:

$$
A^{[0]} = X
$$

Code:

```python
dW1 = (1 / m) * dZ1 @ X.T
```

Shape:

```text
dZ1: (128, m)
X.T: (m, 784)
dW1: (128, 784)
```

This matches:

```text
W1: (128, 784)
```

## 16. Bias Gradient Formula

General formula:

$$
db^{[l]}=\frac{1}{m}\sum_{i=1}^{m}dZ^{[l](i)}
$$

Code:

```python
db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
```

For layer 2:

```python
db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
```

Shape:

```text
dZ2: (10, m)
db2: (10, 1)
```

For layer 1:

```python
db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
```

Shape:

```text
dZ1: (128, m)
db1: (128, 1)
```

Why does summing make `(10, m)` become `(10, 1)`?

Because:

```python
np.sum(dZ2, axis=1, keepdims=True)
```

sums across columns/examples.

Example:

```text
[[1, 2, 3],
 [4, 5, 6]]
```

Summing with `axis=1, keepdims=True` gives:

```text
[[6],
 [15]]
```

Shape changes:

```text
(2, 3) -> (2, 1)
```

`keepdims=True` keeps the bias as a column vector.

Without it:

```text
(10,)
```

With it:

```text
(10, 1)
```

That matches `b2`.

## 17. Hidden Layer Backprop

General hidden-layer formula:

$$
dZ^{[l]} = (W^{[l+1]})^T dZ^{[l+1]} \odot g'^{[l]}(Z^{[l]})
$$

For your layer 1:

$$
dZ^{[1]} = (W^{[2]})^T dZ^{[2]} \odot ReLU'(Z^{[1]})
$$

Code:

```python
dZ1 = W2.T @ dZ2 * relu_derivative(Z1)
```

Shape:

```text
W2:    (10, 128)
W2.T:  (128, 10)
dZ2:   (10, m)

W2.T @ dZ2 = (128, m)
```

ReLU derivative:

```text
relu_derivative(Z1): (128, m)
```

Elementwise multiply:

```text
dZ1: (128, m)
```

Conceptually:

```text
1. Move output error backward through W2.T.
2. Multiply by ReLU derivative.
3. Keep gradient only where hidden neuron was active.
```

ReLU derivative:

$$
ReLU'(z)=
\begin{cases}
1, & z > 0 \\
0, & z \le 0
\end{cases}
$$

Code:

```python
def relu_derivative(z):
    return z > 0
```

In NumPy, `True` acts like `1` and `False` acts like `0`.

## 18. Complete Two-Layer Backprop

Code:

```python
def backprop(Z1, A1, Z2, A2, W2, X, Y):
    m = Y.shape[1]

    dZ2 = A2 - Y
    dW2 = (1 / m) * dZ2 @ A1.T
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = W2.T @ dZ2 * relu_derivative(Z1)
    dW1 = (1 / m) * dZ1 @ X.T
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2
```

Mapping to formulas:

```text
A2 = A^[2]
A1 = A^[1]
X  = A^[0]

dZ2 = dZ^[2]
dW2 = dW^[2]
db2 = db^[2]

dZ1 = dZ^[1]
dW1 = dW^[1]
db1 = db^[1]
```

## 19. Gradient Descent Update

General update:

$$
W^{[l]} := W^{[l]} - \alpha dW^{[l]}
$$

$$
b^{[l]} := b^{[l]} - \alpha db^{[l]}
$$

Code:

```python
def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2

    return W1, b1, W2, b2
```

Shapes do not change during update:

```text
W1:  (128, 784)
dW1: (128, 784)

b1:  (128, 1)
db1: (128, 1)

W2:  (10, 128)
dW2: (10, 128)

b2:  (10, 1)
db2: (10, 1)
```

`alpha` is a scalar learning rate.

It gets broadcast over every value in the gradient.

## 20. Mini-Batch Training

You do not need a separate backprop formula for mini-batches.

The same `forward_prop`, `backprop`, and `update_params` work.

Only the number of examples changes.

Full batch:

```text
X: (784, 60000)
Y: (10, 60000)
m = 60000
```

Mini-batch of 64:

```text
X_batch: (784, 64)
Y_batch: (10, 64)
m = 64
```

The formulas are identical.

Mini-batch loop:

```python
for start in range(0, m, B):
    end = start + B
    X_batch = X_shuffled[:, start:end]
    Y_batch = Y_shuffled[:, start:end]
```

`range(0, m, B)` means:

```text
start at 0
stop before m
jump by B
```

Example:

```python
range(0, 10, 4)
```

gives:

```text
0, 4, 8
```

Then:

```python
X[:, 0:64]
```

means:

```text
all rows/features
columns 0 through 63
```

For MNIST:

```text
all 784 pixels for the first 64 images
```

## 21. Prediction and Accuracy

After training, use forward prop only.

Training:

```text
forward_prop -> backprop -> update_params
```

Prediction/testing:

```text
forward_prop -> argmax -> accuracy
```

Use:

```python
_, _, _, A2_test = forward_prop(W1, b1, W2, b2, X_test)
test_predictions = np.argmax(A2_test, axis=0)
test_accuracy = np.mean(test_predictions == y_test)
```

Why not backprop for prediction?

Backprop computes gradients for learning.

Prediction only needs the output probabilities.

Why compare with `y_test`, not `Y_test`?

Because predictions are class numbers:

```text
[7, 2, 1, 0, ...]
```

Original labels are also class numbers:

```text
[7, 2, 1, 0, ...]
```

One-hot labels are matrices:

```text
Y_test shape = (10, 10000)
```

So accuracy compares predictions with original labels.

## 22. General L-Layer Formulas

For any layer `l`:

Forward:

$$
Z^{[l]} = W^{[l]}A^{[l-1]} + b^{[l]}
$$

$$
A^{[l]} = g^{[l]}(Z^{[l]})
$$

Output error:

$$
dZ^{[L]} = A^{[L]} - Y
$$

Hidden-layer error:

$$
dZ^{[l]} = (W^{[l+1]})^T dZ^{[l+1]} \odot g'^{[l]}(Z^{[l]})
$$

Weight gradient:

$$
dW^{[l]} = \frac{1}{m}dZ^{[l]}(A^{[l-1]})^T
$$

Bias gradient:

$$
db^{[l]} = \frac{1}{m}\sum dZ^{[l]}
$$

Update:

$$
W^{[l]} := W^{[l]} - \alpha dW^{[l]}
$$

$$
b^{[l]} := b^{[l]} - \alpha db^{[l]}
$$

## 23. General Shape Rules

Let:

```text
n^[l] = number of neurons in layer l
m = number of examples in the current batch
```

Then:

```text
A^[l-1]: (n^[l-1], m)
W^[l]:   (n^[l], n^[l-1])
b^[l]:   (n^[l], 1)

Z^[l]:   (n^[l], m)
A^[l]:   (n^[l], m)
dZ^[l]:  (n^[l], m)

dW^[l]:  (n^[l], n^[l-1])
db^[l]:  (n^[l], 1)
```

The most important rules:

```text
dZ has the same shape as Z
dW has the same shape as W
db has the same shape as b
```

## 24. Common Confusions to Avoid

`60000` is not a layer size.

It is the number of examples.

`784` is the number of input features for one MNIST image.

`10` is the number of output classes.

`128` is a hidden-layer size you choose.

`X_train.T` is used because the formulas use examples as columns.

`Y_train` is one-hot and used for backprop.

`y_train` is original labels and used for accuracy.

`A2` and `Y` both have shape `(10, m)`.

`dZ2 = A2 - Y` also has shape `(10, m)`.

`db2` is not `(10, m)`.

It is `(10, 1)` because bias has one value per output neuron, not one value per example.

`keepdims=True` keeps bias gradients shaped like column vectors.

`A1.T` appears in `dW2` because `dW2` must match `W2`.

`X.T` appears in `dW1` because `X = A^[0]`.

Backprop for mini-batches is the same as backprop for full batches.

Only `m` changes.

## 25. Your Mental Checklist

Before coding any neural network layer, ask:

```text
How many neurons are in the previous layer?
How many neurons are in the current layer?
How many examples are in this batch?
```

Then write:

```text
W = (current neurons, previous neurons)
b = (current neurons, 1)
A_prev = (previous neurons, examples)
Z = (current neurons, examples)
A = (current neurons, examples)
```

During backprop:

```text
dZ = same as Z
dW = same as W
db = same as b
```

If a shape error happens, check those three rules first.

