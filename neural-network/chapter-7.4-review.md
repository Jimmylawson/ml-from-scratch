# Chapter 7.4 Review: Vectorization over Training Examples

This note reviews Section 7.4. It explains how to process many training examples efficiently using matrix operations.

---

## 1. Why Vectorization Matters

Earlier sections described forward propagation and backpropagation for one training example:

$$
x^{(i)}
$$

In practice, neural networks usually process several examples together in a mini-batch.

Instead of running the same calculations separately for each example, vectorization combines the examples into matrices.

This allows numerical libraries and hardware to perform the work efficiently.

The main idea is:

$$
\boxed{
\text{one example}
\longrightarrow
\text{vector quantities}
}
$$

$$
\boxed{
\text{multiple examples}
\longrightarrow
\text{matrix quantities}
}
$$

---

## 2. One Example versus Multiple Examples

For one example, the input is:

$$
x
\in
\mathbb{R}^{n^{[0]}\times1}
$$

The forward pass for one layer is:

$$
z^{[l]}
=
W^{[l]}a^{[l-1]}+b^{[l]}
$$

$$
a^{[l]}
=
g\left(z^{[l]}\right)
$$

For a mini-batch of \(B\) examples, store the examples as columns:

$$
X
=
\begin{bmatrix}
\vert & \vert & & \vert \\
x^{(1)} & x^{(2)} & \cdots & x^{(B)} \\
\vert & \vert & & \vert
\end{bmatrix}
$$

The input matrix is:

$$
\boxed{
X
\in
\mathbb{R}^{n^{[0]}\times B}
}
$$

Where:

- \(n^{[0]}\) is the number of input features.
- \(B\) is the number of examples in the mini-batch.

---

## 3. Important Convention: Examples as Columns

In these formulas, each training example is stored as a column.

Suppose:

- Each example has \(4\) features.
- The mini-batch contains \(20\) examples.

Then:

$$
X
\in
\mathbb{R}^{4\times20}
$$

Interpretation:

- \(4\) rows: one row for each feature.
- \(20\) columns: one column for each example.

The matrix looks like:

$$
X
=
\begin{bmatrix}
\vert & \vert & & \vert \\
x^{(1)} & x^{(2)} & \cdots & x^{(20)} \\
\vert & \vert & & \vert
\end{bmatrix}
$$

### NumPy convention warning

Many NumPy datasets are initially loaded with examples as rows:

$$
X_{\text{loaded}}
\in
\mathbb{R}^{20\times4}
$$

To use the column-based notation in this guide, transpose the matrix:

$$
X
=
X_{\text{loaded}}^T
$$

Then:

$$
X
\in
\mathbb{R}^{4\times20}
$$

Both conventions are valid. The important requirement is to use one convention consistently.

---

## 4. Vectorized Forward Propagation

For a mini-batch, the vectorized forward-pass equations are:

$$
\boxed{
Z^{[l]}
=
W^{[l]}A^{[l-1]}+b^{[l]}
}
$$

$$
\boxed{
A^{[l]}
=
g\left(Z^{[l]}\right)
}
$$

The capital letters indicate matrices containing values for multiple examples:

- \(z^{[l]}\): raw scores for one example.
- \(Z^{[l]}\): raw scores for multiple examples.
- \(a^{[l]}\): activations for one example.
- \(A^{[l]}\): activations for multiple examples.

The input matrix is:

$$
A^{[0]}=X
$$

---

## 5. Shape Rule for Matrix Multiplication

The matrix multiplication rule is:

$$
(a,b)(b,c)
=
(a,c)
$$

The inner dimensions must match:

$$
(a,\cancel{b})(\cancel{b},c)
=
(a,c)
$$

The outer dimensions remain.

This rule is essential when debugging a neural network implementation.

---

## 6. Worked Shape Example

Suppose:

- Each input has \(4\) features.
- The mini-batch contains \(20\) examples.
- The first hidden layer has \(6\) neurons.

The input matrix is:

$$
X
\in
\mathbb{R}^{4\times20}
$$

The first-layer weights are:

$$
W^{[1]}
\in
\mathbb{R}^{6\times4}
$$

The first-layer bias is:

$$
b^{[1]}
\in
\mathbb{R}^{6\times1}
$$

Calculate:

$$
Z^{[1]}
=
W^{[1]}X+b^{[1]}
$$

Check the multiplication:

$$
(6,4)(4,20)
=
(6,20)
$$

Therefore:

$$
Z^{[1]}
\in
\mathbb{R}^{6\times20}
$$

Interpretation:

- \(6\) rows: one raw score for each hidden neuron.
- \(20\) columns: one column for each example.

Apply ReLU:

$$
A^{[1]}
=
\operatorname{ReLU}\left(Z^{[1]}\right)
$$

ReLU changes values, not dimensions:

$$
A^{[1]}
\in
\mathbb{R}^{6\times20}
$$

---

## 7. Bias Broadcasting

In the previous example:

$$
W^{[1]}X
\in
\mathbb{R}^{6\times20}
$$

but:

$$
b^{[1]}
\in
\mathbb{R}^{6\times1}
$$

The bias column is added to each example column:

$$
\begin{bmatrix}
\vert \\
b^{[1]} \\
\vert
\end{bmatrix}
$$

is reused across all \(20\) examples.

Conceptually:

$$
(6,20)+(6,1)
=
(6,20)
$$

NumPy performs this operation through **broadcasting**.

Each neuron has one bias value, and that same bias is applied to every example in the mini-batch.

---

## 8. General Forward-Pass Shapes

For a network layer \(l\):

$$
W^{[l]}
\in
\mathbb{R}^{n^{[l]}\times n^{[l-1]}}
$$

$$
b^{[l]}
\in
\mathbb{R}^{n^{[l]}\times1}
$$

For a mini-batch of \(B\) examples:

$$
A^{[l-1]}
\in
\mathbb{R}^{n^{[l-1]}\times B}
$$

Then:

$$
Z^{[l]}
=
W^{[l]}A^{[l-1]}+b^{[l]}
$$

Check the multiplication:

$$
\left(
n^{[l]},
n^{[l-1]}
\right)
\left(
n^{[l-1]},
B
\right)
=
\left(
n^{[l]},
B
\right)
$$

Therefore:

$$
\boxed{
Z^{[l]}
\in
\mathbb{R}^{n^{[l]}\times B}
}
$$

The activation has the same shape:

$$
\boxed{
A^{[l]}
\in
\mathbb{R}^{n^{[l]}\times B}
}
$$

---

## 9. Two-Layer Forward-Pass Example

Suppose:

- The input has \(4\) features.
- The mini-batch contains \(20\) examples.
- Hidden layer 1 has \(6\) neurons.
- The output layer has \(3\) neurons.

The input is:

$$
A^{[0]}
=
X
\in
\mathbb{R}^{4\times20}
$$

### Layer 1

$$
W^{[1]}
\in
\mathbb{R}^{6\times4}
$$

$$
b^{[1]}
\in
\mathbb{R}^{6\times1}
$$

$$
Z^{[1]}
=
W^{[1]}A^{[0]}+b^{[1]}
\in
\mathbb{R}^{6\times20}
$$

$$
A^{[1]}
=
\operatorname{ReLU}\left(Z^{[1]}\right)
\in
\mathbb{R}^{6\times20}
$$

### Output layer

$$
W^{[2]}
\in
\mathbb{R}^{3\times6}
$$

$$
b^{[2]}
\in
\mathbb{R}^{3\times1}
$$

$$
Z^{[2]}
=
W^{[2]}A^{[1]}+b^{[2]}
\in
\mathbb{R}^{3\times20}
$$

$$
A^{[2]}
=
g\left(Z^{[2]}\right)
\in
\mathbb{R}^{3\times20}
$$

Each column of:

$$
A^{[2]}
$$

contains the network output for one example.

---

## 10. Vectorized Backpropagation

For one example, define:

$$
\delta^{[l]}
=
\frac{\partial J}{\partial z^{[l]}}
$$

For a mini-batch of \(B\) examples, collect the error signals into columns:

$$
\Delta^{[l]}
=
\begin{bmatrix}
\vert & \vert & & \vert \\
\delta^{[l](1)}
&
\delta^{[l](2)}
&
\cdots
&
\delta^{[l](B)} \\
\vert & \vert & & \vert
\end{bmatrix}
$$

The shape is:

$$
\Delta^{[l]}
\in
\mathbb{R}^{n^{[l]}\times B}
$$

The vectorized hidden-layer error signal is:

$$
\boxed{
\Delta^{[l]}
=
\left(W^{[l+1]}\right)^T
\Delta^{[l+1]}
\odot
g'\left(Z^{[l]}\right)
}
$$

Where:

$$
\odot
$$

means element-by-element multiplication.

---

## 11. Vectorized Weight Gradient

For one example:

$$
\frac{\partial J}{\partial W^{[l]}}
=
\delta^{[l]}
\left(a^{[l-1]}\right)^T
$$

For a mini-batch:

$$
\boxed{
\frac{\partial J}{\partial W^{[l]}}
=
\frac{1}{B}
\Delta^{[l]}
\left(A^{[l-1]}\right)^T
}
$$

Check the shapes:

$$
\Delta^{[l]}
\in
\mathbb{R}^{n^{[l]}\times B}
$$

$$
\left(A^{[l-1]}\right)^T
\in
\mathbb{R}^{B\times n^{[l-1]}}
$$

Multiply:

$$
\left(
n^{[l]},
B
\right)
\left(
B,
n^{[l-1]}
\right)
=
\left(
n^{[l]},
n^{[l-1]}
\right)
$$

Therefore:

$$
\frac{\partial J}{\partial W^{[l]}}
\in
\mathbb{R}^{n^{[l]}\times n^{[l-1]}}
$$

This matches the weight matrix shape:

$$
W^{[l]}
\in
\mathbb{R}^{n^{[l]}\times n^{[l-1]}}
$$

---

## 12. Vectorized Bias Gradient

Each neuron has one bias value.

For a mini-batch, each example produces a bias-gradient contribution. We average those contributions across the example columns:

$$
\boxed{
\frac{\partial J}{\partial b^{[l]}}
=
\frac{1}{B}
\sum_{i=1}^{B}
\delta^{[l](i)}
}
$$

The output shape is:

$$
\frac{\partial J}{\partial b^{[l]}}
\in
\mathbb{R}^{n^{[l]}\times1}
$$

This matches:

$$
b^{[l]}
\in
\mathbb{R}^{n^{[l]}\times1}
$$

In NumPy, this is commonly written with:

```python
db = np.mean(delta, axis=1, keepdims=True)
```

Where:

- `axis=1` averages across example columns.
- `keepdims=True` preserves the column-vector shape.

---

## 13. Why Divide by the Mini-Batch Size?

The mini-batch loss is usually the average loss:

$$
J
=
\frac{1}{B}
\sum_{i=1}^{B}
J^{(i)}
$$

Therefore, the gradient is also the average:

$$
\nabla J
=
\frac{1}{B}
\sum_{i=1}^{B}
\nabla J^{(i)}
$$

Dividing by:

$$
B
$$

keeps the gradient scale more consistent when the mini-batch size changes.

Without the division, a mini-batch with more examples would usually produce a larger summed gradient simply because it contains more examples.

---

## 14. Vectorized Gradient-Descent Update

After calculating the average mini-batch gradients:

$$
\frac{\partial J}{\partial W^{[l]}}
$$

and:

$$
\frac{\partial J}{\partial b^{[l]}}
$$

update the parameters:

$$
\boxed{
W^{[l]}
:=
W^{[l]}
-
\alpha
\frac{\partial J}{\partial W^{[l]}}
}
$$

$$
\boxed{
b^{[l]}
:=
b^{[l]}
-
\alpha
\frac{\partial J}{\partial b^{[l]}}
}
$$

---

## 15. Mini-Batch Training Loop

A typical mini-batch training loop is:

1. Shuffle the training examples.
2. Split the examples into mini-batches.
3. Run vectorized forward propagation for one mini-batch.
4. Calculate the average mini-batch loss.
5. Run vectorized backpropagation.
6. Update the parameters.
7. Repeat for all mini-batches.
8. Start another epoch.

One epoch means:

$$
\boxed{
\text{one complete pass through the training dataset}
}
$$

If:

$$
n=1000
$$

and:

$$
B=100
$$

then the number of parameter updates per epoch is:

$$
\frac{1000}{100}
=
10
$$

---

## 16. Connection to a NumPy MNIST Project

For MNIST:

$$
28\times28
=
784
$$

Each image becomes a flattened input vector:

$$
x^{(i)}
\in
\mathbb{R}^{784}
$$

For a mini-batch of \(B\) images:

$$
X
\in
\mathbb{R}^{784\times B}
$$

Suppose the hidden layer has \(64\) neurons:

$$
W^{[1]}
\in
\mathbb{R}^{64\times784}
$$

$$
b^{[1]}
\in
\mathbb{R}^{64\times1}
$$

$$
Z^{[1]}
=
W^{[1]}X+b^{[1]}
\in
\mathbb{R}^{64\times B}
$$

$$
A^{[1]}
=
\operatorname{ReLU}\left(Z^{[1]}\right)
\in
\mathbb{R}^{64\times B}
$$

MNIST has \(10\) classes:

$$
W^{[2]}
\in
\mathbb{R}^{10\times64}
$$

$$
b^{[2]}
\in
\mathbb{R}^{10\times1}
$$

$$
Z^{[2]}
=
W^{[2]}A^{[1]}+b^{[2]}
\in
\mathbb{R}^{10\times B}
$$

After softmax:

$$
A^{[2]}
\in
\mathbb{R}^{10\times B}
$$

Each output column contains the predicted probabilities for one image.

---

## 17. Common Mistakes

### Mistake 1: Reversing feature and example dimensions

If examples are stored as columns:

$$
X
\in
\mathbb{R}^{\text{features}\times\text{examples}}
$$

For \(4\) features and \(20\) examples:

$$
X
\in
\mathbb{R}^{4\times20}
$$

not:

$$
X
\in
\mathbb{R}^{20\times4}
$$

unless you intentionally use the row-based convention.

### Mistake 2: Reversing the weight shape

Correct:

$$
W^{[l]}
\in
\mathbb{R}^{n^{[l]}\times n^{[l-1]}}
$$

The number of rows equals the number of neurons in the current layer.

### Mistake 3: Expecting ReLU to change the shape

ReLU changes values only:

$$
A^{[l]}
=
\operatorname{ReLU}\left(Z^{[l]}\right)
$$

Therefore:

$$
\operatorname{shape}\left(A^{[l]}\right)
=
\operatorname{shape}\left(Z^{[l]}\right)
$$

### Mistake 4: Forgetting that bias is broadcast

The bias is:

$$
b^{[l]}
\in
\mathbb{R}^{n^{[l]}\times1}
$$

It is added to every example column.

### Mistake 5: Summing gradients without averaging

If the loss is an average, the gradients should also be averaged:

$$
\frac{1}{B}
$$

---

## 18. Mental Model

Use this mental model:

### One example

$$
x
\longrightarrow
z^{[l]}
\longrightarrow
a^{[l]}
$$

### Multiple examples

$$
X
\longrightarrow
Z^{[l]}
\longrightarrow
A^{[l]}
$$

### One-example backpropagation

$$
\delta^{[l]}
\left(a^{[l-1]}\right)^T
$$

### Mini-batch backpropagation

$$
\frac{1}{B}
\Delta^{[l]}
\left(A^{[l-1]}\right)^T
$$

Vectorization does not change the learning idea. It performs the same calculations for several examples simultaneously.

---

## 19. Self-Test Questions

1. Why do we use vectorization?
2. If each example has \(5\) features and the mini-batch contains \(32\) examples, what is the column-based shape of:

$$
X
$$

3. If a layer receives \(5\) inputs and has \(8\) neurons, what is the shape of:

$$
W^{[l]}
$$

4. If:

$$
W^{[l]}
\in
\mathbb{R}^{8\times5}
$$

and:

$$
A^{[l-1]}
\in
\mathbb{R}^{5\times32}
$$

what is the shape of:

$$
Z^{[l]}
$$

5. Why does:

$$
b^{[l]}
\in
\mathbb{R}^{8\times1}
$$

work when:

$$
Z^{[l]}
\in
\mathbb{R}^{8\times32}
$$

6. What is the difference between:

$$
z^{[l]}
$$

and:

$$
Z^{[l]}
$$

7. Why do we divide mini-batch gradients by:

$$
B
$$

8. Why does:

$$
\frac{\partial J}{\partial W^{[l]}}
$$

have the same shape as:

$$
W^{[l]}
$$

9. What does one epoch mean?
10. If a dataset has \(1200\) examples and the mini-batch size is \(100\), how many parameter updates occur in one epoch?

---

## 20. Final Summary

Vectorization processes several examples using matrices instead of repeating one-example calculations in Python loops.

The core forward-pass equations are:

$$
\boxed{
Z^{[l]}
=
W^{[l]}A^{[l-1]}+b^{[l]}
}
$$

$$
\boxed{
A^{[l]}
=
g\left(Z^{[l]}\right)
}
$$

The core backpropagation equations are:

$$
\boxed{
\Delta^{[l]}
=
\left(W^{[l+1]}\right)^T
\Delta^{[l+1]}
\odot
g'\left(Z^{[l]}\right)
}
$$

$$
\boxed{
\frac{\partial J}{\partial W^{[l]}}
=
\frac{1}{B}
\Delta^{[l]}
\left(A^{[l-1]}\right)^T
}
$$

$$
\boxed{
\frac{\partial J}{\partial b^{[l]}}
=
\frac{1}{B}
\sum_{i=1}^{B}
\delta^{[l](i)}
}
$$

