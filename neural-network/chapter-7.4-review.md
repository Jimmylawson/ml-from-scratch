# Chapter 7.4 Review: Vectorization Over Training Examples

Vectorization means processing many examples at once using matrix operations.

Instead of looping over examples one by one, we put examples into a matrix and compute the whole batch together.

This is faster and closer to how neural networks are implemented in practice.

## 1. One Example vs Many Examples

For one example:

$$
z^{[l]}=W^{[l]}a^{[l-1]}+b^{[l]}
$$

For many examples, we collect activations into a matrix:

$$
A^{[l-1]}=
\begin{bmatrix}
| & | & & | \\
a^{[l-1](1)} & a^{[l-1](2)} & \cdots & a^{[l-1](B)} \\
| & | & & |
\end{bmatrix}
$$

Here `B` is the batch size.

Each column is one training example.

## 2. Input Matrix

If the input has `d` features and the batch has `B` examples, then:

$$
X\in\mathbb{R}^{d\times B}
$$

Example:

- 4 input features
- 20 examples in the batch

Then:

$$
X\in\mathbb{R}^{4\times 20}
$$

The `20` comes from the number of examples in the batch, not from the number of neurons.

## 3. Vectorized Forward Pass

For layer `l`:

$$
Z^{[l]}=W^{[l]}A^{[l-1]}+b^{[l]}
$$

$$
A^{[l]}=g^{[l]}(Z^{[l]})
$$

This computes the entire batch at once.

## 4. Shape Rules

Let:

- `n^[l-1]`: number of units in previous layer
- `n^[l]`: number of units in current layer
- `B`: batch size

Then:

$$
A^{[l-1]}\in\mathbb{R}^{n^{[l-1]}\times B}
$$

$$
W^{[l]}\in\mathbb{R}^{n^{[l]}\times n^{[l-1]}}
$$

$$
b^{[l]}\in\mathbb{R}^{n^{[l]}\times 1}
$$

$$
Z^{[l]}\in\mathbb{R}^{n^{[l]}\times B}
$$

$$
A^{[l]}\in\mathbb{R}^{n^{[l]}\times B}
$$

## 5. Shape Example

Suppose:

- Input features: 4
- Batch size: 20
- Hidden layer neurons: 6

Then:

$$
X\in\mathbb{R}^{4\times 20}
$$

$$
W^{[1]}\in\mathbb{R}^{6\times 4}
$$

Matrix multiplication:

$$
W^{[1]}X=(6,4)(4,20)=(6,20)
$$

Bias shape:

$$
b^{[1]}\in\mathbb{R}^{6\times 1}
$$

So:

$$
Z^{[1]}=W^{[1]}X+b^{[1]}
$$

Shape:

$$
Z^{[1]}=(6,20)+(6,1)
$$

By broadcasting:

$$
Z^{[1]}\in\mathbb{R}^{6\times 20}
$$

Each of the 20 columns gets the same bias vector added.

## 6. Broadcasting Bias

Bias has shape:

$$
b^{[l]}\in\mathbb{R}^{n^{[l]}\times 1}
$$

Raw score matrix has shape:

$$
Z^{[l]}\in\mathbb{R}^{n^{[l]}\times B}
$$

When adding them:

$$
(n^{[l]},B)+(n^{[l]},1)\rightarrow(n^{[l]},B)
$$

The bias column is copied across all batch columns.

## 7. Vectorized Backpropagation

For a mini-batch, collect error terms into:

$$
\Delta^{[l]}\in\mathbb{R}^{n^{[l]}\times B}
$$

Then the weight gradient is:

$$
dW^{[l]}=\frac{1}{B}\Delta^{[l]}(A^{[l-1]})^T
$$

Shape check:

$$
\Delta^{[l]}\in\mathbb{R}^{n^{[l]}\times B}
$$

$$
(A^{[l-1]})^T\in\mathbb{R}^{B\times n^{[l-1]}}
$$

Therefore:

$$
dW^{[l]}\in\mathbb{R}^{n^{[l]}\times n^{[l-1]}}
$$

which matches the shape of `W^[l]`.

## 8. Bias Gradient

The bias gradient averages the error over the batch:

$$
db^{[l]}=\frac{1}{B}\sum_{i=1}^{B}\delta^{[l](i)}
$$

The shape should remain:

$$
db^{[l]}\in\mathbb{R}^{n^{[l]}\times 1}
$$

In NumPy, this is why we often use `keepdims=True`.

Without `keepdims=True`, NumPy may return shape:

$$
(n^{[l]},)
$$

With `keepdims=True`, it keeps:

$$
(n^{[l]},1)
$$

## 9. Why Vectorization Matters

Vectorization is important because:

- It avoids slow Python loops.
- It uses optimized matrix operations.
- It matches mini-batch training.
- It makes the math cleaner.
- It is how neural networks are usually implemented.

## 10. Mini-Batch Reminder

If we have 100 examples and batch size 5:

$$
\frac{100}{5}=20
$$

So one epoch has 20 mini-batches.

Each mini-batch performs one update.

## 11. Main Takeaway

Chapter 7.4 says:

- Put examples as columns in a matrix.
- Use matrix multiplication to process a batch at once.
- Bias vectors broadcast across columns.
- Vectorized forward pass uses `Z = WA + b`.
- Vectorized backprop averages gradients across the batch.
- Shape checking is essential.
