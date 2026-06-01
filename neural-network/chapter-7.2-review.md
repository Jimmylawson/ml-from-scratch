# Chapter 7.2 Review: Neural Networks

This note reviews Section 7.2. It starts with a single neuron, then builds the notation needed to understand layers and forward propagation.

---

## 1. Main Goal of Section 7.2

Section 7.1 introduced the general training workflow:

$$
\text{prediction}
\longrightarrow
\text{loss}
\longrightarrow
\text{gradient}
\longrightarrow
\text{parameter update}
$$

Section 7.2 defines the model that we want to train:

$$
h_\theta(x)
$$

Instead of using only one linear transformation, neural networks combine:

$$
\text{linear transformation}
+
\text{nonlinear activation}
$$

and repeat this pattern across layers.

The core layer equations are:

$$
\boxed{
z^{[l]}
=
W^{[l]}a^{[l-1]}
+
b^{[l]}
}
$$

$$
\boxed{
a^{[l]}
=
g\left(z^{[l]}\right)
}
$$

---

## 2. A Single Neuron

For an input vector:

$$
x \in \mathbb{R}^{d}
$$

a single neuron first calculates a raw linear score:

$$
z
=
w^Tx+b
$$

Then it applies an activation function:

$$
a
=
g(z)
$$

If the activation function is ReLU:

$$
g(z)
=
\operatorname{ReLU}(z)
=
\max(z,0)
$$

then the complete neuron is:

$$
h_\theta(x)
=
\operatorname{ReLU}
\left(
w^Tx+b
\right)
$$

Where:

- \(x\) is the input feature vector.
- \(w\) is the learned weight vector.
- \(b\) is the learned bias.
- \(w^Tx+b\) is the raw linear score.
- \(z\) is the pre-activation.
- \(g(z)\) is the activation function.
- \(a\) is the activated output.

---

## 3. Single-Neuron Example

Suppose:

$$
x
=
\begin{bmatrix}
3 \\
2
\end{bmatrix}
\qquad
w
=
\begin{bmatrix}
2 \\
-1
\end{bmatrix}
\qquad
b=-5
$$

First calculate the raw score:

$$
z
=
w^Tx+b
$$

$$
z
=
2(3)
+
(-1)(2)
-
5
$$

$$
z=-1
$$

Then apply ReLU:

$$
a
=
\operatorname{ReLU}(-1)
$$

$$
a
=
\max(-1,0)
$$

$$
a=0
$$

The neuron outputs:

$$
0
$$

---

## 4. ReLU Activation

The Rectified Linear Unit activation function is:

$$
\operatorname{ReLU}(z)
=
\max(z,0)
$$

This means:

$$
\operatorname{ReLU}(z)
=
\begin{cases}
z, & z > 0 \\
0, & z \leq 0
\end{cases}
$$

### Examples

If:

$$
z=-4
$$

then:

$$
\operatorname{ReLU}(-4)=0
$$

If:

$$
z=7
$$

then:

$$
\operatorname{ReLU}(7)=7
$$

ReLU replaces negative scores with zero and leaves positive scores unchanged.

### Why ReLU Matters

ReLU adds nonlinearity. It creates a bend or kink in the prediction function.

This allows neural networks to represent more flexible patterns than a purely linear model.

---

## 5. Why Activation Functions Are Necessary

Suppose a model contains two linear steps without an activation function:

$$
z^{[1]}
=
W^{[1]}x+b^{[1]}
$$

$$
z^{[2]}
=
W^{[2]}z^{[1]}+b^{[2]}
$$

Substitute the first equation into the second:

$$
z^{[2]}
=
W^{[2]}
\left(
W^{[1]}x+b^{[1]}
\right)
+
b^{[2]}
$$

Expand:

$$
z^{[2]}
=
W^{[2]}W^{[1]}x
+
W^{[2]}b^{[1]}
+
b^{[2]}
$$

Define:

$$
W'
=
W^{[2]}W^{[1]}
$$

and:

$$
b'
=
W^{[2]}b^{[1]}
+
b^{[2]}
$$

Then:

$$
z^{[2]}
=
W'x+b'
$$

This is still one linear transformation.

Therefore:

$$
\text{multiple linear layers without activations}
=
\text{one linear layer}
$$

With an activation function:

$$
a^{[1]}
=
g
\left(
W^{[1]}x+b^{[1]}
\right)
$$

the next layer receives a nonlinear representation:

$$
z^{[2]}
=
W^{[2]}a^{[1]}+b^{[2]}
$$

The activation function prevents the network from collapsing into one linear transformation.

---

## 6. Pre-Activations and Activations

For layer:

$$
l
$$

the raw score is:

$$
z^{[l]}
=
W^{[l]}a^{[l-1]}
+
b^{[l]}
$$

This is called the **pre-activation**.

Then:

$$
a^{[l]}
=
g
\left(
z^{[l]}
\right)
$$

This is called the **activation** or the output of layer \(l\).

The sequence is:

$$
a^{[l-1]}
\longrightarrow
W^{[l]}a^{[l-1]}+b^{[l]}
\longrightarrow
z^{[l]}
\longrightarrow
g\left(z^{[l]}\right)
\longrightarrow
a^{[l]}
$$

### Element-by-Element ReLU Example

If:

$$
z^{[1]}
=
\begin{bmatrix}
-3 \\
2 \\
5
\end{bmatrix}
$$

then:

$$
a^{[1]}
=
\operatorname{ReLU}
\left(
z^{[1]}
\right)
$$

$$
a^{[1]}
=
\begin{bmatrix}
0 \\
2 \\
5
\end{bmatrix}
$$

ReLU does not change positive values. It only replaces negative values with zero.

---

## 7. Input Layer

The original input feature vector is treated as the activation of layer zero:

$$
\boxed{
a^{[0]}=x
}
$$

Layer zero does not perform a calculation. This notation allows us to use the same formula for every later layer:

$$
z^{[l]}
=
W^{[l]}a^{[l-1]}+b^{[l]}
$$

For the first layer:

$$
z^{[1]}
=
W^{[1]}a^{[0]}+b^{[1]}
$$

Since:

$$
a^{[0]}=x
$$

we get:

$$
z^{[1]}
=
W^{[1]}x+b^{[1]}
$$

---

## 8. Understanding Matrix Shapes

The shape of a matrix tells us its number of rows and columns.

For example:

$$
\begin{bmatrix}
50 \\
110 \\
170
\end{bmatrix}
$$

has:

$$
3
$$

rows and:

$$
1
$$

column.

Its shape is:

$$
(3,1)
$$

### Matrix-Multiplication Shape Rule

The general matrix-multiplication rule is:

$$
(a,b)(b,c)
=
(a,c)
$$

The two inner dimensions must match:

$$
\cancel{b}
$$

The two outer dimensions determine the result:

$$
(a,c)
$$

For example:

$$
(3,2)(2,1)
=
(3,1)
$$

This is not symmetry. We do not multiply or add the shape values together.

---

## 9. Weight Matrix for One Layer

Suppose a layer:

- receives 2 input values
- contains 3 neurons

Each neuron requires one weight for each incoming value.

The weight matrix is:

$$
W^{[1]}
=
\begin{bmatrix}
w_{11} & w_{12} \\
w_{21} & w_{22} \\
w_{31} & w_{32}
\end{bmatrix}
$$

Its shape is:

$$
(3,2)
$$

There are:

$$
3
$$

rows because there are 3 neurons.

There are:

$$
2
$$

columns because each neuron receives 2 incoming values.

Each row contains the weights for one neuron.

---

## 10. Bias Vector for One Layer

Each neuron requires one bias value.

For a layer with 3 neurons:

$$
b^{[1]}
=
\begin{bmatrix}
b_1 \\
b_2 \\
b_3
\end{bmatrix}
$$

Its shape is:

$$
(3,1)
$$

---

## 11. Numeric Matrix-Multiplication Example

Suppose:

$$
W^{[1]}
=
\begin{bmatrix}
1 & 2 \\
3 & 4 \\
5 & 6
\end{bmatrix}
$$

and:

$$
x
=
\begin{bmatrix}
10 \\
20
\end{bmatrix}
$$

Then:

$$
W^{[1]}x
=
\begin{bmatrix}
1(10)+2(20) \\
3(10)+4(20) \\
5(10)+6(20)
\end{bmatrix}
$$

$$
W^{[1]}x
=
\begin{bmatrix}
50 \\
110 \\
170
\end{bmatrix}
$$

The dimensions are:

$$
(3,2)(2,1)
=
(3,1)
$$

Each of the three neurons produces one raw score.

---

## 12. General Shape Rules for One Input Example

Let:

$$
n^{[l]}
$$

represent the number of neurons in the current layer.

Let:

$$
n^{[l-1]}
$$

represent the number of outputs from the previous layer.

The weight matrix has shape:

$$
\boxed{
W^{[l]}
\in
\mathbb{R}^{n^{[l]} \times n^{[l-1]}}
}
$$

The bias vector has shape:

$$
\boxed{
b^{[l]}
\in
\mathbb{R}^{n^{[l]} \times 1}
}
$$

The pre-activation vector has shape:

$$
\boxed{
z^{[l]}
\in
\mathbb{R}^{n^{[l]} \times 1}
}
$$

The activation vector has shape:

$$
\boxed{
a^{[l]}
\in
\mathbb{R}^{n^{[l]} \times 1}
}
$$

The matrix calculation confirms these dimensions:

$$
\left(
n^{[l]},
n^{[l-1]}
\right)
\left(
n^{[l-1]},
1
\right)
+
\left(
n^{[l]},
1
\right)
=
\left(
n^{[l]},
1
\right)
$$

---

## 13. Two-Layer Shape Example

Suppose a network has:

- 4 input features
- 5 neurons in hidden layer 1
- 3 neurons in hidden layer 2

The input is:

$$
a^{[0]}
=
x
\in
\mathbb{R}^{4 \times 1}
$$

For layer 1:

$$
W^{[1]}
\in
\mathbb{R}^{5 \times 4}
$$

$$
b^{[1]}
\in
\mathbb{R}^{5 \times 1}
$$

$$
z^{[1]}
\in
\mathbb{R}^{5 \times 1}
$$

$$
a^{[1]}
\in
\mathbb{R}^{5 \times 1}
$$

For layer 2:

$$
W^{[2]}
\in
\mathbb{R}^{3 \times 5}
$$

$$
b^{[2]}
\in
\mathbb{R}^{3 \times 1}
$$

$$
z^{[2]}
\in
\mathbb{R}^{3 \times 1}
$$

$$
a^{[2]}
\in
\mathbb{R}^{3 \times 1}
$$

---

## 14. Input, Hidden, and Output Layers

### Input Layer

The input layer holds the original feature vector:

$$
a^{[0]}=x
$$

It is not a pre-activation or raw score.

### Hidden Layer

A hidden layer creates an intermediate learned representation:

$$
z^{[l]}
=
W^{[l]}a^{[l-1]}+b^{[l]}
$$

$$
a^{[l]}
=
g
\left(
z^{[l]}
\right)
$$

It is called hidden because:

- its values are not provided directly by the dataset
- its values are not the final prediction
- its representation is calculated internally by the network

For example, hidden layers in an image model may learn patterns such as edges and shapes.

### Output Layer

The output layer calculates the final prediction:

$$
\hat{y}
=
h_\theta(x)
$$

The output activation function depends on the task:

| Task | Common output activation |
| --- | --- |
| Regression | Linear output or a suitable constraint |
| Binary classification | Sigmoid |
| Multiclass classification | Softmax |

---

## 15. Forward Propagation

A forward pass moves the input from left to right through the network to calculate the prediction:

$$
\hat{y}
$$

For a network with one hidden layer:

$$
a^{[0]}=x
$$

$$
z^{[1]}
=
W^{[1]}a^{[0]}
+
b^{[1]}
$$

$$
a^{[1]}
=
\operatorname{ReLU}
\left(
z^{[1]}
\right)
$$

$$
z^{[2]}
=
W^{[2]}a^{[1]}
+
b^{[2]}
$$

$$
a^{[2]}
=
\hat{y}
$$

The full sequence is:

$$
x
\longrightarrow
z^{[1]}
\longrightarrow
a^{[1]}
\longrightarrow
z^{[2]}
\longrightarrow
a^{[2]}
\longrightarrow
\hat{y}
$$

A hidden layer calculates an intermediate representation. The output layer calculates the final prediction.

---

## 16. Worked Layer Example

Suppose:

$$
W^{[1]}
=
\begin{bmatrix}
1 & -1 \\
2 & 3 \\
-4 & 2
\end{bmatrix}
$$

$$
x
=
\begin{bmatrix}
2 \\
1
\end{bmatrix}
$$

$$
b^{[1]}
=
\begin{bmatrix}
0 \\
-1 \\
3
\end{bmatrix}
$$

Calculate:

$$
z^{[1]}
=
W^{[1]}x+b^{[1]}
$$

$$
z^{[1]}
=
\begin{bmatrix}
1(2)+(-1)(1)+0 \\
2(2)+3(1)-1 \\
(-4)(2)+2(1)+3
\end{bmatrix}
$$

$$
z^{[1]}
=
\begin{bmatrix}
1 \\
6 \\
-3
\end{bmatrix}
$$

Apply ReLU:

$$
a^{[1]}
=
\operatorname{ReLU}
\left(
z^{[1]}
\right)
$$

$$
a^{[1]}
=
\begin{bmatrix}
1 \\
6 \\
0
\end{bmatrix}
$$

Important: ReLU leaves positive values unchanged. It does not increase them.

---

## 17. Neural Networks Versus Linear Models

A linear model uses one weighted sum:

$$
h_\theta(x)
=
\theta^Tx
$$

Without manually designed nonlinear features, it cannot represent complex nonlinear patterns.

A neural network combines linear transformations and nonlinear activations:

$$
a^{[1]}
=
\operatorname{ReLU}
\left(
W^{[1]}x+b^{[1]}
\right)
$$

$$
\hat{y}
=
W^{[2]}a^{[1]}+b^{[2]}
$$

The activation function prevents multiple layers from collapsing into one linear transformation.

This allows neural networks to learn more flexible representations from data.

---

## 18. Common Mistakes to Avoid

### Mistake 1: Confusing Pre-Activation and Activation

Matrix multiplication plus bias produces the pre-activation:

$$
z^{[l]}
=
W^{[l]}a^{[l-1]}+b^{[l]}
$$

The activation function produces the activation:

$$
a^{[l]}
=
g
\left(
z^{[l]}
\right)
$$

### Mistake 2: Adding Instead of Multiplying Weights and Inputs

Incorrect:

$$
W^{[l]}+a^{[l-1]}+b^{[l]}
$$

Correct:

$$
W^{[l]}a^{[l-1]}+b^{[l]}
$$

### Mistake 3: Using the Wrong Weight-Matrix Shape

Incorrect:

$$
\operatorname{shape}
\left(
W^{[l]}
\right)
=
\left(
n^{[l]},
1
\right)
$$

Correct:

$$
\operatorname{shape}
\left(
W^{[l]}
\right)
=
\left(
n^{[l]},
n^{[l-1]}
\right)
$$

Each current neuron needs one weight for each incoming value.

### Mistake 4: Confusing the Input Layer with a Raw Score

The input layer is:

$$
a^{[0]}=x
$$

It is the original feature vector, not a pre-activation.

### Mistake 5: Saying the Final Prediction Comes from a Hidden Layer

Hidden layers calculate intermediate learned representations.

The output layer calculates:

$$
\hat{y}
$$

### Mistake 6: Changing Positive Values During ReLU

ReLU only replaces negative values with zero:

$$
\operatorname{ReLU}(6)=6
$$

not:

$$
7
$$

---

## 19. Final Mental Model

For each neural-network layer:

$$
\boxed{
\text{previous activation}
\longrightarrow
\text{linear score}
\longrightarrow
\text{nonlinear activation}
}
$$

Mathematically:

$$
\boxed{
a^{[l-1]}
\longrightarrow
W^{[l]}a^{[l-1]}+b^{[l]}
\longrightarrow
z^{[l]}
\longrightarrow
g\left(z^{[l]}\right)
\longrightarrow
a^{[l]}
}
$$

The network repeats this process until the output layer produces:

$$
\hat{y}
$$

---

## 20. Self-Test Questions

1. What does a single neuron calculate before applying an activation function?
2. What does ReLU do when the raw score is negative?
3. What does ReLU do when the raw score is positive?
4. Why do multiple linear layers collapse into one linear transformation if there are no activation functions?
5. What is the difference between \(z^{[l]}\) and \(a^{[l]}\)?
6. What does \(a^{[0]}=x\) mean?
7. What is the matrix-multiplication shape rule?
8. If a layer receives 4 values and contains 7 neurons, what is the shape of \(W^{[l]}\)?
9. If a layer contains 7 neurons, what are the shapes of \(b^{[l]}\), \(z^{[l]}\), and \(a^{[l]}\)?
10. What is a hidden layer?
11. What does the output layer calculate?
12. What is a forward pass?
13. Why can neural networks represent more complex patterns than a basic linear model?

