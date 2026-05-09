# Naive Bayes Spam Classifier Revision Notes

This folder implements a Bernoulli Naive Bayes spam classifier from scratch.

The dataset is the UCI SMS Spam Collection.

Labels:

```text
spam = 1
ham  = 0
```

## 1. What Naive Bayes Is

Naive Bayes is a generative learning algorithm.

Instead of directly learning:

$$
p(y\mid x)
$$

it learns:

$$
p(x\mid y)\quad \text{and}\quad p(y)
$$

Then it uses Bayes rule:

$$
p(y\mid x)
=
\frac{p(x\mid y)p(y)}{p(x)}
$$

For prediction, `p(x)` is the same for every class, so we do not need to compute it.

We compare:

$$
\arg\max_y\ p(x\mid y)p(y)
$$

In log space:

$$
\arg\max_y
\left[
\log p(y) + \log p(x\mid y)
\right]
$$

That is what your `predict_one` function does.

## 2. Why It Is Called Generative

Naive Bayes models how messages are generated from each class.

For spam:

```text
p(x | y = 1)
```

For ham:

```text
p(x | y = 0)
```

It asks:

```text
If this message came from the spam class, how likely would these words be?
If this message came from the ham class, how likely would these words be?
```

Then it combines those likelihoods with the class priors.

## 3. Loading the SMS Data

Your loader reads a file where each line contains:

```text
label<TAB>message
```

Example:

```text
ham     Go until jurong point...
spam    Free entry in 2 a wkly comp...
```

Code idea:

```python
parts = line.split("\t", 1)
label_str, message = parts
```

Then:

```python
if label_str == "spam":
    y = 1
elif label_str == "ham":
    y = 0
```

The final data format is:

```python
[(0, "Go until jurong point..."), (1, "Free entry...")]
```

## 4. Train/Test Split

The data is shuffled:

```python
rng = np.random.default_rng(42)
indices = list(rng.permutation(len(sms_data)))
```

Then split:

```python
split = int(0.8 * len(indices))
train_indices = indices[:split]
test_indices = indices[split:]
```

Then:

```python
train_data = [sms_data[i] for i in train_indices]
test_data = [sms_data[i] for i in test_indices]
```

Why split?

- training data estimates the probabilities
- test data checks whether the model works on messages it did not learn from

## 5. Tokenization

Tokenization means turning a message into words.

Your function:

```python
def tokenization(message: str) -> List[str]:
    message = message.lower()
    clean = re.sub(r"[^a-z0-9\s]", " ", message)
    tokens = clean.split()
    return tokens
```

It does:

1. lowercase the text
2. replace punctuation with spaces
3. split into words

Example:

```text
"WIN!!! Free Prize"
```

becomes:

```python
["win", "free", "prize"]
```

This makes words easier to count consistently.

## 6. Vocabulary

The vocabulary is the list of unique words from the training data.

Code:

```python
def build_vocabulary(train_data):
    vocab_set = set()
    for y, message in train_data:
        token = tokenization(message)
        vocab_set.update(token)

    vocab = sorted(vocab_set)
    words_to_idx = {word: i for i, word in enumerate(vocab)}

    return vocab, words_to_idx
```

Why use a `set`?

Because each word should appear only once in the vocabulary.

Why sort?

So the vocabulary order is stable and reproducible.

Why `words_to_idx`?

Because a model needs numbers, not raw words.

Example:

```python
vocab = ["call", "free", "prize", "win"]
words_to_idx = {
    "call": 0,
    "free": 1,
    "prize": 2,
    "win": 3
}
```

## 7. Bernoulli Vectorization

This project uses the Bernoulli event model.

That means each word feature is binary:

```text
x_j = 1 if word j appears in the message
x_j = 0 otherwise
```

It does not count how many times the word appears.

Code:

```python
def vectorization(vocab, words_to_idx, message):
    vector = np.zeros(len(vocab))
    token = tokenization(message)

    for word in token:
        if word in words_to_idx:
            vector[words_to_idx[word]] = 1

    return vector
```

Example:

```python
vocab = ["call", "free", "prize", "win"]
message = "free prize free"
```

Vector:

```text
call  free  prize  win
 0      1      1     0
```

So:

```python
[0, 1, 1, 0]
```

Because this is Bernoulli Naive Bayes, `free` is still just `1`, not `2`.

## 8. Building X and y

Training matrix:

```python
X_train = np.array([
    vectorization(vocab, words_to_idx, msg)
    for (y, msg) in train_data
])
```

This converts every training message into a binary vector.

Training labels:

```python
y_train = np.array([y for y, _ in train_data])
```

The `_` means:

```text
ignore the message part
```

Example:

```python
train_data = [(0, "hello"), (1, "free prize")]
```

Then:

```python
y_train = [0, 1]
```

## 9. Class Prior

The class prior is:

$$
\phi_y = p(y=1)
$$

It means:

```text
Before seeing the words, how common is spam?
```

Code:

```python
def prior(y_train):
    return np.mean(y_train)
```

This works because:

```text
spam = 1
ham = 0
```

Example:

```python
y_train = [0, 1, 0, 1, 1]
```

Then:

```text
mean = (0 + 1 + 0 + 1 + 1) / 5 = 3/5 = 0.6
```

So:

```text
p(spam) = 0.6
```

Then:

$$
p(ham) = 1 - \phi_y
$$

## 10. Bernoulli Word Probability

For each word `j` and class `y`, estimate:

```text
phi_{j|y} = p(x_j = 1 | y)
```

Meaning:

```text
Among messages in class y, how often does word j appear?
```

For spam:

```text
phi_{j|1} = p(word j appears | spam)
```

For ham:

```text
phi_{j|0} = p(word j appears | ham)
```

Without smoothing:

```text
phi_{j|1}
= count(spam messages where word j appears) / count(spam messages)
```

```text
phi_{j|0}
= count(ham messages where word j appears) / count(ham messages)
```

## 11. Laplace Smoothing

Problem:

If a word never appears in spam training messages, then:

```text
p(word | spam) = 0
```

That can destroy the whole product during prediction.

Laplace smoothing prevents zero probabilities.

For Bernoulli Naive Bayes:

$$
\phi_{j|1}
=
\frac{\text{spam_word_count}_j + \alpha}
{n_{\text{spam}} + 2\alpha}
$$

$$
\phi_{j|0}
=
\frac{\text{ham_word_count}_j + \alpha}
{n_{\text{ham}} + 2\alpha}
$$

What each term means:

- $\phi_{j|1}$ means probability that word $j$ appears in a spam message.
- $\phi_{j|0}$ means probability that word $j$ appears in a ham message.
- $\alpha$ is the Laplace smoothing value.
- $n_{\text{spam}}$ is the number of spam training messages.
- $n_{\text{ham}}$ is the number of ham training messages.
- $2\alpha$ appears because a Bernoulli feature has two outcomes: word appears or word does not appear.

Why `2 alpha`?

Because Bernoulli has two possible outcomes:

```text
x_j = 1
x_j = 0
```

Code:

```python
def laplace_smoothing(X_train, y_train, alpha=1.0):
    n_spam, n_ham = class_count(y_train)

    X_spam = X_train[y_train == 1]
    X_ham = X_train[y_train == 0]

    spam_words_count = np.sum(X_spam, axis=0)
    ham_words_count = np.sum(X_ham, axis=0)

    phi_x_given_spam = (spam_words_count + alpha) / (n_spam + 2 * alpha)
    phi_x_given_ham = (ham_words_count + alpha) / (n_ham + 2 * alpha)

    return phi_x_given_spam, phi_x_given_ham
```

`axis=0` means count each word column across all rows/messages.

## 12. Naive Independence Assumption

Naive Bayes assumes that words are conditionally independent given the class.

That means:

$$
p(x\mid y)
=
\prod_{j=1}^{V}p(x_j\mid y)
$$

This is not perfectly true in real language, but it works surprisingly well.

For Bernoulli features:

$$
p(x_j \mid y)
=
\phi_{j|y}^{x_j}
\left(1-\phi_{j|y}\right)^{1-x_j}
$$

Why?

If `x_j = 1`:

```text
p(x_j | y) = phi_{j|y}
```

If `x_j = 0`:

```text
p(x_j | y) = 1 - phi_{j|y}
```

For the whole message:

$$
p(x \mid y)
=
\prod_{j=1}^{V}
\phi_{j|y}^{x_j}
\left(1-\phi_{j|y}\right)^{1-x_j}
$$

Here $V$ is the vocabulary size.

## 13. Prediction in Log Space

Direct products can underflow because many probabilities are tiny.

So use logs:

$$
\log p(x \mid y)
=
\sum_{j=1}^{V}
\left[
x_j\log\phi_{j|y}
+
(1-x_j)\log(1-\phi_{j|y})
\right]
$$

Spam score:

$$
score_{\text{spam}}
=
\log p(y=1)
+
\sum_{j=1}^{V}
\left[
x_j\log\phi_{j|1}
+
(1-x_j)\log(1-\phi_{j|1})
\right]
$$

Ham score:

$$
score_{\text{ham}}
=
\log p(y=0)
+
\sum_{j=1}^{V}
\left[
x_j\log\phi_{j|0}
+
(1-x_j)\log(1-\phi_{j|0})
\right]
$$

Predict:

```text
if score_spam > score_ham: spam
else: ham
```

Code:

```python
def predict_one(x, phi_y, phi_x_given_spam, phi_x_given_ham):
    spam_score = np.log(phi_y) + np.sum(
        x * np.log(phi_x_given_spam)
        + (1 - x) * np.log(1 - phi_x_given_spam)
    )

    ham_score = np.log(1 - phi_y) + np.sum(
        x * np.log(phi_x_given_ham)
        + (1 - x) * np.log(1 - phi_x_given_ham)
    )

    return 1 if spam_score > ham_score else 0
```

## 14. Batch Prediction

Your code predicts every test example:

```python
y_pred = np.array([
    predict_one(x, phi_y, phi_spam, phi_ham)
    for x in X_test
])
```

Each `x` is one message vector.

The result is an array of predicted labels:

```text
[0, 1, 0, 0, 1, ...]
```

## 15. Accuracy

Code:

```python
acc = np.mean(y_pred == y_test)
```

Meaning:

```text
accuracy = number correct / number of test examples
```

Example:

```text
y_pred = [0, 1, 0, 1]
y_test = [0, 1, 1, 1]
```

Comparison:

```text
[True, True, False, True]
```

Mean:

```text
3/4 = 0.75
```

## 16. Indicative Word Scores

You computed:

```python
spam_log_ratio = np.log(phi_spam / phi_ham)
ham_log_ratio = np.log(phi_ham / phi_spam)
```

Spam-indicative score:

$$
\log
\frac{\phi_{j|1}}{\phi_{j|0}}
$$

Meaning:

- positive: word appears more in spam than ham
- negative: word appears more in ham than spam
- zero: word appears equally often

Example:

```text
phi_{free|spam} = 0.40
phi_{free|ham} = 0.02
```

Then:

```text
ratio = 0.40 / 0.02 = 20
log ratio = log(20)
```

Large positive score means the word is strong spam evidence.

Why log?

Because Naive Bayes prediction also works in log space, and log ratios are additive.

## 17. Top Words

Code:

```python
top_spam_idx = np.argsort(spam_log_ratio)[-15:][::-1]
top_ham_idx = np.argsort(ham_log_ratio)[-15:][::-1]
```

`np.argsort(...)` returns indices sorted from smallest score to largest score.

`[-15:]` takes the largest 15.

`[::-1]` reverses them so the biggest score comes first.

Then:

```python
for i in top_spam_idx:
    print(f"{vocab[i]:<15} score={spam_log_ratio[i]:.3f}")
```

prints the most spam-indicative words.

`:.3f` means print the number with 3 digits after the decimal.

## 18. Scoring One Message

Your `score_message` function:

```python
def score_message(msg, vocab, words_to_idx, phi_y, phi_x_given_spam, phi_x_given_ham):
    x = vectorization(vocab, words_to_idx, msg)
    ...
    present_idx = np.where(x == 1)[0]
    contrib = np.log(phi_x_given_spam[present_idx]) - np.log(phi_x_given_ham[present_idx])
    ...
    return pred, spam_score, ham_score, present_idx, contrib
```

It does:

1. vectorize one message
2. compute spam score
3. compute ham score
4. find words present in the message
5. compute how much each present word pushes toward spam

This line:

```python
present_idx = np.where(x == 1)[0]
```

returns the indices of words that appear in the message.

This line:

```python
contrib = np.log(phi_x_given_spam[present_idx]) - np.log(phi_x_given_ham[present_idx])
```

is the same as:

```text
log(phi_{j|1} / phi_{j|0})
```

for only the words present in the message.

## 19. Important Interpretation

The top contributing present words do not alone decide the final prediction.

Final prediction uses:

- class prior
- present-word evidence
- absent-word evidence
- spam score
- ham score

So a word can be spam-indicative, but the whole message can still be ham if the total ham score is larger.

## 20. Big Picture

Naive Bayes spam classification flow:

```text
load text
tokenize
build vocabulary
vectorize messages
estimate p(y)
estimate p(word appears | spam)
estimate p(word appears | ham)
compute spam score and ham score
choose larger score
```

Most important prediction formula:

$$
score_y
=
\log p(y)
+
\sum_{j=1}^{V}
\left[
x_j\log\phi_{j|y}
+
(1-x_j)\log(1-\phi_{j|y})
\right]
$$

That is the heart of your Bernoulli Naive Bayes implementation.
