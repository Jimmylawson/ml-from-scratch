
from typing import List
import re
import numpy as np
def tokenization(message:str)-> List[str]:
    message = message.lower()
    clean = re.sub(r"[^a-z0-9\s]", " ", message)
    tokens = clean.split()
    return tokens


def build_vocabulary(train_data):
    words_to_idx = {}
    vocab_set = set()
    for y, message in train_data:
        token = tokenization(message)
        vocab_set.update(token)

    vocab = sorted(vocab_set)
    words_to_idx = {word: i for i, word in enumerate(vocab)}

    return vocab, words_to_idx
#Bernoulli vectorization
def vectorization(vocab, words_to_idx, message):
    vector = np.zeros(len(vocab))
    token = tokenization(message)

    for word in token:
        if word in words_to_idx:
            vector[words_to_idx[word]] = 1

    return vector


