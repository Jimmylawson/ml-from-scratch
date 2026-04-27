
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

def prior(y_train):
    return np.mean(y_train)

def class_count(y_train):
    n_spam = np.sum(y_train == 1)
    n_ham = np.sum(y_train == 0)

    return n_spam, n_ham

def laplace_smoothing(X_train, y_train, alpha= 1.0):
    n_spam, n_ham = class_count(y_train)
    #split row by class
    X_spam = X_train[y_train == 1]
    X_ham = X_train[y_train == 0]
    #count appearance per word
    spam_words_count  = np.sum(X_spam, axis=0)
    ham_words_count = np.sum(X_ham, axis=0)

    #apply laplace smoothing
    phi_x_given_spam  = (spam_words_count + alpha) / (n_spam +  2 * alpha)
    phi_x_given_ham = (ham_words_count + alpha) / (n_ham + 2 * alpha)
    return phi_x_given_spam, phi_x_given_ham



def predict_one(x,phi_y,phi_x_given_spam,phi_x_given_ham):
    spam_score = np.log(phi_y) + np.sum(
        x * np.log(phi_x_given_spam) + (1 - x) * np.log(1 - phi_x_given_spam)
    )

    ham_score = np.log(1 - phi_y) + np.sum(
        x * np.log(phi_x_given_ham) + (1 - x) * np.log(1 - phi_x_given_ham)
    )
    return 1 if spam_score > ham_score else 0

def score_message(msg,vocab, words_to_idx, phi_y, phi_x_given_spam, phi_x_given_ham):
    x = vectorization(vocab, words_to_idx, msg)
    spam_score = np.log(phi_y) + np.sum(
        x * np.log(phi_x_given_spam) + (1 - x) * np.log(1 - phi_x_given_spam)
    )

    ham_score = np.log(1 - phi_y) + np.sum(
        x * np.log(phi_x_given_ham) + (1 - x) * np.log(1 - phi_x_given_ham)
    )
    present_idx = np.where(x == 1)[0]
    contrib = np.log(phi_x_given_spam[present_idx]) - np.log(phi_x_given_ham[present_idx])

    #sort biggest spam-push first
    order = np.argsort(contrib)[:: -1]
    present_idx = present_idx[order]
    contrib = contrib[order]

    pred = 1 if spam_score > ham_score else 0

    return pred, spam_score,ham_score, present_idx, contrib


