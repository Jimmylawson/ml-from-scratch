

import numpy as np
from typing import List, Tuple
from model import build_vocabulary, vectorization,prior,laplace_smoothing,predict_one

def load_sms_data(file_path: str) -> List[Tuple[int, str]]:
    """
    Load SMS spam data from tab-separated file
    Returns list of (label, message) tuples
    label: 1 for spam, 0 for ham
    """
    data = []

    # Open the file for reading with UTF-8 encoding
    with open(file_path, "r", encoding="utf-8") as f:
        # Enumerate through each line in the file with line numbers starting from 1
        for line_num, raw_line in enumerate(f, start=1):
            # Remove leading/trailing whitespace from the line
            line = raw_line.strip()
            # Skip empty lines
            if not line:
                continue
            # Split the line on the first tab character to separate label and message
            parts = line.split("\t", 1)
            # Skip lines that don't have exactly 2 parts (label and message)
            if len(parts) != 2:
                print(f"[warn] skipping malformed line {line_num}:{line[:60]}")
                continue

            # Unpack the parts into label_str and message
            label_str, message = parts
            # Clean up the label: remove whitespace and convert to lowercase
            label_str = label_str.strip().lower()
            # Convert string labels to numeric values
            if label_str == "spam":
                y = 1  # Spam = 1
            elif label_str == "ham":
                y = 0  # Ham = 0
            else:
                # Handle unexpected labels
                print(f"[warn] unknown label at line {line_num}: {label_str}")
                continue

            # Clean up the message by removing whitespace
            message = message.strip()
            # Add the (label, message) tuple to our data list
            data.append((y, message))

    return data




if __name__ == "__main__":
    sms_data = load_sms_data("data/SMSSpamCollection")
    rng = np.random.default_rng(42)
    indices = list(rng.permutation(len(sms_data)))

    split = int(0.8 * len(indices))
    train_indices = indices[:split]
    test_indices = indices[split:]
    train_data = [sms_data[i] for i in train_indices]
    test_data = [sms_data[i] for i in test_indices]

    #train and test
    vocab,words_to_idx = build_vocabulary(train_data)
    X_train = np.array([vectorization(vocab, words_to_idx, msg)  for (y,msg) in train_data])
    y_train = np.array([y for y, _ in train_data])
    messages = [msg for _, msg in test_data]
    X_test = np.array([vectorization(vocab, words_to_idx, msg)  for (y,msg) in test_data])
    y_test = np.array([y for y, _ in test_data])
    # print(f"X_train shape: {X_train.shape}")
    # print(f"Y_train shape: {y_train.shape}")
    # print(f"X_test shape: {X_test.shape}")
    # print(f"Y_test shape: {y_test.shape}")

    #predict
    #Estimate parameters from train
    phi_y = prior(y_train)
    phi_spam, phi_ham = laplace_smoothing(X_train, y_train)

    #Predict test tables
    y_pred = np.array([predict_one(x, phi_y, phi_spam, phi_ham) for x in X_test])

    #evaluate
    acc = np.mean(y_pred == y_test)
    print(f"accuracy: {acc:.4f}")

    #computer top indicative words using
    # print(np.log(phi_spam/ phi_ham)) # (spam-indictive)
    # print(np.log(phi_ham/ phi_spam)) # (ham-indictive)

    spam_log_ratio = np.log(phi_spam / phi_ham)  # bigger => more spam-like
    ham_log_ratio = np.log(phi_ham / phi_spam)  # bigger => more ham-like

    # indices of top words
    top_spam_idx = np.argsort(spam_log_ratio)[-15:][::-1]
    top_ham_idx = np.argsort(ham_log_ratio)[-15:][::-1]

    print("\nTop spam-indicative words:")
    for i in top_spam_idx:
        print(f"{vocab[i]:<15} score={spam_log_ratio[i]:.3f}")

    print("\nTop ham-indicative words:")
    for i in top_ham_idx:
        print(f"{vocab[i]:<15} score={ham_log_ratio[i]:.3f}")
    # print(f"tota l: {len(sms_data)}")
    # print(f"train: {len(train_data)}, test: {len(test_data)}")
    # train_spam = sum(y for y, _ in train_data) / len(train_data)
    # test_spam = sum(y for y, _ in test_data) / len(test_data)
    # print(f"train spam rate: {train_spam:.4f}")
    # print(f"test spam rate: {test_spam:.4f}")
    #
    # spam_count  = sum(y for y, _ in sms_data)
    # print(f"spam count {spam_count}")
    # print(f"ham count {len(sms_data) - spam_count}")
    # print(sms_data[:2])
