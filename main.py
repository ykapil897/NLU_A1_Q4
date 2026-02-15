from dataset import load_dataset
from preprocess import tokenize
from features import *
from naive_bayes import NaiveBayes
from logistic_regression import LogisticRegression
from knn import KNN
from evaluation import accuracy
import random

print("======================================")
print("SPORTS vs POLITICS CLASSIFIER STARTED")
print("======================================\n")

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------
print("[1] Loading dataset...")
texts, labels = load_dataset("data/bbc-text.csv", max_samples=800)
print(f"    Total samples loaded: {len(texts)}")

# --------------------------------------------------
# 2. TOKENIZATION + BIGRAMS
# --------------------------------------------------
print("\n[2] Tokenizing + Generating Bigrams...")
tokenized = []
for i, text in enumerate(texts):
    tokens = tokenize(text)
    tokens = generate_bigrams(tokens)
    tokenized.append(tokens)

    if (i+1) % 200 == 0:
        print(f"    Processed {i+1} documents")

print("    Tokenization complete.")

# --------------------------------------------------
# 3. BUILD VOCAB
# --------------------------------------------------
print("\n[3] Building vocabulary...")
vocab = build_vocab(tokenized)
print(f"    Vocabulary size: {len(vocab)}")

# --------------------------------------------------
# 4. COMPUTE IDF (needed for TF-IDF)
# --------------------------------------------------
print("\n[4] Computing IDF...")
idf = compute_idf(tokenized, vocab)
print("    IDF computation complete.")

# --------------------------------------------------
# 5. SHUFFLE DATA
# --------------------------------------------------
print("\n[5] Shuffling dataset...")
combined = list(zip(tokenized, labels))
random.shuffle(combined)
tokenized, labels = zip(*combined)

split = int(0.8 * len(tokenized))
train_tokens = tokenized[:split]
test_tokens = tokenized[split:]
y_train = labels[:split]
y_test = labels[split:]

print(f"    Train size: {len(train_tokens)}")
print(f"    Test size: {len(test_tokens)}")

# --------------------------------------------------
# FEATURE TYPES
# --------------------------------------------------
feature_types = ["BoW", "TF-IDF", "NGRAM"]

results = {}

for feature in feature_types:

    print("\n======================================")
    print(f"Running Feature Type: {feature}")
    print("======================================")

    # ------------------------------
    # Prepare tokens
    # ------------------------------
    if feature == "NGRAM":
        train_tokens_mod = [generate_bigrams(t) for t in train_tokens]
        test_tokens_mod = [generate_bigrams(t) for t in test_tokens]
    else:
        train_tokens_mod = train_tokens
        test_tokens_mod = test_tokens

    # ------------------------------
    # Build vocab
    # ------------------------------
    vocab = build_vocab(train_tokens_mod, max_features=1500)

    # ------------------------------
    # Compute IDF if needed
    # ------------------------------
    if feature == "TF-IDF":
        idf = compute_idf(train_tokens_mod, vocab)

    # ------------------------------
    # Vectorize
    # ------------------------------
    if feature == "BoW" or feature == "NGRAM":
        X_train = [vectorize_bow(t, vocab) for t in train_tokens_mod]
        X_test = [vectorize_bow(t, vocab) for t in test_tokens_mod]

    elif feature == "TF-IDF":
        X_train = [vectorize_tfidf(t, vocab, idf) for t in train_tokens_mod]
        X_test = [vectorize_tfidf(t, vocab, idf) for t in test_tokens_mod]

    print("    Vectorization complete.")

    results[feature] = {}

    # ==============================
    # NAIVE BAYES
    # ==============================
    print("    Training Naive Bayes...")
    nb = NaiveBayes()
    nb.train(X_train, y_train)
    nb_preds = [nb.predict(x) for x in X_test]
    nb_acc = accuracy(y_test, nb_preds)
    print(f"    NB Accuracy: {nb_acc:.4f}")
    results[feature]["NB"] = nb_acc

    # ==============================
    # LOGISTIC REGRESSION
    # ==============================
    print("    Training Logistic Regression...")
    lr = LogisticRegression(lr=0.01, epochs=20)
    lr.train(X_train, y_train)
    lr_preds = [lr.predict(x) for x in X_test]
    lr_acc = accuracy(y_test, lr_preds)
    print(f"    LR Accuracy: {lr_acc:.4f}")
    results[feature]["LR"] = lr_acc

    # ==============================
    # KNN
    # ==============================
    print("    Training KNN...")
    knn = KNN(k=3)
    knn.train(X_train, y_train)
    knn_preds = [knn.predict(x) for x in X_test]
    knn_acc = accuracy(y_test, knn_preds)
    print(f"    KNN Accuracy: {knn_acc:.4f}")
    results[feature]["KNN"] = knn_acc


# --------------------------------------------------
# FINAL COMPARISON TABLE
# --------------------------------------------------
print("\n======================================")
print("FINAL COMPARISON TABLE")
print("======================================")

print(f"{'Feature':<10} | {'NB':<8} | {'LR':<8} | {'KNN':<8}")
print("-" * 45)

for feature in feature_types:
    print(f"{feature:<10} | "
          f"{results[feature]['NB']:.4f} | "
          f"{results[feature]['LR']:.4f} | "
          f"{results[feature]['KNN']:.4f}")

print("\nPROCESS COMPLETED SUCCESSFULLY.")
