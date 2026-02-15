from dataset import load_dataset
from preprocess import tokenize
from features import build_vocab, vectorize_bow, vectorize_tfidf, compute_idf, generate_bigrams
from naive_bayes import NaiveBayes
from logistic_regression import LogisticRegression
from knn import KNN
from evaluation import accuracy
import random

print("Starting classification...\n")

texts, labels = load_dataset("data/bbc-text.csv")

print("Total documents:", len(texts))

# basic tokenization
tokens = [tokenize(t) for t in texts]

# shuffle once
combined = list(zip(tokens, labels))
random.shuffle(combined)
tokens, labels = zip(*combined)

split = int(0.8 * len(tokens))
train_tokens = tokens[:split]
test_tokens = tokens[split:]
y_train = labels[:split]
y_test = labels[split:]

print("Train size:", len(train_tokens))
print("Test size:", len(test_tokens))

features_to_run = ["bow", "tfidf", "ngram"]

all_results = {}

for feature_type in features_to_run:

    print("\nRunning:", feature_type.upper())

    if feature_type == "ngram":
        train_mod = [generate_bigrams(t) for t in train_tokens]
        test_mod = [generate_bigrams(t) for t in test_tokens]
    else:
        train_mod = train_tokens
        test_mod = test_tokens

    vocab = build_vocab(train_mod, max_features=1500)

    if feature_type == "tfidf":
        idf = compute_idf(train_mod, vocab)

    if feature_type == "bow" or feature_type == "ngram":
        X_train = [vectorize_bow(t, vocab) for t in train_mod]
        X_test = [vectorize_bow(t, vocab) for t in test_mod]
    else:
        X_train = [vectorize_tfidf(t, vocab, idf) for t in train_mod]
        X_test = [vectorize_tfidf(t, vocab, idf) for t in test_mod]

    results = {}

    # Naive Bayes
    nb = NaiveBayes()
    nb.train(X_train, y_train)
    preds_nb = [nb.predict(x) for x in X_test]
    acc_nb = accuracy(y_test, preds_nb)
    print("NB accuracy:", round(acc_nb, 4))
    results["NB"] = acc_nb

    # Logistic Regression
    lr = LogisticRegression(learning_rate=0.01, num_epochs=8)
    lr.train(X_train, y_train)
    preds_lr = [lr.predict(x) for x in X_test]
    acc_lr = accuracy(y_test, preds_lr)
    print("LR accuracy:", round(acc_lr, 4))
    results["LR"] = acc_lr

    # KNN
    knn = KNN(k=3)
    knn.train(X_train, y_train)
    preds_knn = [knn.predict(x) for x in X_test]
    acc_knn = accuracy(y_test, preds_knn)
    print("KNN accuracy:", round(acc_knn, 4))
    results["KNN"] = acc_knn

    all_results[feature_type] = results

print("\nFinal comparison:\n")

for f in features_to_run:
    r = all_results[f]
    print(f, "-> NB:", round(r["NB"], 4),
          "LR:", round(r["LR"], 4),
          "KNN:", round(r["KNN"], 4))

print("\nDone.")
