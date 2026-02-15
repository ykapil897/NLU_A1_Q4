import math
from collections import defaultdict

# ------------------------
# BUILD VOCAB (LIMITED)
# ------------------------
def build_vocab(tokenized_texts, max_features=1500):
    word_freq = {}

    for tokens in tokenized_texts:
        for word in tokens:
            word_freq[word] = word_freq.get(word, 0) + 1

    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

    vocab = {}
    for i, (word, _) in enumerate(sorted_words[:max_features]):
        vocab[word] = i

    return vocab


# ------------------------
# SPARSE BOW
# ------------------------
def vectorize_bow(tokens, vocab):
    vec = {}
    for word in tokens:
        if word in vocab:
            idx = vocab[word]
            vec[idx] = vec.get(idx, 0) + 1
    return vec


# ------------------------
# SPARSE TF
# ------------------------
def vectorize_tf(tokens, vocab):
    vec = {}
    total = len(tokens)

    for word in tokens:
        if word in vocab:
            idx = vocab[word]
            vec[idx] = vec.get(idx, 0) + 1

    for k in vec:
        vec[k] = vec[k] / total

    return vec


# ------------------------
# IDF
# ------------------------
def compute_idf(tokenized_texts, vocab):
    N = len(tokenized_texts)
    df = defaultdict(int)

    for tokens in tokenized_texts:
        seen = set()
        for word in tokens:
            if word in vocab and word not in seen:
                df[word] += 1
                seen.add(word)

    idf = {}
    for word in vocab:
        idf[word] = math.log(N / (1 + df[word]))

    return idf


# ------------------------
# SPARSE TF-IDF
# ------------------------
def vectorize_tfidf(tokens, vocab, idf):
    vec = vectorize_tf(tokens, vocab)

    for word in tokens:
        if word in vocab:
            idx = vocab[word]
            vec[idx] = vec[idx] * idf[word]

    return vec


# ------------------------
# BIGRAMS
# ------------------------
def generate_bigrams(tokens):
    bigrams = []
    for i in range(len(tokens)-1):
        bigrams.append(tokens[i] + "_" + tokens[i+1])
    return tokens + bigrams
