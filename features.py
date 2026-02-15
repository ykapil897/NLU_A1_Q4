import math
from collections import defaultdict


def build_vocab(all_tokens, max_features=1500):

    freq = {}

    for tokens in all_tokens:
        for word in tokens:
            if word in freq:
                freq[word] += 1
            else:
                freq[word] = 1

    # sort by frequency
    sorted_words = sorted(freq.items(), key=lambda item: item[1], reverse=True)

    vocab = {}
    for i in range(min(max_features, len(sorted_words))):
        word = sorted_words[i][0]
        vocab[word] = i

    return vocab


def vectorize_bow(tokens, vocab):

    vector = {}

    for word in tokens:
        if word in vocab:
            index = vocab[word]
            if index in vector:
                vector[index] += 1
            else:
                vector[index] = 1

    return vector


def vectorize_tf(tokens, vocab):

    vector = {}
    length = len(tokens)

    for word in tokens:
        if word in vocab:
            index = vocab[word]
            if index in vector:
                vector[index] += 1
            else:
                vector[index] = 1

    # normalize
    if length > 0:
        for index in vector:
            vector[index] = vector[index] / length

    return vector


def compute_idf(all_tokens, vocab):

    total_docs = len(all_tokens)
    doc_freq = defaultdict(int)

    for tokens in all_tokens:
        seen = set()
        for word in tokens:
            if word in vocab and word not in seen:
                doc_freq[word] += 1
                seen.add(word)

    idf_values = {}

    for word in vocab:
        df = doc_freq[word]
        idf_values[word] = math.log(total_docs / (1 + df))

    return idf_values


def vectorize_tfidf(tokens, vocab, idf_values):

    tf_vector = vectorize_tf(tokens, vocab)
    tfidf_vector = {}

    for index in tf_vector:
        # need reverse lookup from index → word
        # so we reconstruct word from vocab
        for word in vocab:
            if vocab[word] == index:
                tfidf_vector[index] = tf_vector[index] * idf_values[word]
                break

    return tfidf_vector


def generate_bigrams(tokens):

    combined = list(tokens)

    for i in range(len(tokens) - 1):
        bigram = tokens[i] + "_" + tokens[i + 1]
        combined.append(bigram)

    return combined
