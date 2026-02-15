import math
from collections import defaultdict


class NaiveBayes:

    def __init__(self):
        self.class_counts = defaultdict(int)
        self.word_counts = {}
        self.total_words = {}
        self.num_docs = 0
        self.vocab_size = 0

    def train(self, X, y):

        self.num_docs = len(X)

        classes = set(y)

        for label in classes:
            self.word_counts[label] = defaultdict(float)
            self.total_words[label] = 0

        for i in range(len(X)):

            label = y[i]
            self.class_counts[label] += 1

            for index, value in X[i].items():
                self.word_counts[label][index] += value
                self.total_words[label] += value

        # count unique feature indices seen in training data
        all_indices = set()
        for sample in X:
            for index in sample:
                all_indices.add(index)

        self.vocab_size = len(all_indices)

    def predict(self, sample):

        best_label = None
        best_score = None

        for label in self.class_counts:

            # prior probability
            score = math.log(self.class_counts[label] / self.num_docs)

            for index, value in sample.items():

                count = self.word_counts[label].get(index, 0)
                count += 1  # Laplace smoothing

                total = self.total_words[label] + self.vocab_size

                score += value * math.log(count / total)

            if best_score is None or score > best_score:
                best_score = score
                best_label = label

        return best_label
