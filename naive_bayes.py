import math
from collections import defaultdict

class NaiveBayes:

    def train(self, X, y):
        self.class_counts = defaultdict(int)
        self.feature_counts = {}
        self.total_feature_count = {}

        for c in set(y):
            self.feature_counts[c] = defaultdict(float)
            self.total_feature_count[c] = 0

        # Count frequencies
        for i in range(len(X)):
            c = y[i]
            self.class_counts[c] += 1

            for idx, value in X[i].items():
                self.feature_counts[c][idx] += value
                self.total_feature_count[c] += value

        self.total_docs = len(X)
        self.vocab_size = len({idx for x in X for idx in x})

    def predict(self, x):
        scores = {}

        for c in self.class_counts:
            log_prob = math.log(self.class_counts[c] / self.total_docs)

            for idx, value in x.items():
                word_count = self.feature_counts[c].get(idx, 0) + 1
                total = self.total_feature_count[c] + self.vocab_size
                log_prob += value * math.log(word_count / total)

            scores[c] = log_prob

        return max(scores, key=scores.get)
