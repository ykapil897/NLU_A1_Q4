from collections import Counter
import math

class KNN:

    def __init__(self, k=3):
        self.k = k
        self.train_data = None
        self.train_labels = None

    def train(self, X, y):
        self.train_data = X
        self.train_labels = y

    def _euclidean_distance(self, a, b):

        # union of feature indices
        all_indices = set(a.keys()).union(b.keys())

        total = 0
        for idx in all_indices:
            val1 = a.get(idx, 0)
            val2 = b.get(idx, 0)
            diff = val1 - val2
            total += diff * diff

        return math.sqrt(total)

    def predict(self, sample):

        distances = []

        for i in range(len(self.train_data)):
            d = self._euclidean_distance(sample, self.train_data[i])
            distances.append((d, self.train_labels[i]))

        # sort by distance
        distances.sort(key=lambda x: x[0])

        nearest_labels = []
        for i in range(self.k):
            nearest_labels.append(distances[i][1])

        vote = Counter(nearest_labels).most_common(1)

        return vote[0][0]
