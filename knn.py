from collections import Counter
import math

class KNN:

    def __init__(self, k=3):
        self.k = k

    def train(self, X, y):
        self.X = X
        self.y = y

    def distance(self, v1, v2):
        keys = set(v1.keys()) | set(v2.keys())
        s = 0
        for k in keys:
            s += (v1.get(k, 0) - v2.get(k, 0)) ** 2
        return math.sqrt(s)

    def predict(self, x):
        distances = []
        for i in range(len(self.X)):
            d = self.distance(x, self.X[i])
            distances.append((d, self.y[i]))
        distances.sort()
        top_k = [label for _, label in distances[:self.k]]
        return Counter(top_k).most_common(1)[0][0]
