import math

class LogisticRegression:

    def __init__(self, lr=0.01, epochs=8):
        self.lr = lr
        self.epochs = epochs

    def sigmoid(self, z):
        if z < -500:
            return 0
        return 1 / (1 + math.exp(-z))

    def train(self, X, y):
        self.weights = {}

        for _ in range(self.epochs):
            for i in range(len(X)):
                z = 0
                for idx, val in X[i].items():
                    z += self.weights.get(idx, 0) * val

                pred = self.sigmoid(z)
                error = y[i] - pred

                for idx, val in X[i].items():
                    self.weights[idx] = self.weights.get(idx, 0) + self.lr * error * val

    def predict(self, x):
        z = 0
        for idx, val in x.items():
            z += self.weights.get(idx, 0) * val
        return 1 if self.sigmoid(z) >= 0.5 else 0
