import math

class LogisticRegression:

    def __init__(self, learning_rate=0.01, num_epochs=8):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weights = {}

    def _sigmoid(self, value):
        # simple protection against overflow
        if value < -500:
            return 0
        return 1.0 / (1.0 + math.exp(-value))

    def train(self, X, y):

        for epoch in range(self.num_epochs):

            for i in range(len(X)):

                sample = X[i]
                label = y[i]

                score = 0

                # compute dot product
                for index in sample:
                    weight = self.weights.get(index, 0)
                    score += weight * sample[index]

                prediction = self._sigmoid(score)
                difference = label - prediction

                # update only features present
                for index in sample:
                    current = self.weights.get(index, 0)
                    self.weights[index] = current + self.learning_rate * difference * sample[index]

    def predict(self, sample):

        score = 0

        for index in sample:
            weight = self.weights.get(index, 0)
            score += weight * sample[index]

        prob = self._sigmoid(score)

        if prob >= 0.5:
            return 1
        else:
            return 0
