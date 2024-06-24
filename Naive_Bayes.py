import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
# P(y/X) = P(X/y) * P(X) / p(y)


class Naive_Bayes:
    def __init__(self):
        pass

    def fit(self, X, y):
        self.x_train = X
        self.y_train = y
        self.n_samples, self.n_features = self.x_train.shape # Store no of Samples and Features
        self._classes = np.unique(y) # Store all the classes in _classes
        self.n_classes = len(self._classes)

        # Calculate mean,variance and prior for each class

        self._mean = np.zeros((self.n_classes, self.n_features))
        self._var = np.zeros((self.n_classes, self.n_features))
        self._priors = np.zeros(self.n_classes)

        for i, c in enumerate(self._classes):
            xc = X[y == c]  # Get only those features where class is equal to current class.
            self._mean[i, :] = xc.mean(axis=0)  # Mean of features with respect to current class.
            self._var[i, :] = xc.var(axis=0)  # Variance of features with respect to current class.
            self._priors[i] = xc.shape[0] / float(self.n_samples)  # No of samples in current class divided by total no of samples.

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return y_pred

    def _pdf(self, class_index, x):
        mean = self._mean[class_index]
        var = self._var[class_index]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    def _predict(self, x):
        posteriors = []
        for i, c in enumerate(self._classes):
            prior = np.log(self._priors[i])
            posterior = np.sum(np.log(self._pdf(i, x)))
            posterior = posterior + prior
            posteriors.append(posterior)

        return self._classes[np.argmax(posteriors)]


def accuracy(y_true, pred):
    acc = np.sum(y_true == pred) / len(y_true)
    return acc


if __name__ == "__main__":
    X, y = make_classification(
        n_samples=500, n_features=10, n_classes=2, random_state=44
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=44)
    classifier = Naive_Bayes()
    classifier.fit(X_train, y_train)
    preds = classifier.predict(X_test)
    acc = accuracy(y_test, preds)
    print(f"Accuracy: {acc}")
