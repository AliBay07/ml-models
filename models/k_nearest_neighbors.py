import numpy as np
from collections import Counter

class KNN:
    """
    K-Nearest Neighbors classifier.

    Parameters:
        k (int): Number of neighbors to consider.
    """

    def __init__(self, k=3):
        self.y_train = None
        self.X_train = None
        self.k = k

    def fit(self, X, y):
        """
        Fit the model using the training data.

        Parameters:
            X (array-like): Training samples.
            y (array-like): Target values.
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X):
        """
        Predict the class labels for the provided samples.

        Parameters:
            X (array-like): Samples to classify.

        Returns:
            array: Predicted class labels.
        """
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        # Compute distances between x and all examples in the training set
        distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        # Sort by distance and return indices of the first k neighbors
        k_indices = np.argsort(distances)[:self.k]
        # Extract the labels of the k nearest neighbor training samples
        k_nearest_labels = self.y_train[k_indices]
        # Return the most common class label (for classification)
        return Counter(k_nearest_labels).most_common(1)[0][0]
