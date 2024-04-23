import numpy as np

class LinearRegression:
    def __init__(self, method='mean_squared'):
        """
        Initialize the Linear Regression model.

        Parameters:
            method (str): Method for fitting the model ('mean_squared' etc.).
                          Default is 'least_squares'.
        """
        self.method = method
        self.coefficients = None

    def fit(self, X, y):
        """
        Fit the Custom Linear Regression model to the training data.

        Parameters:
            X (array-like): Training input features.
            y (array-like): Target values.

        Returns:
            self: The fitted Custom Linear Regression model.
        """

        if self.method == 'mean_squared':
            self.coefficients = self._fit_mean_squared(X, y)
        else:
            raise ValueError("Invalid method. Choose from 'mean_squared' etc.")

        return self

    def predict(self, X):
        """
        Make predictions using the Linear Regression model.

        Parameters:
            X (array-like): Input features for prediction.

        Returns:
            array: Predicted target values.
        """
        if self.coefficients is None:
            raise ValueError("Model has not been trained yet. Please call fit() first.")

        return np.dot(X, self.coefficients)

    def _fit_mean_squared(self, X, y):
        """
        Fit the model using the mean squared method.

        Parameters:
            X (array-like): Training input features.
            y (array-like): Target values.

        Returns:
            array: Coefficients of the linear model.
        """

        coefficients = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

        return coefficients
