from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, \
    recall_score, f1_score


class ModelEvaluator:
    def __init__(self):
        pass

    def evaluate_regression_model(self, model, X_test, y_test):
        """
        Evaluate the performance of a regression model on test data.

        Parameters:
            model (object): The trained regression model.
            X_test (array-like): Test input features.
            y_test (array-like): True labels for the test data.

        Returns:
            dict: A dictionary containing evaluation metrics.
        """
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        evaluation_metrics = {
            'mean_squared_error': mse,
            'mean_absolute_error': mae,
            'r_squared': r2
        }

        return evaluation_metrics

    def evaluate_classification_model(self, model, X_test, y_test):
        """
        Evaluate the performance of a classification model on test data.

        Parameters:
            model (object): The trained classification model.
            X_test (array-like): Test input features.
            y_test (array-like): True labels for the test data.

        Returns:
            dict: A dictionary containing evaluation metrics.
        """
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        evaluation_metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

        return evaluation_metrics
