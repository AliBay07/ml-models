from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from models.linear_regression import LinearRegression
from utils.evaluation import ModelEvaluator

def test_fit_predict():
    dataset = load_iris()
    X_iris = dataset.data
    y_iris = dataset.target
    X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris, random_state=1)

    model = LinearRegression(method="mean_squared")
    model.fit(X_train, y_train)

    evaluation_metrics = ModelEvaluator().evaluate_regression_model(model, X_test, y_test)

    print("\n=====================================================================")
    print("Evaluation Metrics for Linear Regression with mean_squared method:")
    for metric, value in evaluation_metrics.items():
        print(f"{metric}: {round(value, 3)}")
    print("=====================================================================")

    min_mean_squared_error = 0.5
    min_mean_absolute_error = 0.5
    min_r_squared = 0.9

    assert evaluation_metrics['mean_squared_error'] <= min_mean_squared_error
    assert evaluation_metrics['mean_absolute_error'] <= min_mean_absolute_error
    assert evaluation_metrics['r_squared'] >= min_r_squared
