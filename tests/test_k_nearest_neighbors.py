from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from models.k_nearest_neighbors import KNN
from utils.evaluation import ModelEvaluator

def test_knn():
    dataset = load_iris()
    X_iris = dataset.data
    y_iris = dataset.target

    X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris, random_state=1)

    model = KNN(k=3)

    model.fit(X_train, y_train)

    evaluation_metrics = ModelEvaluator().evaluate_classification_model(model, X_test, y_test)

    print("\n=====================================================================")
    print("Evaluation Metrics for KNN:")
    for metric, value in evaluation_metrics.items():
        print(f"{metric}: {round(value, 3)}")
    print("\n=====================================================================")

    min_accuracy = 0.9

    assert evaluation_metrics['accuracy'] >= min_accuracy
