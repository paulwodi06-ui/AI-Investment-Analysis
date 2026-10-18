from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dataset_builder import build_ml_dataset

def train_logistic_regression():
    dataset = build_ml_dataset()

    features = [
        "Daily_Return",
        "Momentum_5d",
        "Volatility_20d",
        "MA_Ratio_20d"
    ]

    split_date = "2024-01-01"

    train_data = dataset[
        dataset["Date"] < split_date
    ]

    test_data = dataset[
        dataset["Date"] >= split_date
    ]

    X_train = train_data[features]
    y_train = train_data["Target"]

    X_test = test_data[features]
    y_test = test_data["Target"]

    model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
    )

    model.fit(X_train, y_train)

    print("\nModel coefficients:")

    logistic_model = model.named_steps["logisticregression"]

    for feature, coefficient in zip(
        features,
        logistic_model.coef_[0]
    ):
        print(
            feature,
            round(coefficient, 4)
        )

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(
    y_test,
    probabilities
    )
    accuracy = accuracy_score(
        y_test,
        predictions
    )

    baseline_accuracy = y_test.value_counts(normalize=True).max()

    print("Training observations:", len(X_train))
    print("Testing observations:", len(X_test))
    print("Baseline accuracy:", round(baseline_accuracy, 4))
    print("Model accuracy:", round(accuracy, 4))

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification report:")
    print(classification_report(y_test, predictions))

    print("ROC-AUC:", round(roc_auc, 4))

    return model

if __name__ == "__main__":
    train_logistic_regression()