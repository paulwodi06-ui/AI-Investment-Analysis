from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from dataset_builder import build_ml_dataset

def train_random_forest():
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

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    print("\nFeature importances:")

    for feature, importance in zip(
        features,
        model.feature_importances_
    ):
        print(
            feature,
            round(importance, 4)
        )

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    baseline_accuracy = y_test.value_counts(
        normalize=True
    ).max()

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

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
    train_random_forest()