import pandas as pd
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix
from src.train_model   import train_model
def evaluate_model():
    # ... existing code ...
    test_df = pd.read_csv("data/processed/clean_test.csv")

    X_test = test_df.drop("Survived", axis=1)
    y_test = test_df["Survived"]

    # Get trained models
    log_model, tree_model = train_model()

    # Predictions
    log_preds = log_model.predict(X_test)
    tree_preds = tree_model.predict(X_test)
    
    metrics = {
        "logistic": {
            "accuracy": accuracy_score(y_test, log_preds),
            "precision": precision_score(y_test, log_preds),
            "recall": recall_score(y_test, log_preds),
            "f1": f1_score(y_test, log_preds),
            "conf_matrix": confusion_matrix(y_test, log_preds)
        },
        "decision_tree": {
            "accuracy": accuracy_score(y_test, tree_preds),
            "precision": precision_score(y_test, tree_preds),
            "recall": recall_score(y_test, tree_preds),
            "f1": f1_score(y_test, tree_preds),
            "conf_matrix": confusion_matrix(y_test, tree_preds)
        }
    }
    return metrics