import argparse
import os
import pickle

import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split


def parse_args():
    parser = argparse.ArgumentParser(description="Train entity resolution models")
    parser.add_argument(
        "--input",
    )
    parser.add_argument("--output-dir", type=str)
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["xgboost", "random_forest", "logistic"],
        default="xgboost",
    )
    parser.add_argument("--test-size", type=float)
    return parser.parse_args()


def prepare_data(df):
    df = df[~(df.label.fillna("") == "")]
    df.loc[:, "label"] = df["label"].astype(int)

    feature_cols = [
        col
        for col in df.columns
        if any(x in col for x in ["jaro", "levenshtein", "length_ratio", "embedding"])
    ]

    X = df[feature_cols]
    y = df["label"]

    return X, y


def train_xgboost(X_train, y_train):
    """Train XGBoost model with grid search."""
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 6, 9],
        "learning_rate": [0.01, 0.1, 0.2],
        "subsample": [0.8, 1],
        "colsample_bytree": [0.8, 1],
    }

    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

    grid_search.fit(X_train, y_train)
    print("Best XGBoost Parameters:", grid_search.best_params_)

    return grid_search.best_estimator_


def train_random_forest(X_train, y_train):
    """Train Random Forest model with grid search."""
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

    grid_search.fit(X_train, y_train)
    print("Best Random Forest Parameters:", grid_search.best_params_)

    return grid_search.best_estimator_


def train_logistic(X_train, y_train):
    """Train Logistic Regression model with grid search."""
    param_grid = {
        "C": [0.01, 0.1, 1, 10, 100],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear", "saga"],
    }

    model = LogisticRegression(random_state=42, max_iter=1000)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

    grid_search.fit(X_train, y_train)
    print("Best Logistic Regression Parameters:", grid_search.best_params_)

    return grid_search.best_estimator_


def compare_models(X_train, X_test, y_train, y_test):
    """Train and compare all models, returning the best one."""
    models = {
        "xgboost": train_xgboost,
        "random_forest": train_random_forest,
        "logistic": train_logistic,
    }

    results = {}
    best_accuracy = 0
    best_model = None
    best_model_name = None

    print("\nModel Comparison:")
    print("=" * 50)

    for model_name, model_trainer in models.items():
        print(f"\nTraining {model_name}...")
        model = model_trainer(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        results[model_name] = {
            "accuracy": accuracy,
            "report": classification_report(y_test, y_pred, output_dict=True),
            "model": model,
        }

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_model_name = model_name

    # Print comparison
    print("\nFinal Results:")
    print("=" * 50)
    for model_name, result in results.items():
        print(f"\n{model_name.upper()}:")
        print(f"Accuracy: {result['accuracy']:.4f}")
        print("Detailed Performance:")
        print(
            f"Precision (class 1): "
            f"{result['report'].get('1', result['report'].get(1, {})).get('precision', 0):.4f}"
        )
        print(
            f"Recall (class 1): "
            f"{result['report'].get('1', result['report'].get(1, {})).get('recall', 0):.4f}"
        )
        print(
            f"F1-score (class 1): "
            f"{result['report'].get('1', result['report'].get(1, {})).get('f1-score', 0):.4f}"
        )

    print("\nBest Model:")
    print(f"{best_model_name} (Accuracy: {best_accuracy:.4f})")

    return best_model, best_model_name, results


def main():
    args = parse_args()

    print(f"Reading data from {args.input}")
    df = pd.read_csv(args.input)

    print("Preparing features and labels...")
    X, y = prepare_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42
    )

    best_model, best_model_name, results = compare_models(X_train, X_test, y_train, y_test)

    model_path = os.path.join(args.output_dir, f"best_model_{best_model_name}.pkl")
    print(f"\nSaving best model ({best_model_name}) to {model_path}")
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)


if __name__ == "__main__":
    main()
