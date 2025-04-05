import joblib
import yaml
import argparse

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score

from preprocessing import load_and_preprocess_data

# Konfugurasi model yang tersedia
MODELS = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "knn": KNeighborsClassifier,
}


def train_model(model_name, config_path):
    """
    The `train_model` function trains a machine learning model specified by the user with
    hyperparameters loaded from a config file and saves the trained model for evaluation.

    :param model_name: The `model_name` parameter specifies the name of the machine learning model to
    train. It can take values such as "logistic_regression", "random_forest", or "knn" based on the
    available models in the script
    :param config_path: The `config_path` parameter in the `train_model` function is the path to the
    configuration file that contains hyperparameters for the machine learning model. This file is used
    to load hyperparameters for the specified model during training
    :return: The `train_model` function returns the following values:
    1. `model_path`: Path to the saved model file.
    2. `val_accuracy`: Validation accuracy score of the trained model.
    3. `val_f1_score`: Validation F1 score of the trained model.
    4. `val_recall_score`: Validation recall score of the trained model.
    5. `val_precision_score`: Validation precision score
    """
    # Load hyperparameters from config file
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Load dataset
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess_data(
        "data/raw/heart-disease.csv", "data/processed/"
    )

    # Check if model_name is valid
    if model_name not in MODELS:
        raise ValueError(
            f"Model {model_name} is not supported. Choose from {list(MODELS.keys())}."
        )

    # Initialize model
    model = MODELS[model_name](**config)

    #! Need Improvement and refactoring for adding hyperparameter tuning knn, logistic regression, and random forest using GridSearchCV
    # TODO: Add hyperparameter tuning for knn and logistic regression using GridSearchCV
    # * Done implement hyperparameter tuning for random forest, knn, logistic regression using GridSearchCV

    #! Need evaluate model and bese model for best score evaluate
    # TODO: Add RandomSearchCV for hyperparameter tuning and try using cross-val-score for evaluation

    # Get Hyperparameters for GridSeacrhCV
    param_grid = config.copy()
    param_grid.pop("random_state", None)

    # Check if model for hyperparameter tuning or not
    if model_name == "random_forest":
        param_grid["max_depth"] = [
            None if x == "None" else x for x in param_grid.get("max_depth", [])
        ]

    # GrisdSearchCV for hyperparameter tuning
    if param_grid:
        print("use grid search")
        grid_search = GridSearchCV(
            model, param_grid, cv=5, n_jobs=-1, verbose=1, error_score="raise"
        )
        grid_search.fit(X_train, y_train)

        # Get best model from grid search
        model = grid_search.best_estimator_
        print(model)
        print(f"Best parameters for {model_name}: {grid_search.best_params_}")
    # Model without hyperparameter tuning
    else:
        print("fit only")
        # Training model
        model.fit(X_train, y_train)

    # Save the model
    model_path = f"models/{model_name}_model.pkl"
    joblib.dump(model, model_path)

    # Evaluate model
    y_val_pred = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_f1_score = f1_score(y_val, y_val_pred)
    val_recall_score = recall_score(y_val, y_val_pred)
    val_precision_score = precision_score(y_val, y_val_pred)

    print(f"{model_name} Validation Accuracy: {val_accuracy:.4f}")
    print(f"{model_name} Validation F1 Score: {val_f1_score:.4f}")
    print(f"{model_name} Validation Recall Score: {val_recall_score:.4f}")
    print(f"{model_name} Validation Precision Score: {val_precision_score:.4f}")

    return model_path, val_accuracy, val_f1_score, val_recall_score, val_precision_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a machine learning model.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name to train (logistic_regression, random_forest, knn)",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the config file for hyperparameters",
    )

    args = parser.parse_args()

    train_model(args.model, args.config)
