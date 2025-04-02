import joblib
import yaml
import argparse

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from preprocessing import load_and_preprocess_data

# Konfugurasi model yang tersedia
MODELS = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "knn": KNeighborsClassifier,
}


def train_model(model_name, config_path):
    print('config_path: %s' % config_path)
    print('model_name: %s' % model_name)
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
    print(model)
    # Training model
    model.fit(X_train, y_train)

    # Save the model
    model_path = f"models/{model_name}_model.pkl"
    joblib.dump(model, model_path)

    # Evaluate model
    y_val_pred = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)

    print(f"{model_name} Validation Accuracy: {val_accuracy*100:.2f}%")

    return model_path, val_accuracy


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
