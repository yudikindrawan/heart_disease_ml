import joblib
import os
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from preprocessing import load_and_preprocess_data


def evaluate_models():
    """
    The `evaluate_models` function loads trained models, evaluates their performance on a test set,
    selects the best model based on accuracy and F1 score, and saves the evaluation results and the best
    model for deployment.
    """
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess_data(
        "data/raw/heart-disease.csv", "data/processed/"
    )

    results = []
    best_model = None
    best_accuracy = 0
    best_f1 = 0

    for model_file in os.listdir("models"):
        if model_file.endswith(".pkl") and model_file != "scaler.pkl":
            model_path = os.path.join("models", model_file)
            model = joblib.load(model_path)
            
            # Evaluasi pada validation set
            # y_val_pred = model.predict(X_val)
            # val_accuracy = accuracy_score(y_val, y_val_pred)
            
            # results.append({"model": model_file, "accuracy": val_accuracy})
            
            # if val_accuracy > best_accuracy:
            #     best_accuracy = val_accuracy
            #     best_model = model_file

            #  Evaluasi pada test set
            y_test_pred        = model.predict(X_test)
            val_test_accuracy  = accuracy_score(y_test, y_test_pred)
            val_test_f1        = f1_score(y_test, y_test_pred)
            val_test_recall    = recall_score(y_test, y_test_pred)
            val_test_precision = precision_score(y_test, y_test_pred)
            
            results.append({"model": model_file,
                            "val_test_accuracy": val_test_accuracy,
                            "test_accuracy": val_test_accuracy,
                            "f1_score": val_test_f1,
                            "recall_score": val_test_recall,
                            "precision_score": val_test_precision})
            
            if (val_test_accuracy + val_test_f1) / 2 > (best_accuracy + best_f1) / 2:
                best_accuracy = val_test_accuracy
                best_f1       = val_test_f1
                best_model    = model_file
    
    print('best_accuracy update', best_accuracy)
    print('best_f1 update', best_f1)
    
    # Simpan hasil evaluasi
    results_df = pd.DataFrame(results)
    results_df.to_csv("models/evaluation_results.csv", index=False)
    
    print(f"Best Model: {best_model} Accuracy {best_accuracy:.4f}")
    print(f"Best Model: {best_model} F1 Score {best_f1:.4f}")

    # Tandai model terbaik untuk deployment
    os.rename(f"models/{best_model}", "models/best_model.pkl")


if __name__ == "__main__":
    evaluate_models()
