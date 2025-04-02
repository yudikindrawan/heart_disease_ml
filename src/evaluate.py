import joblib
import os
import pandas as pd

from sklearn.metrics import accuracy_score
from preprocessing import load_and_preprocess_data

def evaluate_models():
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess_data('data/raw/heart_disease.csv', 'data/processed/')

    results = []
    best_model = None
    best_accuracy = 0

    for model_file in os.listdir("models"):
        if model_file.endswith(".pkl"):
            model_path = os.path.join("models", model_file)
            model = joblib.load(model_path)

            # Evaluasi pada validation set
            y_val_pred = model.predict(X_val)
            val_accuracy = accuracy_score(y_val, y_val_pred)

            results.append({"model": model_file, "accuracy": val_accuracy})

            # Pilih model terbaik
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_model = model_file

    # Simpan hasil evaluasi
    results_df = pd.DataFrame(results)
    results_df.to_csv("models/evaluation_results.csv", index=False)

    print(f"Model terbaik: {best_model} dengan akurasi {best_accuracy:.4f}")

    # Tandai model terbaik untuk deployment
    os.rename(f"models/{best_model}", "models/best_model.pkl")

if __name__ == "__main__":
    evaluate_models()