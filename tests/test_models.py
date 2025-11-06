import os
import pandas as pd
import joblib
from src.models.model_utils import get_model_dict

TRAIN_PATH = "data/processed/train_features.csv"
TEST_PATH = "data/processed/test_features.csv"

def test_train_and_predict():
    df_train = pd.read_csv(TRAIN_PATH)
    df_test = pd.read_csv(TEST_PATH)

    X_train, y_train = df_train.drop(columns=["target"]), df_train["target"]
    X_test, y_test = df_test.drop(columns=["target"]), df_test["target"]

    models = get_model_dict()
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        assert len(y_pred) == len(y_test), f"❌ El modelo {name} no predijo correctamente."

def test_best_model_exists():
    assert os.path.exists("models/local_best_model.pkl"), "❌ No se encontró el modelo local_best_model.pkl"

def test_best_model_can_predict():
    model = joblib.load("models/local_best_model.pkl")
    assert hasattr(model, "predict"), "❌ El modelo cargado no tiene método predict()."
