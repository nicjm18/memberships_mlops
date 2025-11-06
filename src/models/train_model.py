import os
import pandas as pd
import mlflow
import mlflow.sklearn
from src.models.model_utils import get_model_dict, calculate_metrics
from src.utils.logger import get_logger
import joblib

logger = get_logger(__name__)

TRAIN_PATH = "data/processed/train_features.csv"
TEST_PATH = "data/processed/test_features.csv"
MLFLOW_TRACKING_URI = "file:./mlruns"  # Puedes cambiar a servidor remoto


def train_and_log_models():
    """
    Entrena varios modelos, registra métricas y artefactos en MLflow.
    """
    logger.info("=== Iniciando entrenamiento de modelos ===")

    # --- 1️⃣ Configurar MLflow ---
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("membresias_premium_models")

    # --- 2️⃣ Cargar datos ---
    if not os.path.exists(TRAIN_PATH) or not os.path.exists(TEST_PATH):
        raise FileNotFoundError("Archivos de entrenamiento o prueba no encontrados.")
    df_train = pd.read_csv(TRAIN_PATH)
    df_test = pd.read_csv(TEST_PATH)

    X_train, y_train = df_train.drop(columns=["target"]), df_train["target"]
    X_test, y_test = df_test.drop(columns=["target"]), df_test["target"]

    models = get_model_dict()

    # --- 3️⃣ Entrenar y loggear modelos ---
    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            logger.info(f"Entrenando modelo: {model_name}")

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

            metrics = calculate_metrics(y_test, y_pred, y_proba)

            # --- Log de métricas ---
            for k, v in metrics.items():
                mlflow.log_metric(k, v)

            # --- Guardar modelo ---
            mlflow.sklearn.log_model(model, artifact_path="model")
            logger.info(f"Modelo '{model_name}' registrado en MLflow.")

            os.makedirs("models", exist_ok=True)
            local_model_path = f"models/{model_name}.pkl"
            joblib.dump(model, local_model_path)
            logger.info(f"Modelo '{model_name}' guardado localmente en {local_model_path}")

    logger.info("=== Entrenamiento completado ===")


if __name__ == "__main__":
    train_and_log_models()
