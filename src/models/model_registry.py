import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
from src.utils.logger import get_logger
import joblib
import os
from mlflow import sklearn as mlflow_sklearn

logger = get_logger(__name__)

MLFLOW_TRACKING_URI = "file:./mlruns"
EXPERIMENT_NAME = "membresias_premium_models"


def register_best_model():
    """
    Registra el mejor modelo (por f1_score) en el MLflow Model Registry
    y guarda una copia local del modelo real.
    """
    logger.info("=== Iniciando registro del mejor modelo ===")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    if not experiment:
        raise ValueError(f"Experimento '{EXPERIMENT_NAME}' no encontrado.")

    runs = client.search_runs(experiment.experiment_id, order_by=["metrics.f1_score DESC"])
    best_run = runs[0]

    model_uri = f"runs:/{best_run.info.run_id}/model"
    model_name = "membresia_premium_best_model"

    logger.info(f"Registrando modelo {model_name} con run_id {best_run.info.run_id}")

    mlflow.register_model(model_uri=model_uri, name=model_name)
    logger.info("=== Modelo registrado exitosamente en MLflow Registry ===")

    # Guardar una copia local del mejor modelo REAL
    os.makedirs("models", exist_ok=True)
    best_model = mlflow_sklearn.load_model(model_uri)
    joblib.dump(best_model, "models/local_best_model.pkl")
    logger.info("✅ Copia local del mejor modelo guardada en models/local_best_model.pkl")


if __name__ == "__main__":
    register_best_model()
