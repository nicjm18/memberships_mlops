import os
import pandas as pd
import joblib
from mlflow import sklearn as mlflow_sklearn
from src.utils.logger import get_logger

logger = get_logger(__name__)

MODEL_PATH = "models/feature_pipeline.pkl"
DATA_PATH = "data/new_data.csv"          # archivo con nuevos clientes o registros
OUTPUT_PATH = "data/predictions.csv"


def run_prediction_pipeline():
    """
    Carga el modelo final y genera predicciones para nuevos datos.
    """
    logger.info("=== Iniciando pipeline de predicciones ===")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("No se encontró el pipeline de features.")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("No se encontró el archivo de nuevos datos.")

    # --- 1️⃣ Cargar pipeline de features y modelo ---
    logger.info("Cargando pipeline de features...")
    feature_pipeline = joblib.load(MODEL_PATH)

    logger.info("Cargando modelo de MLflow...")
    # Aquí podrías usar MLflow Registry directamente si estás conectado
    model_uri = "models:/membresia_premium_best_model/Production"
    try:
        model = mlflow_sklearn.load_model(model_uri)
        logger.info("Modelo cargado desde MLflow Registry.")
    except Exception:
        logger.warning("No se encontró modelo en MLflow, cargando localmente.")
        model = joblib.load("models/local_best_model.pkl")

    # --- 2️⃣ Transformar nuevos datos ---
    df_new = pd.read_csv(DATA_PATH)
    df_transformed = feature_pipeline.transform(df_new)

    # --- 3️⃣ Generar predicciones ---
    preds = model.predict(df_transformed)
    preds_proba = (
        model.predict_proba(df_transformed)[:, 1] if hasattr(model, "predict_proba") else None
    )

    df_new["prediccion"] = preds
    if preds_proba is not None:
        df_new["probabilidad"] = preds_proba

    # --- 4️⃣ Guardar resultados ---
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_new.to_csv(OUTPUT_PATH, index=False)
    logger.info(f"Predicciones guardadas en {OUTPUT_PATH}")

    logger.info("=== Pipeline de predicción completado ===")


if __name__ == "__main__":
    run_prediction_pipeline()
