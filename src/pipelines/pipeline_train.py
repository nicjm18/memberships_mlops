import os
from src.data.preprocess_data import preprocess_data
from src.features.feature_engineering import build_feature_pipeline
from src.models.train_model import train_and_log_models
from src.models.model_eval import evaluate_models
from src.models.model_registry import register_best_model
from src.utils.logger import get_logger

logger = get_logger(__name__)

def run_training_pipeline():
    """
    Orquesta todo el flujo de entrenamiento de modelos ML:
    1. Carga de datos desde GCP
    2. Preprocesamiento
    3. Feature Engineering
    4. Entrenamiento + Tracking
    5. Evaluación + Registro del mejor modelo
    """
    logger.info("=== Iniciando pipeline completo de entrenamiento ===")

    # --- 1️⃣ Carga de datos ---
    raw_path = "data/raw/base_datos_restaurantes_USA_v2.csv"
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"No se encontró el dataset local en {raw_path}")
    logger.info(f"Usando dataset local desde {raw_path}")

    # --- 2️⃣ Preprocesamiento ---
    logger.info("Ejecutando preprocesamiento...")
    preprocess_data(input_path=raw_path, output_path="data/processed/restaurantes_USA_clean.csv")

    # --- 3️⃣ Feature Engineering ---
    logger.info("Ejecutando feature engineering...")
    build_feature_pipeline()

    # --- 4️⃣ Entrenamiento + MLflow ---
    logger.info("Entrenando modelos y registrando en MLflow...")
    train_and_log_models()

    # --- 5️⃣ Evaluación y Registro ---
    logger.info("Evaluando modelos...")
    evaluate_models()
    logger.info("Registrando mejor modelo...")
    register_best_model()

    logger.info("=== Pipeline de entrenamiento completado exitosamente ===")


if __name__ == "__main__":
    run_training_pipeline()
