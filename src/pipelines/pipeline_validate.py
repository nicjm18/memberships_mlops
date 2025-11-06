import os
import sys
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)

def validate_environment():
    """
    Verifica la estructura, archivos críticos y columnas requeridas.
    """
    logger.info("=== Iniciando validaciones del entorno ===")

    required_files = [
        "data/raw/base_datos_restaurantes_USA_v2.csv",
        "data/processed/restaurantes_USA_clean.csv",
        "models/feature_pipeline.pkl"
    ]

    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        logger.error(f"Archivos faltantes: {missing}")
        sys.exit(1)
    else:
        logger.info("✅ Todos los archivos requeridos existen.")

    # Validar columnas mínimas
    df = pd.read_csv("data/processed/restaurantes_USA_clean.csv")
    expected_cols = [
        "edad", "frecuencia_visita", "promedio_gasto_comida",
        "ingresos_mensuales", "membresia_premium"
    ]
    missing_cols = [c for c in expected_cols if c not in df.columns]
    if missing_cols:
        logger.error(f"Columnas faltantes: {missing_cols}")
        sys.exit(1)

    logger.info("✅ Validación de columnas completada.")
    logger.info("=== Validaciones completadas correctamente ===")

if __name__ == "__main__":
    validate_environment()
