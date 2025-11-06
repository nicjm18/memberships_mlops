import os
import pandas as pd
import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)

RAW_PATH = "data/raw/base_datos_restaurantes_USA_v2.csv"
PROCESSED_PATH = "data/processed/restaurantes_USA_clean.csv"

# Columnas a eliminar (irrelevantes para entrenamiento)
DROP_COLS = ["telefono_contacto", "correo_electronico", "nombre", "apellido", "id_persona"]

# Columnas a imputar y método de imputación
IMPUTATION_RULES = {
    "preferencias_alimenticias": "mode",
    "promedio_gasto_comida": "median",
    "edad": "median",
}

# Reglas de outliers
OUTLIER_RULES = {
    "edad": {"min": 0, "max": 100},
    "frecuencia_visita": {"min": 0, "max": 20},
}


def preprocess_data(input_path=RAW_PATH, output_path=PROCESSED_PATH):
    """
    Limpieza e imputación de datos para el dataset de membresías premium.
    """
    logger.info("=== Iniciando preprocesamiento de datos ===")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"No se encontró el archivo raw en {input_path}")

    df = pd.read_csv(input_path)
    logger.info(f"Datos cargados correctamente: {df.shape[0]} filas, {df.shape[1]} columnas")

    # --- 1️⃣ Eliminar columnas irrelevantes ---
    existing_drop = [col for col in DROP_COLS if col in df.columns]
    df.drop(columns=existing_drop, inplace=True)
    logger.info(f"Columnas eliminadas: {existing_drop}")

    # --- 2️⃣ Convertir object a category (opcional pero eficiente) ---
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype("category")
    logger.info("Columnas tipo 'object' convertidas a 'category'.")

    # --- 3️⃣ Imputación de valores faltantes ---
    for col, method in IMPUTATION_RULES.items():
        if col in df.columns:
            if method == "mode":
                value = df[col].mode()[0]
            elif method == "mean":
                value = df[col].mean()
            elif method == "median":
                value = df[col].median()
            else:
                raise ValueError(f"Método de imputación desconocido: {method}")

            df[col].fillna(value, inplace=True)
            logger.info(f"Columna '{col}' imputada por {method}: {value}")

    # --- 4️⃣ Eliminar outliers de edad ---
    if "edad" in df.columns:
        min_val = OUTLIER_RULES["edad"]["min"]
        max_val = OUTLIER_RULES["edad"]["max"]
        before = df.shape[0]
        df = df[(df["edad"] >= min_val) & (df["edad"] <= max_val)]
        after = df.shape[0]
        logger.info(f"Filas eliminadas por outliers en 'edad': {before - after}")

    # --- 5️⃣ Validación de datos post-procesamiento ---
    if df.isna().sum().any():
        missing_cols = df.columns[df.isna().any()].tolist()
        logger.warning(f"Aún existen valores faltantes en: {missing_cols}")

    # --- 6️⃣ Guardar dataset limpio ---
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Datos procesados guardados en {output_path}")
    logger.info("=== Preprocesamiento completado exitosamente ===")

    return df


if __name__ == "__main__":
    preprocess_data()
