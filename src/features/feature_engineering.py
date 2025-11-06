import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
from src.utils.logger import get_logger

logger = get_logger(__name__)

PROCESSED_PATH = "data/processed/restaurantes_USA_clean.csv"
TRAIN_PATH = "data/processed/train_features.csv"
TEST_PATH = "data/processed/test_features.csv"
PIPELINE_PATH = "models/feature_pipeline.pkl"


def build_feature_pipeline():
    """
    Construye el pipeline de ingeniería de características.
    Escala las variables numéricas y codifica las categóricas.
    """
    logger.info("=== Iniciando feature engineering ===")

    # Cargar datos limpios
    if not os.path.exists(PROCESSED_PATH):
        raise FileNotFoundError(f"No se encontró el archivo procesado en {PROCESSED_PATH}")

    df = pd.read_csv(PROCESSED_PATH)
    logger.info(f"Datos cargados correctamente: {df.shape[0]} filas, {df.shape[1]} columnas")

    # --- 1️⃣ Separar target y features ---
    if "membresia_premium" not in df.columns:
        raise ValueError("La columna 'membresia_premium' no existe en el dataset limpio.")

    y = df["membresia_premium"].apply(lambda x: 1 if str(x).lower() in ["sí"] else 0)
    X = df.drop(columns=["membresia_premium"])

    # --- 2️⃣ Definir columnas ---
    numeric_features = ["edad", "frecuencia_visita", "promedio_gasto_comida", "ingresos_mensuales"]
    categorical_features = [
        "genero",
        "ciudad_residencia",
        "estrato_socioeconomico",
        "ocio",
        "consume_licor",
        "preferencias_alimenticias",
        "tipo_de_pago_mas_usado",
    ]

    # --- 3️⃣ Dividir datos antes de transformar ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    logger.info(f"División de datos: Train {X_train.shape}, Test {X_test.shape}")

    # --- 4️⃣ Crear transformadores ---
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(
        steps=[("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # --- 5️⃣ Ajustar con training set (para evitar leakage) ---
    logger.info("Entrenando transformador con training set...")
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # --- 6️⃣ Reconstruir DataFrames con nombres de columnas ---
    encoded_cat_cols = preprocessor.named_transformers_["cat"]["encoder"].get_feature_names_out(categorical_features)
    final_cols = numeric_features + list(encoded_cat_cols)

    X_train_final = pd.DataFrame(X_train_transformed, columns=final_cols)
    X_test_final = pd.DataFrame(X_test_transformed, columns=final_cols)

    X_train_final["target"] = y_train.values
    X_test_final["target"] = y_test.values

    # --- 7️⃣ Guardar datasets procesados ---
    os.makedirs(os.path.dirname(TRAIN_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(TEST_PATH), exist_ok=True)
    X_train_final.to_csv(TRAIN_PATH, index=False)
    X_test_final.to_csv(TEST_PATH, index=False)
    logger.info(f"Datos de entrenamiento guardados en {TRAIN_PATH}")
    logger.info(f"Datos de prueba guardados en {TEST_PATH}")

    # --- 8️⃣ Guardar el pipeline ---
    os.makedirs(os.path.dirname(PIPELINE_PATH), exist_ok=True)
    joblib.dump(preprocessor, PIPELINE_PATH)
    logger.info(f"Pipeline de features guardado en {PIPELINE_PATH}")

    logger.info("=== Feature engineering completado exitosamente ===")

    return X_train_final, X_test_final


if __name__ == "__main__":
    build_feature_pipeline()
