import os
import pandas as pd

RAW_PATH = "data/raw/base_datos_restaurantes_USA_v2.csv"

def test_raw_file_exists():
    assert os.path.exists(RAW_PATH), "❌ No se encontró el archivo raw."

def test_raw_not_empty():
    df = pd.read_csv(RAW_PATH)
    assert len(df) > 0, "❌ El dataset raw está vacío."

def test_no_duplicated_rows():
    df = pd.read_csv(RAW_PATH)
    duplicates = df.duplicated().sum()
    assert duplicates == 0, f"⚠️ Existen {duplicates} filas duplicadas en el dataset."

def test_required_columns():
    df = pd.read_csv(RAW_PATH)
    expected_cols = [
        "edad", "frecuencia_visita", "promedio_gasto_comida",
        "ingresos_mensuales", "membresia_premium"
    ]
    for col in expected_cols:
        assert col in df.columns, f"❌ Falta la columna requerida: {col}"
