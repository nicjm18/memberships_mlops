import os
import pandas as pd
import pytest

PROCESSED_PATH = "data/processed/restaurantes_USA_clean.csv"

@pytest.fixture(scope="module")
def df():
    """
    Carga el dataset limpio para todos los tests.
    """
    if not os.path.exists(PROCESSED_PATH):
        pytest.fail(f"❌ No se encontró el archivo procesado en {PROCESSED_PATH}")
    return pd.read_csv(PROCESSED_PATH)


def test_file_exists():
    """
    Verifica que el archivo procesado exista.
    """
    assert os.path.exists(PROCESSED_PATH), f"❌ No existe el archivo {PROCESSED_PATH}"
    print("✅ Archivo de datos procesados encontrado correctamente.")


def test_no_missing_values(df):
    """
    Verifica que no existan valores nulos en el dataset.
    """
    missing = df.isna().sum().sum()
    assert missing == 0, f"❌ El dataset contiene {missing} valores nulos."
    print("✅ No se encontraron valores nulos.")


def test_age_range(df):
    """
    Verifica que los valores de 'edad' estén entre 0 y 100.
    """
    if "edad" not in df.columns:
        pytest.skip("⚠️ La columna 'edad' no existe en el dataset limpio.")
    invalid_rows = df[(df["edad"] < 0) | (df["edad"] > 100)]
    assert invalid_rows.empty, f"❌ Se encontraron edades fuera del rango 0–100: {invalid_rows['edad'].tolist()}"
    print("✅ Todas las edades están dentro del rango permitido (0–100).")


def test_non_empty_dataset(df):
    """
    Verifica que el dataset no esté vacío.
    """
    assert df.shape[0] > 0, "❌ El dataset está vacío después del preprocesamiento."
    print(f"✅ Dataset con {df.shape[0]} filas y {df.shape[1]} columnas cargado correctamente.")
