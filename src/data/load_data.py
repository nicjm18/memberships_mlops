import json
from google.cloud import bigquery
import pandas as pd
from src.utils.logger import get_logger
import os

logger = get_logger(__name__)

def load_from_bigquery(config_path: str) -> pd.DataFrame:
    """
    Carga datos desde BigQuery usando las credenciales activas.

    Args:
        config_path (str): Ruta al archivo JSON con 'project_id' y 'query'.
    """
    try:
        with open(config_path, "r") as f:
            config = json.load(f)

        project_id = config.get("project_id")
        query = config.get("query")

        if not project_id or not query:
            raise ValueError("El JSON debe contener 'project_id' y 'query' válidos.")

        client = bigquery.Client(project=project_id)

        logger.info(f"Ejecutando consulta en BigQuery para el proyecto: {project_id}")
        df = client.query(query).to_dataframe()
        logger.info(f"Datos cargados correctamente: {df.shape[0]} filas, {df.shape[1]} columnas")

        # Guardar localmente
        raw_path = "data/raw/base_datos_restaurantes_USA_v2.csv"
        os.makedirs(os.path.dirname(raw_path), exist_ok=True)
        df.to_csv(raw_path, index=False)
        logger.info(f"Datos guardados en {raw_path}")

        return df

    except Exception as e:
        logger.error(f"Error al cargar datos desde BigQuery: {str(e)}")
        raise


if __name__ == "__main__":
    config_path = "config/credentials.json"
    load_from_bigquery(config_path)