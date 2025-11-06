import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime
from src.utils.logger import get_logger

logger = get_logger(__name__)

API_URL = "http://localhost:8000/predict"
REPORT_PATH = "reports/monitoring_log.csv"

def simulate_request():
    """Simula una predicción aleatoria para verificar rendimiento."""
    payload = {
        "edad": np.random.randint(18, 70),
        "frecuencia_visita": np.random.randint(1, 10),
        "promedio_gasto_comida": np.random.uniform(20, 80),
        "ingresos_mensuales": np.random.uniform(1000, 18000),
        "genero": "Masculino",
        "ciudad_residencia": "NYC",
        "estrato_socioeconomico": "Medio",
        "ocio": "Si",
        "consume_licor": "No",
        "preferencias_alimenticias": "Carnes",
        "tipo_de_pago_mas_usado": "Tarjeta"
    }

    response = requests.post(API_URL, json=payload)
    latency = response.elapsed.total_seconds()

    if response.status_code == 200:
        result = response.json()
        logger.info(f"Predicción OK - Latencia: {latency:.3f}s - Prob: {result['probability']}")
        return {"timestamp": datetime.now(), "status": "OK", "latency": latency}
    else:
        logger.warning(f"Error en API: {response.status_code}")
        return {"timestamp": datetime.now(), "status": "FAIL", "latency": None}

def monitor_loop(interval_sec=60):
    """Ejecuta monitoreo continuo."""
    logger.info("=== Iniciando monitorización de API ===")
    os.makedirs("reports", exist_ok=True)
    history = []

    while True:
        record = simulate_request()
        history.append(record)
        pd.DataFrame(history).to_csv(REPORT_PATH, index=False)
        time.sleep(interval_sec)

if __name__ == "__main__":
    monitor_loop(interval_sec=300)  # cada 5 minutos
