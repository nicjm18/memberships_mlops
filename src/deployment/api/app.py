from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import uvicorn
from mlflow import sklearn as mlflow_sklearn

app = FastAPI(title="Membresías Premium API", version="1.0.0")

# === Cargar modelo y pipeline ===
MODEL_URI = "models:/membresia_premium_best_model/Production"
LOCAL_MODEL_PATH = "models/local_best_model.pkl"
PIPELINE_PATH = "models/feature_pipeline.pkl"

try:
    model = mlflow_sklearn.load_model(MODEL_URI)
    print("✅ Modelo cargado desde MLflow Registry.")
except Exception:
    model = joblib.load(LOCAL_MODEL_PATH)
    print("⚠️  No se pudo acceder a MLflow. Usando modelo local.")
    
feature_pipeline = joblib.load(PIPELINE_PATH)


# === Esquema de entrada ===
class ClientData(BaseModel):
    edad: float
    frecuencia_visita: float
    promedio_gasto_comida: float
    ingresos_mensuales: float
    genero: str
    ciudad_residencia: str
    estrato_socioeconomico: str
    ocio: str
    consume_licor: str
    preferencias_alimenticias: str
    tipo_de_pago_mas_usado: str


@app.get("/")
def root():
    return {"status": "API Online", "version": "1.0.0"}


@app.post("/predict")
def predict(data: ClientData):
    """
    Recibe un cliente en formato JSON y devuelve la predicción de membresía premium.
    """
    input_df = pd.DataFrame([data.dict()])

    # Aplicar pipeline de features
    transformed = feature_pipeline.transform(input_df)

    # Predicciones
    pred = model.predict(transformed)[0]
    proba = model.predict_proba(transformed)[0, 1] if hasattr(model, "predict_proba") else None

    return {
        "prediction": int(pred),
        "probability": round(float(proba), 4) if proba is not None else None
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
