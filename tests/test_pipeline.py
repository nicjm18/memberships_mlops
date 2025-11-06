import os
from src.pipelines.pipeline_train import run_training_pipeline

def test_pipeline_runs_successfully():
    run_training_pipeline()
    assert os.path.exists("data/processed/train_features.csv"), "❌ No se generó train_features.csv"
    assert os.path.exists("models/local_best_model.pkl"), "❌ No se generó el modelo local."
    assert os.path.exists("reports/model_evaluation.csv"), "❌ No se generó el reporte de evaluación."
