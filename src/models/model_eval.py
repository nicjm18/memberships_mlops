import mlflow
import pandas as pd
from sklearn.model_selection import cross_val_score
from src.models.model_utils import calculate_metrics, get_model_dict
from src.utils.logger import get_logger
import time
import os

logger = get_logger(__name__)

TRAIN_PATH = "data/processed/train_features.csv"
TEST_PATH = "data/processed/test_features.csv"


def evaluate_models():
    """
    Evalúa performance, consistencia y escalabilidad de cada modelo.
    """
    logger.info("=== Iniciando evaluación de modelos ===")

    df_train = pd.read_csv(TRAIN_PATH)
    df_test = pd.read_csv(TEST_PATH)
    X_train, y_train = df_train.drop(columns=["target"]), df_train["target"]
    X_test, y_test = df_test.drop(columns=["target"]), df_test["target"]

    models = get_model_dict()
    eval_results = []

    for name, model in models.items():
        logger.info(f"Evaluando modelo: {name}")

        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        # Performance
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        metrics = calculate_metrics(y_test, y_pred, y_proba)

        # Consistencia (CV)
        cv_score = cross_val_score(model, X_train, y_train, cv=5, scoring="f1").mean()
        metrics["cross_val_f1"] = cv_score

        # Escalabilidad (tiempo de entrenamiento)
        metrics["train_time_sec"] = round(train_time, 3)

        eval_results.append({"modelo": name, **metrics})

    df_results = pd.DataFrame(eval_results)
    os.makedirs("reports", exist_ok=True)
    df_results.to_csv("reports/model_evaluation.csv", index=False)
    logger.info("Resultados de evaluación guardados en reports/model_evaluation.csv")

    best_model = df_results.sort_values("f1_score", ascending=False).iloc[0]
    summary_path = "reports/model_summary.md"

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# 📊 Resumen de Evaluación de Modelos\n\n")
        f.write(f"**Mejor modelo:** {best_model['modelo']}\n\n")
        f.write("### Métricas principales:\n")
        for metric, value in best_model.items():
            if metric != "modelo":
                f.write(f"- **{metric}:** {round(value, 4)}\n")

    logger.info(f"Resumen de evaluación guardado en {summary_path}")


    return df_results


if __name__ == "__main__":
    results = evaluate_models()
    print(results)
