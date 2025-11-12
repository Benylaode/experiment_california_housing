#!/usr/bin/env python3
import os
import time
import sys
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# === PROMETHEUS ===
from prometheus_client import Counter, Gauge, Summary, start_http_server

sys.stdout.reconfigure(line_buffering=True)

# ============================================================
# CONFIG
# ============================================================
PROMETHEUS_PORT = int(os.getenv("PROMETHEUS_PORT", "8002"))
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT", "california_housing_exp")

DATA_DIR = os.getenv("DATA_DIR", "../Experiment_California_Housing/preprosesing/california_housing_data/namadataset_preprocessing")
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/tmp/model_benchmark_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# PROMETHEUS SETUP
# ============================================================
start_http_server(PROMETHEUS_PORT)
print(f"📊 Prometheus aktif di port {PROMETHEUS_PORT}")

# Metrics definitions
EXPERIMENT_RUN_COUNT = Counter("mlflow_experiment_runs_total", "Total eksperimen dijalankan")
TRAINING_DURATION = Summary("model_training_duration_seconds", "Durasi training model")
MODEL_MSE = Gauge("model_mse_score", "Nilai MSE model terakhir", ["model"])
MODEL_R2 = Gauge("model_r2_score", "Nilai R2 Score model terakhir", ["model"])
MODEL_MAE = Gauge("model_mae_score", "Nilai MAE model terakhir", ["model"])

# ============================================================
# LOAD DATA
# ============================================================
print("📂 Memuat dataset...")
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

TARGET_COL = "MedHouseVal"
if TARGET_COL not in train_df.columns:
    raise ValueError(f"Kolom target '{TARGET_COL}' tidak ditemukan di {TRAIN_PATH}")

X_train = train_df.drop(columns=[TARGET_COL])
y_train = train_df[TARGET_COL]
X_test = test_df.drop(columns=[TARGET_COL])
y_test = test_df[TARGET_COL]

# ============================================================
# MLFLOW SETUP
# ============================================================
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)
print(f"🔍 MLflow tracking: {MLFLOW_TRACKING_URI} (experiment: {EXPERIMENT_NAME})")

# ============================================================
# MODELS
# ============================================================
models = {
    "LinearRegression": LinearRegression(),
    "DecisionTree": DecisionTreeRegressor(random_state=42),
    "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=1, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
}

# ============================================================
# TRAINING FUNCTION
# ============================================================
@TRAINING_DURATION.time()
def train_one_model(name, model, X_train, y_train, X_test, y_test):
    EXPERIMENT_RUN_COUNT.inc()
    print(f"\n🚀 Mulai training model: {name}")
    t0 = time.time()
    model.fit(X_train, y_train)
    duration = time.time() - t0
    print(f"✅ Selesai training {name} dalam {duration:.2f}s")

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"📊 Hasil {name} -> MSE: {mse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}")

    # Update Prometheus metrics
    MODEL_MSE.labels(model=name).set(mse)
    MODEL_MAE.labels(model=name).set(mae)
    MODEL_R2.labels(model=name).set(r2)

    # Log to MLflow
    with mlflow.start_run(run_name=f"{name}_run"):
        mlflow.log_param("model_name", name)
        mlflow.log_metric("mse", float(mse))
        mlflow.log_metric("mae", float(mae))
        mlflow.log_metric("r2", float(r2))
        mlflow.log_metric("training_duration_sec", float(duration))
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(model, artifact_path="model", signature=signature)

    return {"model": name, "mse": mse, "mae": mae, "r2": r2, "training_duration": duration}

# ============================================================
# RUN BENCHMARK
# ============================================================
def run_benchmark():
    results = []
    for name, model in models.items():
        res = train_one_model(name, model, X_train, y_train, X_test, y_test)
        results.append(res)

    results_df = pd.DataFrame(results).sort_values("mse")
    print("\n📋 Perbandingan metrik semua model:")
    print(results_df.to_string(index=False))

    summary_csv = os.path.join(OUTPUT_DIR, "benchmark_summary.csv")
    results_df.to_csv(summary_csv, index=False)

    # Save bar plots
    def save_bar(metric, title, filename):
        plt.figure(figsize=(8, 5))
        plt.bar(results_df["model"], results_df[metric])
        plt.title(title)
        plt.ylabel(metric)
        plt.xlabel("Model")
        plt.tight_layout()
        outpath = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(outpath)
        plt.close()
        return outpath

    mse_png = save_bar("mse", "Perbandingan MSE antar Model", "mse_comparison.png")
    mae_png = save_bar("mae", "Perbandingan MAE antar Model", "mae_comparison.png")
    r2_png = save_bar("r2", "Perbandingan R2 antar Model", "r2_comparison.png")

    with mlflow.start_run(run_name="benchmark_summary_run"):
        mlflow.log_artifact(summary_csv, artifact_path="summary")
        mlflow.log_artifact(mse_png, artifact_path="plots")
        mlflow.log_artifact(mae_png, artifact_path="plots")
        mlflow.log_artifact(r2_png, artifact_path="plots")

    print(f"🖼️ Grafik dan summary telah disimpan ke {OUTPUT_DIR} dan di-log ke MLflow.")

    return results_df

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    summary_df = run_benchmark()
    print(f"\n📡 Prometheus Metrics aktif di http://localhost:{PROMETHEUS_PORT}/metrics")
    print("⏳ Container tetap hidup untuk diekspos oleh Prometheus (CTRL+C untuk stop).")
    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        print("✋ Dihentikan oleh user.")
