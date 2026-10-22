"""
modelling.py
============
Baseline klasifikasi kelas kompleksitas waktu kode sumber (7 kelas)
dengan MLflow autolog.

- Model    : RandomForestClassifier, tanpa hyperparameter tuning
- Data     : codecomplex_preprocessing/train.csv dan test.csv
- Tracking : MLFLOW_TRACKING_URI jika tersedia; jika tidak, SQLite lokal
- Logging  : mlflow.sklearn.autolog() + model dan evaluasi test secara manual

Jalankan langsung:
    python modelling.py

Jalankan sebagai MLflow Project:
    mlflow run MLProject --env-manager=local
"""

import os
import tempfile
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "codecomplex_preprocessing"

# Gunakan tracking URI dari environment (mis. CI/DagsHub), jika tersedia.
# Jika tidak, gunakan SQLite lokal di folder parent BASE_DIR.
TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{BASE_DIR.parent / 'mlflow.db'}",
)

EXPERIMENT_NAME = "BigOracle_Baseline"
RUN_NAME = "random_forest_baseline"

N_ESTIMATORS = 200
MAX_DEPTH = None
SEED = 42

CLASS_NAMES = [
    "O(1)",
    "O(log n)",
    "O(n)",
    "O(n log n)",
    "O(n^2)",
    "O(n^3)",
    "NP-hard/eksponensial",
]
CLASS_SHORT = [
    "O(1)",
    "O(log n)",
    "O(n)",
    "O(n log n)",
    "O(n^2)",
    "O(n^3)",
    "NP/exp",
]

TOTAL_STEPS = 7
T_START = time.time()


def step(n, text):
    print(f"\n[{n}/{TOTAL_STEPS}] {text}  (+{time.time() - T_START:.1f}s)")
    print("-" * 70)


def ok(text):
    print(f"  [OK] {text}")


# ---------------------------------------------------------------------------
# 1. Memuat data
# ---------------------------------------------------------------------------
print("=" * 70)
print(" BigOracle | Baseline RandomForest + MLflow autolog")
print("=" * 70)

step(1, "Memuat dataset hasil preprocessing")

if not (DATA_DIR / "train.csv").exists() or not (DATA_DIR / "test.csv").exists():
    raise SystemExit(
        f"\n[GAGAL] train.csv / test.csv tidak ditemukan di {DATA_DIR}\n"
        "Pastikan folder codecomplex_preprocessing/ berada di samping modelling.py.\n"
    )

train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

X_train = train.drop(columns="target")
y_train = train["target"]
X_test = test.drop(columns="target")
y_test = test["target"]

input_example = X_train.iloc[:5]

ok(f"Data latih : {X_train.shape[0]:,} baris x {X_train.shape[1]} fitur")
ok(f"Data uji   : {X_test.shape[0]:,} baris x {X_test.shape[1]} fitur")
ok(
    f"Nilai kosong: latih={int(X_train.isna().sum().sum())}, "
    f"uji={int(X_test.isna().sum().sum())}"
)

# ---------------------------------------------------------------------------
# 2. Menyiapkan MLflow
# ---------------------------------------------------------------------------
step(2, "Menyiapkan MLflow (tracking, experiment, autolog)")

mlflow.set_tracking_uri(TRACKING_URI)

# `mlflow run` menetapkan MLFLOW_RUN_ID dan mengelola run/experiment sendiri.
under_mlflow_project = "MLFLOW_RUN_ID" in os.environ

if not under_mlflow_project:
    experiment = mlflow.set_experiment(EXPERIMENT_NAME)
    ok(f"Experiment: {EXPERIMENT_NAME} (id={experiment.experiment_id})")
else:
    experiment = None
    ok("Run dan experiment dikelola oleh MLflow Project")

mlflow.sklearn.autolog(log_models=False)
ok("Autolog aktif (parameter dan metrik training; model disimpan manual)")

# ---------------------------------------------------------------------------
# 3. Run MLflow
# ---------------------------------------------------------------------------
with mlflow.start_run(
    run_name=None if under_mlflow_project else RUN_NAME
) as run:
    ok(f"Run dimulai: {run.info.run_id}")

    # -----------------------------------------------------------------------
    # 4. Pelatihan
    # -----------------------------------------------------------------------
    step(3, "Melatih RandomForestClassifier")

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=SEED,
        n_jobs=-1,
    )

    print(
        f"  Parameter: n_estimators={N_ESTIMATORS}, "
        f"max_depth={MAX_DEPTH}, random_state={SEED}"
    )
    print("  Melatih model ...")

    t0 = time.time()
    model.fit(X_train, y_train)

    fit_time = time.time() - t0
    ok(f"Pelatihan selesai dalam {fit_time:.1f} detik")

    train_acc = accuracy_score(y_train, model.predict(X_train))
    ok(f"Akurasi data latih: {train_acc:.4f}")

    # -----------------------------------------------------------------------
    # 5. Menyimpan model sebagai artefak MLflow
    # -----------------------------------------------------------------------
    step(4, "Menyimpan model sebagai artefak MLflow")

    with tempfile.TemporaryDirectory() as tmp:
        model_dir = Path(tmp) / "model"

        mlflow.sklearn.save_model(
            model,
            str(model_dir),
            signature=infer_signature(
                input_example,
                model.predict(input_example),
            ),
            input_example=input_example,
            serialization_format="cloudpickle",
        )

        mlflow.log_artifacts(str(model_dir), artifact_path="model")

    ok("Model tersimpan di artefak: model/")

    # -----------------------------------------------------------------------
    # 6. Evaluasi data uji dan metrik tambahan
    # -----------------------------------------------------------------------
    step(5, "Evaluasi pada data uji")

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    # Metrik evaluasi test tambahan, dicatat secara manual.
    mlflow.log_metrics(
        {
            "test_accuracy_score": acc,
            "test_macro_f1": macro_f1,
            "test_weighted_f1": weighted_f1,
            "train_test_accuracy_gap": train_acc - acc,
            "fit_time_seconds": fit_time,
        }
    )

    ok(f"Akurasi     : {acc:.4f}")
    ok(f"Macro-F1    : {macro_f1:.4f}")
    ok(f"Weighted-F1 : {weighted_f1:.4f}")
    ok(f"Train-test accuracy gap: {train_acc - acc:.4f}")

    # -----------------------------------------------------------------------
    # 7. Artefak evaluasi tambahan
    # -----------------------------------------------------------------------
    step(6, "Menyimpan artefak evaluasi")

    report = classification_report(
        y_test,
        y_pred,
        target_names=CLASS_NAMES,
        zero_division=0,
    )

    print("\n" + report)
    mlflow.log_text(report, "classification_report.txt")

    fig, ax = plt.subplots(figsize=(8.5, 7))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=CLASS_SHORT,
        normalize="true",
        values_format=".2f",
        cmap="Blues",
        ax=ax,
        colorbar=False,
    )

    ax.set_title("Confusion Matrix (data uji, dinormalisasi per kelas)")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()

    mlflow.log_figure(fig, "confusion_matrix_test.png")
    plt.close(fig)

    ok("Artefak tambahan: classification_report.txt dan confusion_matrix_test.png")

    run_id = run.info.run_id

# ---------------------------------------------------------------------------
# 8. Ringkasan
# ---------------------------------------------------------------------------
step(7, "Ringkasan")

print(f"  Run ID     : {run_id}")
print(f"  Tracking   : {TRACKING_URI}")
print(f"  Akurasi uji: {acc:.4f} | Macro-F1 uji: {macro_f1:.4f}")
print(f"  Total waktu: {time.time() - T_START:.1f} detik")

if experiment is not None:
    run_url = (
        f"{TRACKING_URI.rstrip('/')}/#/experiments/"
        f"{experiment.experiment_id}/runs/{run_id}"
    )
    print(f"\n  Buka run ini di MLflow UI:\n  {run_url}")

print("=" * 70)