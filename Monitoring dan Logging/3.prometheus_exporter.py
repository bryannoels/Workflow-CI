"""
3.prometheus_exporter.py
========================
Prometheus exporter untuk monitoring sistem machine learning BigOracle.

Exporter berperan sebagai lapisan di depan model server (container Docker hasil Workflow-CI atau
`mlflow models serve`): klien mengirim permintaan ke exporter, exporter meneruskannya ke model,
dan mencatat metrik performa serta kualitas data masukan.

    klien (7.inference.py)  --POST /predict-->  exporter :8000  --POST /invocations-->  model server :5001
                                                   |
                                          GET /metrics  <--  Prometheus

Endpoint:
    POST /predict   format sama dengan MLflow /invocations (dataframe_split / dataframe_records)
    GET  /metrics   metrik Prometheus
    GET  /health    status exporter dan model server

Metrik (prefix bigoracle_):
    Permintaan : requests_total, request_latency_seconds, model_inference_seconds, errors_total,
                 requests_in_progress, request_payload_bytes, batch_size_rows
    Prediksi   : predictions_total (per kelas kompleksitas)
    Drift      : input_drift_score, input_drift_score_rolling, input_rows_total, ood_rows_total
    Sistem     : model_up, cpu_usage_percent, memory_usage_percent, process_memory_bytes, uptime_seconds

Cara menjalankan:
    python 3.prometheus_exporter.py
    python 3.prometheus_exporter.py --model-url http://127.0.0.1:5001/invocations --port 8000
"""

import argparse
import json
import os
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import requests
from flask import Flask, Response, jsonify, request
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

BASE_DIR = Path(__file__).resolve().parent
CLASS_NAMES = ["O(1)", "O(log n)", "O(n)", "O(n log n)", "O(n^2)", "O(n^3)", "NP-hard/eksponensial"]
ERROR_TYPES = ["invalid_request", "connection", "timeout", "model_4xx", "model_5xx", "bad_response"]
START_TIME = time.time()

# ---------------------------------------------------------------------------
# Definisi metrik
# ---------------------------------------------------------------------------
REQUESTS = Counter("bigoracle_requests_total", "Jumlah permintaan prediksi", ["status"])
LATENCY = Histogram("bigoracle_request_latency_seconds", "Latensi end-to-end permintaan prediksi",
                    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10))
INFERENCE = Histogram("bigoracle_model_inference_seconds", "Waktu respons model server",
                      buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10))
ERRORS = Counter("bigoracle_errors_total", "Jumlah kesalahan berdasarkan jenis", ["type"])
IN_PROGRESS = Gauge("bigoracle_requests_in_progress", "Permintaan yang sedang diproses")
PAYLOAD_BYTES = Histogram("bigoracle_request_payload_bytes", "Ukuran badan permintaan (byte)",
                          buckets=(256, 512, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304))
BATCH_ROWS = Histogram("bigoracle_batch_size_rows", "Jumlah baris data per permintaan",
                       buckets=(1, 2, 5, 10, 50, 100, 500, 1000, 5000))
PREDICTIONS = Counter("bigoracle_predictions_total", "Jumlah prediksi per kelas kompleksitas", ["predicted_class"])
DRIFT_LAST = Gauge("bigoracle_input_drift_score", "Skor drift permintaan terakhir (rata-rata |z-score| fitur kontinu)")
DRIFT_ROLLING = Gauge("bigoracle_input_drift_score_rolling", "Rata-rata skor drift pada 50 baris terakhir")
INPUT_ROWS = Counter("bigoracle_input_rows_total", "Jumlah baris data masukan")
OOD_ROWS = Counter("bigoracle_ood_rows_total", "Jumlah baris data masukan di luar distribusi latih")
MODEL_UP = Gauge("bigoracle_model_up", "1 bila model server sehat, 0 bila tidak")
CPU_PERCENT = Gauge("bigoracle_cpu_usage_percent", "Penggunaan CPU sistem (%)")
MEM_PERCENT = Gauge("bigoracle_memory_usage_percent", "Penggunaan memori sistem (%)")
PROC_MEM = Gauge("bigoracle_process_memory_bytes", "Memori resident proses exporter (byte)")
UPTIME = Gauge("bigoracle_uptime_seconds", "Lama exporter berjalan (detik)")
UPTIME.set_function(lambda: time.time() - START_TIME)

# deret waktu langsung terlihat dengan nilai 0 sejak awal (grafik tidak "kosong" sebelum ada kejadian)
for _status in ("success", "error"):
    REQUESTS.labels(status=_status)
for _t in ERROR_TYPES:
    ERRORS.labels(type=_t)
for _c in CLASS_NAMES + ["unknown"]:
    PREDICTIONS.labels(predicted_class=_c)

app = Flask(__name__)
CFG = {}            # diisi pada saat start
REF = {}            # statistik distribusi latih
ROLLING = deque(maxlen=50)
ROLLING_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Referensi distribusi latih untuk deteksi drift
# ---------------------------------------------------------------------------
def load_reference(path):
    path = Path(path)
    if not path.exists():
        print(f"[PERINGATAN] File referensi {path} tidak ditemukan; metrik drift dinonaktifkan.")
        return None
    df = pd.read_csv(path).drop(columns=["target"], errors="ignore")
    mu, sd = df.mean(), df.std(ddof=0)
    cont = [c for c in df.columns if df[c].nunique() > 2 and sd[c] > 0]   # fitur kontinu (bukan biner)
    return {"mu": mu[cont], "sd": sd[cont], "cols": cont}


def rows_from_payload(payload):
    """Ubah payload MLflow (dataframe_split / dataframe_records) menjadi DataFrame; None bila tidak dikenali."""
    if isinstance(payload, dict):
        if "dataframe_split" in payload:
            d = payload["dataframe_split"]
            return pd.DataFrame(d["data"], columns=d["columns"])
        if "dataframe_records" in payload:
            return pd.DataFrame(payload["dataframe_records"])
    return None


def drift_scores(df):
    if not REF or df is None:
        return None
    cols = [c for c in REF["cols"] if c in df.columns]
    if not cols:
        return None
    x = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    z = np.abs((x - REF["mu"][cols].to_numpy()) / REF["sd"][cols].to_numpy())
    return np.nanmean(z, axis=1)


def record_drift(df):
    scores = drift_scores(df)
    if scores is None or len(scores) == 0:
        return
    scores = scores[np.isfinite(scores)]
    if len(scores) == 0:
        return
    INPUT_ROWS.inc(len(scores))
    OOD_ROWS.inc(int((scores > CFG["ood_threshold"]).sum()))
    DRIFT_LAST.set(float(scores.mean()))
    with ROLLING_LOCK:
        ROLLING.extend(float(s) for s in scores)
        DRIFT_ROLLING.set(float(np.mean(ROLLING)))


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------
def _error(err_type, status_code, message):
    ERRORS.labels(type=err_type).inc()
    return jsonify({"error": err_type, "message": message}), status_code


@app.route("/predict", methods=["POST"])
def predict():
    t_start = time.perf_counter()
    IN_PROGRESS.inc()
    status = "error"
    try:
        raw = request.get_data()
        PAYLOAD_BYTES.observe(len(raw))
        try:
            payload = json.loads(raw)
        except ValueError:
            return _error("invalid_request", 400, "Badan permintaan bukan JSON yang valid")

        try:
            df = rows_from_payload(payload)
            if df is not None:
                BATCH_ROWS.observe(len(df))
                record_drift(df)
        except Exception as exc:    # payload aneh tidak boleh menjatuhkan exporter; model server yang menilai
            print(f"[peringatan] gagal memproses payload untuk metrik drift: {exc}")

        t_model = time.perf_counter()
        try:
            resp = requests.post(CFG["model_url"], data=raw, headers={"Content-Type": "application/json"},
                                 timeout=CFG["timeout"])
        except requests.Timeout:
            return _error("timeout", 504, "Model server tidak merespons tepat waktu")
        except requests.ConnectionError:
            return _error("connection", 503, f"Model server tidak dapat dihubungi di {CFG['model_url']}")
        INFERENCE.observe(time.perf_counter() - t_model)

        if resp.status_code != 200:
            ERRORS.labels(type="model_4xx" if resp.status_code < 500 else "model_5xx").inc()
            return Response(resp.content, status=resp.status_code, mimetype="application/json")

        try:
            preds = resp.json().get("predictions", [])
        except ValueError:
            return _error("bad_response", 502, "Respons model bukan JSON yang valid")
        for p in preds:
            label = CLASS_NAMES[int(p)] if isinstance(p, (int, float)) and 0 <= int(p) < len(CLASS_NAMES) else "unknown"
            PREDICTIONS.labels(predicted_class=label).inc()

        status = "success"
        return Response(resp.content, status=200, mimetype="application/json")
    finally:
        LATENCY.observe(time.perf_counter() - t_start)
        REQUESTS.labels(status=status).inc()
        IN_PROGRESS.dec()


@app.route("/metrics")
def metrics():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


@app.route("/health")
def health():
    return jsonify({"exporter": "ok", "model_up": bool(MODEL_UP._value.get()), "model_url": CFG["model_url"],
                    "drift_reference": bool(REF)})


@app.route("/")
def index():
    return jsonify({"service": "bigoracle-prometheus-exporter", "endpoints": ["/predict", "/metrics", "/health"]})


# ---------------------------------------------------------------------------
# Pengumpul metrik latar belakang
# ---------------------------------------------------------------------------
def system_collector():
    proc = psutil.Process(os.getpid())
    psutil.cpu_percent(interval=None)
    while True:
        CPU_PERCENT.set(psutil.cpu_percent(interval=None))
        MEM_PERCENT.set(psutil.virtual_memory().percent)
        PROC_MEM.set(proc.memory_info().rss)
        time.sleep(2)


def model_health_checker():
    ping_url = CFG["model_url"].rsplit("/", 1)[0] + "/ping"
    while True:
        try:
            MODEL_UP.set(1 if requests.get(ping_url, timeout=2).status_code == 200 else 0)
        except requests.RequestException:
            MODEL_UP.set(0)
        time.sleep(5)


def parse_args():
    p = argparse.ArgumentParser(description="Prometheus exporter untuk BigOracle.")
    p.add_argument("--model-url", default=os.getenv("MODEL_URL", "http://127.0.0.1:5001/invocations"))
    p.add_argument("--port", type=int, default=int(os.getenv("EXPORTER_PORT", "8000")))
    p.add_argument("--reference-csv", default=os.getenv("REFERENCE_CSV", str(BASE_DIR / "data" / "train.csv")),
                   help="train.csv hasil preprocessing, dipakai sebagai referensi distribusi (drift)")
    p.add_argument("--ood-threshold", type=float, default=float(os.getenv("OOD_THRESHOLD", "4.0")),
                   help="baris dianggap out-of-distribution bila rata-rata |z-score| melebihi nilai ini")
    p.add_argument("--timeout", type=float, default=30.0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    CFG.update(model_url=args.model_url, ood_threshold=args.ood_threshold, timeout=args.timeout)
    ref = load_reference(args.reference_csv)
    if ref:
        REF.update(ref)
    print("=" * 70)
    print(" BigOracle | Prometheus exporter")
    print("=" * 70)
    print(f"  Model server : {CFG['model_url']}")
    print(f"  Metrik       : http://127.0.0.1:{args.port}/metrics")
    print(f"  Prediksi     : POST http://127.0.0.1:{args.port}/predict")
    print(f"  Drift        : {'aktif, ' + str(len(REF['cols'])) + ' fitur kontinu' if REF else 'nonaktif'} "
          f"(ambang OOD = {CFG['ood_threshold']})")
    print("=" * 70)
    threading.Thread(target=system_collector, daemon=True).start()
    threading.Thread(target=model_health_checker, daemon=True).start()
    app.run(host="0.0.0.0", port=args.port, threaded=True)
