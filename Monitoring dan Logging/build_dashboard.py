"""
build_dashboard.py
==================
Membuat berkas dashboard Grafana (grafana/dashboards/bigoracle.json) dengan judul yang memuat
username Dicoding, sesuai ketentuan Kriteria 4.

    python build_dashboard.py --username <username_dicoding>

Dashboard memuat 17 panel yang mencakup 15 metrik berbeda (syarat Advance: minimal 10).
Grafana membaca ulang berkas ini otomatis (sekitar 10 detik) setelah dibuat ulang.
"""

import argparse
import json
from pathlib import Path

DS = {"type": "prometheus", "uid": "prometheus"}
_next_id = [0]


def _pid():
    _next_id[0] += 1
    return _next_id[0]


def target(expr, legend="", ref="A"):
    return {"refId": ref, "datasource": DS, "expr": expr, "legendFormat": legend, "range": True, "instant": False}


def steps(*pairs):
    """steps((None,'green'),(10,'yellow'),(20,'red'))"""
    return {"mode": "absolute", "steps": [{"color": c, "value": v} for v, c in pairs]}


def stat(title, desc, expr, unit, x, y, w=4, h=4, thresholds=None, mappings=None, decimals=None):
    defaults = {"unit": unit, "thresholds": thresholds or steps((None, "green")), "mappings": mappings or []}
    if decimals is not None:
        defaults["decimals"] = decimals
    return {
        "id": _pid(), "type": "stat", "title": title, "description": desc, "datasource": DS,
        "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "targets": [target(expr)],
        "fieldConfig": {"defaults": defaults, "overrides": []},
        "options": {"reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
                    "colorMode": "value", "graphMode": "area", "textMode": "auto", "justifyMode": "auto"},
    }


def timeseries(title, desc, targets, unit, x, y, w=8, h=8, stacked=False, thresholds=None, ymin=None):
    defaults = {"unit": unit, "color": {"mode": "palette-classic"},
                "custom": {"drawStyle": "line", "lineWidth": 2, "fillOpacity": 15, "showPoints": "never",
                           "spanNulls": True,
                           "stacking": {"mode": "normal" if stacked else "none", "group": "A"},
                           "thresholdsStyle": {"mode": "line" if thresholds else "off"}},
                "thresholds": thresholds or steps((None, "green"))}
    if ymin is not None:
        defaults["min"] = ymin
    return {
        "id": _pid(), "type": "timeseries", "title": title, "description": desc, "datasource": DS,
        "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "targets": [target(e, l, chr(65 + i)) for i, (e, l) in enumerate(targets)],
        "fieldConfig": {"defaults": defaults, "overrides": []},
        "options": {"legend": {"displayMode": "list", "placement": "bottom", "showLegend": True},
                    "tooltip": {"mode": "multi", "sort": "none"}},
    }


def build(username):
    _next_id[0] = 0
    ratio = "clamp_min(sum(rate({d}[1m])), 0.000001)"
    panels = [
        # ---- baris 1: ringkasan ----
        stat("Total Requests (bigoracle_requests_total)", "Total permintaan prediksi sejak exporter dijalankan",
             "sum(bigoracle_requests_total)", "short", 0, 0),
        stat("Model Server (bigoracle_model_up)", "1 = model server sehat, 0 = tidak dapat dihubungi",
             "bigoracle_model_up", "none", 4, 0,
             thresholds=steps((None, "red"), (1, "green")),
             mappings=[{"type": "value", "options": {"0": {"text": "DOWN", "color": "red"},
                                                     "1": {"text": "UP", "color": "green"}}}]),
        stat("Uptime (bigoracle_uptime_seconds)", "Lama exporter berjalan",
             "bigoracle_uptime_seconds", "s", 8, 0),
        stat("Error Ratio (bigoracle_errors_total)", "Persentase permintaan gagal pada 1 menit terakhir",
             "100 * sum(rate(bigoracle_errors_total[1m])) / " + ratio.format(d="bigoracle_requests_total"),
             "percent", 12, 0, thresholds=steps((None, "green"), (10, "yellow"), (20, "red")), decimals=1),
        stat("OOD Ratio (bigoracle_ood_rows_total)", "Persentase baris masukan di luar distribusi latih (1 menit terakhir)",
             "100 * sum(rate(bigoracle_ood_rows_total[1m])) / " + ratio.format(d="bigoracle_input_rows_total"),
             "percent", 16, 0, thresholds=steps((None, "green"), (10, "yellow"), (30, "red")), decimals=1),
        stat("In-flight Requests (bigoracle_requests_in_progress)", "Permintaan yang sedang diproses",
             "bigoracle_requests_in_progress", "short", 20, 0),

        # ---- baris 2: performa permintaan ----
        timeseries("Request Throughput (bigoracle_requests_total)", "Permintaan per detik berdasarkan status",
                   [("sum by (status) (rate(bigoracle_requests_total[1m]))", "{{status}}")], "reqps", 0, 4),
        timeseries("P95 Request Latency (bigoracle_request_latency_seconds)",
                   "Persentil ke-95 latensi end-to-end; garis merah = ambang alert (0,5 detik)",
                   [("histogram_quantile(0.95, sum by (le) (rate(bigoracle_request_latency_seconds_bucket[1m])))", "p95")],
                   "s", 8, 4, thresholds=steps((None, "green"), (0.5, "red")), ymin=0),
        timeseries("Avg Model Inference Time (bigoracle_model_inference_seconds)", "Rata-rata waktu respons model server",
                   [("sum(rate(bigoracle_model_inference_seconds_sum[1m])) / "
                     + ratio.format(d="bigoracle_model_inference_seconds_count"), "rata-rata")], "s", 16, 4, ymin=0),

        # ---- baris 3: prediksi, drift, kesalahan ----
        timeseries("Predictions by Class (bigoracle_predictions_total)", "Distribusi prediksi per kelas kompleksitas",
                   [("sum by (predicted_class) (rate(bigoracle_predictions_total[1m]))", "{{predicted_class}}")],
                   "ops", 0, 12, stacked=True),
        timeseries("Input Drift Score (bigoracle_input_drift_score_rolling)",
                   "Rata-rata |z-score| fitur kontinu terhadap data latih; garis merah = ambang OOD (4)",
                   [("bigoracle_input_drift_score_rolling", "rata-rata 50 baris terakhir"),
                    ("bigoracle_input_drift_score", "permintaan terakhir")],
                   "none", 8, 12, thresholds=steps((None, "green"), (4, "red")), ymin=0),
        timeseries("Errors by Type (bigoracle_errors_total)", "Kesalahan per detik berdasarkan jenis",
                   [("sum by (type) (rate(bigoracle_errors_total[1m]))", "{{type}}")], "ops", 16, 12, ymin=0),

        # ---- baris 4: ukuran permintaan dan CPU ----
        timeseries("Avg Batch Size (bigoracle_batch_size_rows)", "Rata-rata jumlah baris per permintaan",
                   [("sum(rate(bigoracle_batch_size_rows_sum[1m])) / "
                     + ratio.format(d="bigoracle_batch_size_rows_count"), "baris/permintaan")], "short", 0, 20, ymin=0),
        timeseries("Avg Request Payload (bigoracle_request_payload_bytes)", "Rata-rata ukuran badan permintaan",
                   [("sum(rate(bigoracle_request_payload_bytes_sum[1m])) / "
                     + ratio.format(d="bigoracle_request_payload_bytes_count"), "byte")], "bytes", 8, 20, ymin=0),
        timeseries("CPU Usage (bigoracle_cpu_usage_percent)", "Penggunaan CPU sistem",
                   [("bigoracle_cpu_usage_percent", "CPU")], "percent", 16, 20, ymin=0),

        # ---- baris 5: memori ----
        timeseries("Memory Usage (bigoracle_memory_usage_percent)", "Penggunaan memori sistem",
                   [("bigoracle_memory_usage_percent", "memori sistem")], "percent", 0, 28, w=12, ymin=0),
        timeseries("Exporter Process Memory (bigoracle_process_memory_bytes)", "Memori resident proses exporter",
                   [("bigoracle_process_memory_bytes", "RSS")], "bytes", 12, 28, w=12, ymin=0),
    ]
    return {
        "uid": "bigoracle-monitoring",
        "title": f"{username} - BigOracle ML Monitoring",
        "tags": ["bigoracle", "mlops"],
        "timezone": "browser",
        "schemaVersion": 39,
        "version": 1,
        "editable": True,
        "refresh": "5s",
        "time": {"from": "now-15m", "to": "now"},
        "templating": {"list": []},
        "annotations": {"list": []},
        "panels": panels,
    }


def main():
    p = argparse.ArgumentParser(description="Membuat dashboard Grafana BigOracle.")
    p.add_argument("--username", required=True, help="username akun Dicoding Anda")
    p.add_argument("--out", default=str(Path(__file__).resolve().parent / "grafana" / "dashboards" / "bigoracle.json"))
    args = p.parse_args()
    username = args.username.strip()
    if not username:
        raise SystemExit("--username tidak boleh kosong")
    dash = build(username)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dash, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Dashboard '{dash['title']}' ditulis ke {out}")
    print(f"{len(dash['panels'])} panel:")
    for p_ in dash["panels"]:
        print(f"  {p_['id']:>2}. [{p_['type']}] {p_['title']}")


if __name__ == "__main__":
    main()
