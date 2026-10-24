"""
7.inference.py
==============
Pembangkit lalu lintas untuk menguji sistem monitoring: mengirim permintaan ke exporter (POST /predict),
yang meneruskannya ke model server dan mencatat metrik Prometheus.

Mode:
    normal : baris acak dari data/test.csv (lalu lintas sehat)
    drift  : baris out-of-distribution (fitur kontinu digeser puluhan simpangan baku; menyerupai kode
             yang sangat panjang / bersarang dalam) -> memicu alert drift
    error  : payload dengan kolom yang salah -> model menolak (HTTP 4xx) -> memicu alert error
    slow   : beberapa permintaan batch besar sekaligus (paralel) -> antrean -> latensi tinggi
             -> memicu alert latensi
    mixed  : campuran semua mode (bobot dapat diatur di MIXED_WEIGHTS)

Contoh:
    python 7.inference.py                         # normal, tanpa batas waktu (Ctrl+C untuk berhenti)
    python 7.inference.py --mode drift --duration 120
    python 7.inference.py --mode error --duration 120
    python 7.inference.py --mode slow --duration 120            # batch 5000 baris x 6 permintaan paralel
    python 7.inference.py --mode slow --batch 20000 --threads 8  # bila alert latensi belum terpicu
    python 7.inference.py --mode mixed --duration 300
"""

import argparse
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent
CLASS_NAMES = ["O(1)", "O(log n)", "O(n)", "O(n log n)", "O(n^2)", "O(n^3)", "NP-hard/eksponensial"]
MIXED_WEIGHTS = {"normal": 0.70, "drift": 0.10, "error": 0.10, "slow": 0.10}
PRINT_LOCK = threading.Lock()


def build_payload(mode, df, continuous, batch):
    columns = df.columns.tolist()
    if mode == "normal":
        rows = df.sample(1).to_numpy()
    elif mode == "drift":
        sample = df.sample(1).copy()
        for c in continuous:
            sample[c] = sample[c] + random.uniform(15, 40)
        rows = sample.to_numpy()
    elif mode == "slow":
        rows = df.sample(batch, replace=True).to_numpy()
    elif mode == "error":
        return {"dataframe_split": {"columns": ["kolom_salah_1", "kolom_salah_2"], "data": [[1.0, 2.0]]}}
    else:
        raise ValueError(mode)
    return {"dataframe_split": {"columns": columns, "data": rows.tolist()}}


def send(mode, args, df, continuous, stats):
    """Kirim satu permintaan dan catat hasilnya."""
    payload = build_payload(mode, df, continuous, args.batch)
    t0 = time.perf_counter()
    try:
        res = requests.post(args.url, json=payload, timeout=120)
    except requests.RequestException as exc:
        with PRINT_LOCK:
            stats["failed"] += 1
            print(f"[{time.strftime('%X')}] Exporter di {args.url} belum menyala atau tidak dapat dihubungi "
                  f"({type(exc).__name__})")
        return False
    dt = (time.perf_counter() - t0) * 1000
    with PRINT_LOCK:
        stats["sent"] += 1
        if res.status_code == 200:
            preds = res.json().get("predictions", [])
            stats["ok"] += 1
            shown = CLASS_NAMES[int(preds[0])] if len(preds) == 1 else f"{len(preds)} prediksi"
            print(f"[{time.strftime('%X')}] {mode:<6} OK    {dt:7.0f} ms -> {shown}")
        else:
            stats["failed"] += 1
            print(f"[{time.strftime('%X')}] {mode:<6} GAGAL {dt:7.0f} ms -> HTTP {res.status_code}")
    return True


def main():
    p = argparse.ArgumentParser(description="Pembangkit lalu lintas untuk BigOracle.")
    p.add_argument("--mode", choices=["normal", "drift", "error", "slow", "mixed"], default="normal")
    p.add_argument("--url", default="http://127.0.0.1:8000/predict")
    p.add_argument("--data", default=str(BASE_DIR / "data" / "test.csv"))
    p.add_argument("--duration", type=float, default=0, help="detik; 0 = tanpa batas")
    p.add_argument("--min-delay", type=float, default=0.3)
    p.add_argument("--max-delay", type=float, default=1.2)
    p.add_argument("--batch", type=int, default=5000, help="baris per permintaan pada mode slow")
    p.add_argument("--threads", type=int, default=0,
                   help="permintaan paralel per putaran (bawaan: 1; mode slow: 6)")
    args = p.parse_args()
    threads = args.threads or (6 if args.mode == "slow" else 1)

    df = pd.read_csv(args.data).drop(columns=["target"], errors="ignore")
    continuous = [c for c in df.columns if df[c].nunique() > 2]
    print("=" * 72)
    print(f" BigOracle | inference | mode={args.mode} | paralel={threads} | {len(df):,} baris data uji")
    print(f" Target: {args.url}")
    print("=" * 72)

    stats = {"sent": 0, "ok": 0, "failed": 0}
    start = time.time()
    try:
        with ThreadPoolExecutor(max_workers=max(threads, 1)) as pool:
            while args.duration <= 0 or time.time() - start < args.duration:
                mode = args.mode
                if mode == "mixed":
                    mode = random.choices(list(MIXED_WEIGHTS), weights=MIXED_WEIGHTS.values())[0]
                results = list(pool.map(lambda _: send(mode, args, df, continuous, stats), range(threads)))
                if not any(results):
                    time.sleep(3)
                    continue
                time.sleep(random.uniform(args.min_delay, args.max_delay) if mode != "slow" else 0.05)
    except KeyboardInterrupt:
        print("\nDihentikan oleh pengguna.")
    print(f"\nRingkasan: {stats['sent']} permintaan | {stats['ok']} berhasil | {stats['failed']} gagal | "
          f"{time.time() - start:.0f} detik")


if __name__ == "__main__":
    main()
