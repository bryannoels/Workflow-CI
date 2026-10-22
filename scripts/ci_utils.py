"""
scripts/ci_utils.py
===================
Skrip bantu untuk workflow CI.

    python scripts/ci_utils.py latest-run --experiment BigOracle_CI
        -> mencetak RUN_ID=..., EXPERIMENT_ID=..., ARTIFACT_URI=... (siap ditambahkan ke $GITHUB_ENV)

    python scripts/ci_utils.py collect --run-id <RUN_ID> --out artifacts/latest
        -> mengunduh seluruh artefak run (termasuk model/), menulis run_info.json dan metrics.md,
           dan menambahkan ringkasan metrik ke $GITHUB_STEP_SUMMARY bila tersedia

Tracking URI dibaca dari variabel lingkungan MLFLOW_TRACKING_URI.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient


def latest_run(experiment_name):
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        sys.exit(f"Experiment '{experiment_name}' tidak ditemukan pada {mlflow.get_tracking_uri()}")
    runs = client.search_runs([experiment.experiment_id], filter_string="attributes.status = 'FINISHED'",
                              order_by=["attributes.start_time DESC"], max_results=1)
    if not runs:
        sys.exit(f"Tidak ada run berstatus FINISHED pada experiment '{experiment_name}'")
    run = runs[0]
    print(f"RUN_ID={run.info.run_id}")
    print(f"EXPERIMENT_ID={experiment.experiment_id}")
    print(f"ARTIFACT_URI={run.info.artifact_uri}")


def collect(run_id, out_dir):
    out = Path(out_dir)
    if out.exists():
        import shutil
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    client = MlflowClient()
    run = client.get_run(run_id)
    mlflow.artifacts.download_artifacts(run_id=run_id, dst_path=str(out))

    info = {
        "run_id": run_id,
        "run_name": run.info.run_name,
        "experiment_id": run.info.experiment_id,
        "status": run.info.status,
        "collected_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": os.getenv("GITHUB_SHA"),
        "params": run.data.params,
        "metrics": run.data.metrics,
    }
    (out / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    metrics = pd.DataFrame({"metrik": list(run.data.metrics), "nilai": [round(v, 4) for v in run.data.metrics.values()]})
    table = metrics.to_markdown(index=False)
    md = f"## Hasil training (run `{run_id}`)\n\n{table}\n"
    (out / "metrics.md").write_text(md, encoding="utf-8")

    summary = os.getenv("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write(md)

    files = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())
    print(f"Artefak run {run_id} tersimpan di {out} ({len(files)} berkas):")
    for name in files:
        print(f"  - {name}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("latest-run")
    a.add_argument("--experiment", required=True)
    b = sub.add_parser("collect")
    b.add_argument("--run-id", required=True)
    b.add_argument("--out", default="artifacts/latest")
    args = p.parse_args()
    if args.cmd == "latest-run":
        latest_run(args.experiment)
    else:
        collect(args.run_id, args.out)


if __name__ == "__main__":
    main()
