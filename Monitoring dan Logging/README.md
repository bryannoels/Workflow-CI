# Monitoring dan Logging: BigOracle

Monitoring stack for the BigOracle time-complexity classifier: model serving, a Prometheus exporter, Prometheus, and a Grafana dashboard with alert rules.

```
7.inference.py --POST /predict--> 3.prometheus_exporter.py :8000 --POST /invocations--> model server :5001 (Docker image from Workflow-CI)
                                         ^
                       Prometheus :9090 scrapes /metrics every 5s  -->  Grafana :3000 (dashboard + alerts -> Discord)
```

## Files

| File | Purpose |
|---|---|
| `2.prometheus.yml` | Prometheus scrape configuration (target `host.docker.internal:8000`) |
| `3.prometheus_exporter.py` | Flask exporter: proxies `/predict` to the model, exposes `/metrics` |
| `7.inference.py` | Traffic generator: `normal`, `drift`, `error`, `slow`, `mixed` modes |
| `docker-compose.yml` | Prometheus and Grafana containers (pinned versions) |
| `build_dashboard.py` | Generates the Grafana dashboard JSON titled with the Dicoding username |
| `grafana/provisioning/` | Datasource and dashboard provisioning |
| `data/` | `train.csv` (drift reference) and `test.csv` (inference traffic), copied from `Workflow-CI/MLProject/codecomplex_preprocessing/` |
| `1.bukti_serving`, `4.`, `5.`, `6.` | Evidence screenshots |

## Metrics

| Group | Metric | Type |
|---|---|---|
| Requests | `bigoracle_requests_total{status}` | counter |
| | `bigoracle_request_latency_seconds` | histogram |
| | `bigoracle_model_inference_seconds` | histogram |
| | `bigoracle_errors_total{type}` | counter |
| | `bigoracle_requests_in_progress` | gauge |
| | `bigoracle_request_payload_bytes` | histogram |
| | `bigoracle_batch_size_rows` | histogram |
| Predictions | `bigoracle_predictions_total{predicted_class}` | counter |
| Data drift | `bigoracle_input_drift_score`, `bigoracle_input_drift_score_rolling` | gauge |
| | `bigoracle_input_rows_total`, `bigoracle_ood_rows_total` | counter |
| System | `bigoracle_model_up`, `bigoracle_cpu_usage_percent`, `bigoracle_memory_usage_percent`, `bigoracle_process_memory_bytes`, `bigoracle_uptime_seconds` | gauge |

**Drift score.** For every input row, the mean absolute z-score of the 14 continuous features relative to the training distribution (`data/train.csv`). A row is out-of-distribution (OOD) when the score exceeds 4.0. On the held-out test set the maximum normal score is 3.99, while rows shifted by 15 to 40 standard deviations score above 21.

## Alert Rules

| Alert | PromQL | Condition | Triggered by |
|---|---|---|---|
| High request latency | `histogram_quantile(0.95, sum by (le) (rate(bigoracle_request_latency_seconds_bucket[1m])))` | `> 0.5` for 30s | `7.inference.py --mode slow` |
| High error ratio | `sum(rate(bigoracle_errors_total[1m])) / clamp_min(sum(rate(bigoracle_requests_total[1m])), 0.000001)` | `> 0.2` for 30s | `7.inference.py --mode error` |
| Input drift (OOD) | `sum(rate(bigoracle_ood_rows_total[1m])) / clamp_min(sum(rate(bigoracle_input_rows_total[1m])), 0.000001)` | `> 0.3` for 30s | `7.inference.py --mode drift` |

Healthy traffic keeps p95 latency far below 0.5 s, and both ratios at 0.

## Run

```bash
pip install -r requirements.txt
cp ../Workflow-CI/MLProject/codecomplex_preprocessing/{train.csv,test.csv} data/

# terminal 1: model server (image published by Workflow-CI)
docker run --rm --name bigoracle-model -p 5001:8080 <DOCKERHUB_USERNAME>/bigoracle-model:latest

# terminal 2: exporter
python 3.prometheus_exporter.py

# terminal 3: dashboard titled with the Dicoding username, then Prometheus and Grafana
python build_dashboard.py --username <DICODING_USERNAME>
docker compose up -d

# terminal 4: traffic
python 7.inference.py --mode normal
```

Prometheus: http://127.0.0.1:9090 (Status -> Targets must show `bigoracle_exporter` UP). Grafana: http://127.0.0.1:3000 (`admin` / `admin`).

Alternative to Docker for serving: `mlflow models serve -m <Workflow-CI>/artifacts/latest/model -p 5001 --env-manager local`.

## Email Notifications (optional)

Grafana needs SMTP credentials to send alert emails. Copy `.env.example` to `.env`, fill in a Gmail address and an
[app password](https://myaccount.google.com/apppasswords) (2-step verification required), then run `docker compose up -d`.
In Grafana create a contact point of type **Email**. Keep `.env` out of version control and out of screenshots.

## Notes

- The exporter forwards the request body unchanged, so clients use the same JSON as the MLflow `/invocations` endpoint.
- Counters are initialised at zero for every label value, so panels do not appear empty before the first event.
- If the latency alert does not fire on a fast machine, increase the load: `python 7.inference.py --mode slow --batch 20000 --threads 8`.
