# Workflow-CI: BigOracle Retraining Pipeline

CI pipeline that retrains the BigOracle time-complexity classifier through an **MLflow Project**, stores the resulting MLflow artifacts in this repository, and publishes the model as a **Docker image** to Docker Hub using `mlflow models build-docker`.

```
push to main / manual dispatch
   -> mlflow run MLProject            (retrain RandomForest on the preprocessed dataset)
   -> artifacts/latest/               (MLflow artifacts committed back to this repository)
   -> mlflow models build-docker      (image serving the model at port 8080)
   -> docker push                     (<user>/bigoracle:latest and :<run_id>)
```

## Repository Structure

```
Workflow-CI/
├── .github/workflows/ci.yml          # workflow executed by GitHub Actions
├── .workflow/ci.yml                  # copy of the workflow, kept for the required layout
├── MLProject/
│   ├── MLProject                     # MLflow Project definition (entry point + parameters)
│   ├── conda.yaml                    # environment definition (python 3.12.7, pinned dependencies)
│   ├── modelling.py                  # training script executed by the project
│   ├── requirements.txt              # pinned dependencies used by the CI runner
│   ├── codecomplex_preprocessing/    # preprocessed dataset (train.csv, test.csv, ...)
│   └── DockerHub.txt                 # link to the published Docker Hub repository
├── scripts/ci_utils.py               # latest-run lookup and artifact collection
└── artifacts/latest/                 # created by CI: MLflow artifacts of the latest run
```

## Workflow Steps

| #     | Step                        | Purpose                                                        |
| ----- | --------------------------- | -------------------------------------------------------------- |
| 1     | Set up job                  | runner provisioning (automatic)                                |
| 2     | Run actions/checkout@v3     | fetch the repository                                           |
| 3     | Set up Python 3.12.7        | Python with pip cache                                          |
| 4     | Check Env                   | print tool versions, verify dataset files and required secrets |
| 5     | Install dependencies        | `pip install -r MLProject/requirements.txt`                    |
| 6     | Set MLflow Tracking URI     | SQLite backend in the workspace (`mlflow.db`)                  |
| 7     | Run mlflow project          | `mlflow run MLProject --env-manager=local`                     |
| 8     | Get latest MLflow run_id    | exports `RUN_ID` to the job environment                        |
| 9     | Install Python dependencies | `tabulate` for the metrics summary                             |
| 10    | Upload to GitHub            | commit the run's artifacts to `artifacts/latest/`              |
| 11    | Build Docker Model          | `mlflow models build-docker --model-uri runs:/<RUN_ID>/model`  |
| 12    | Log in to Docker Hub        | `docker/login-action` with repository secrets                  |
| 13    | Tag Docker Image            | tag `latest` and `<RUN_ID>`, then push                         |
| 14-17 | Post steps, Complete job    | automatic cleanup                                              |

Triggers: `push` to `main` touching `MLProject/**`, `scripts/**` or the workflow file, and manual `workflow_dispatch`. The artifact commit uses `[skip ci]` and targets `artifacts/`, which is outside the trigger paths, so it does not retrigger the workflow.

## Setup

1. Copy the preprocessed dataset into `MLProject/codecomplex_preprocessing/` (`train.csv`, `test.csv`, `scaler.joblib`, `preprocessing_meta.json`).
2. Create the repository secrets under **Settings -> Secrets and variables -> Actions**:

   | Secret               | Value                                              |
   | -------------------- | -------------------------------------------------- |
   | `DOCKERHUB_USERNAME` | Docker Hub username                                |
   | `DOCKERHUB_TOKEN`    | Docker Hub access token with read/write permission |

3. Make sure **Settings -> Actions -> General -> Workflow permissions** allows read and write access (the workflow also requests `contents: write`).
4. Replace `<DOCKERHUB_USERNAME>` in `MLProject/DockerHub.txt`.

## Run Locally

```bash
pip install -r MLProject/requirements.txt
export MLFLOW_TRACKING_URI="sqlite:///$PWD/mlflow.db"
mlflow run MLProject --env-manager=local --experiment-name BigOracle_CI
mlflow run MLProject --env-manager=local --experiment-name BigOracle_CI -P n_estimators=200 -P max_depth=None
```

| Parameter          | Default                     | Description                                |
| ------------------ | --------------------------- | ------------------------------------------ |
| `data_dir`         | `codecomplex_preprocessing` | dataset directory relative to `MLProject/` |
| `n_estimators`     | `100`                       | number of trees                            |
| `max_depth`        | `"20"`                      | integer or `None`                          |
| `min_samples_leaf` | `2`                         | minimum samples per leaf                   |
| `seed`             | `42`                        | random seed                                |

The defaults keep `model.pkl` at about 30 MB. Unbounded depth with 200 trees reaches about 97 MB, which exceeds GitHub's per-file limit.

## Serve the Published Image

```bash
docker pull <DOCKERHUB_USERNAME>/bigoracle:latest
docker run --rm -p 5001:8080 <DOCKERHUB_USERNAME>/bigoracle:latest
curl -X POST http://127.0.0.1:5001/invocations \
     -H "Content-Type: application/json" \
     -d @artifacts/latest/model/serving_input_example.json
```

The model expects the 23 preprocessed (scaled) features, in the order stored in `preprocessing_meta.json`.

## Notes

- Each CI run commits about 30 MB of artifacts (roughly 7 MB compressed in the Git history), so the repository grows with every run. Reduce `n_estimators` or move `artifacts/` to Git LFS if the repository becomes large.
- The CI tracking store is ephemeral (a fresh SQLite file per run). Persistent experiment tracking lives in DagsHub (`Membangun_model`).
