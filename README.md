# CropVision API

**Backend service** for **CropVision**, a plant disease detection platform that pairs a PyTorch computer-vision model with a FastAPI service, and is built to run predictions both in the cloud and directly on edge hardware such as a **Raspberry Pi**.

🔗 Frontend client: [crop-vision-frontend](https://github.com/tahirmohammedaman/crop-vision-frontend)

## What it does

Upload a photo of a plant leaf and get back the crop, the detected disease (or a healthy verdict), and a confidence score — powered by a self-trained CNN. Predictions are logged, searchable, and can be routed through a human review queue for low-confidence cases, which doubles as a feedback loop for the model.

## Features

- **Disease classification** — a custom **ResNet9** convolutional network (trained from scratch, notebook included) classifies 38 crop/disease combinations across 14 crop types.
- **Prediction history** — every inference is persisted with confidence scores, per-class probability breakdowns, tags, and source metadata, with paginated search and filtering by crop, disease, tag, date range, confidence, and origin.
- **Human-in-the-loop review** — a low-confidence review queue lets a user confirm or correct predictions, which feeds back into per-model accuracy stats.
- **JWT authentication** — user accounts secured with hashed passwords and bearer tokens.
- **Edge device fleet management** — devices are first-class citizens: register a device, issue it a scoped API key, and track its health independently of the web UI.
- **Model introspection & stats** — endpoints expose the active model's metadata and running accuracy metrics computed from confirmed/corrected predictions.
- **Dockerized & CI-checked** — ships with a `Dockerfile` and a GitHub Actions pipeline.

## Edge computing / Raspberry Pi integration

CropVision was designed to run beyond the browser — as a field-deployed sensor. The API has a dedicated device-auth layer (`X-Device-ID` / `X-Device-Key` headers, independent of user JWTs) and two companion scripts meant to run on a Raspberry Pi (or any Linux-capable microcontroller board) pointed at a camera module:

- **`pi_predict_edge.py`** — captures/reads an image on the device and submits it as a registered edge device. Supports two delivery modes:
  - `server_edge` — the device sends the raw image to the API, which runs inference server-side (useful for low-power boards without a GPU).
  - `device_offline` — the device runs the model **locally** (e.g. via a distilled/quantized checkpoint) and syncs the already-computed label, confidence, and probability vector back once connectivity is available — enabling fully offline field operation.
- **`pi_heartbeat_script.py`** — a background daemon that reports device telemetry (CPU/mem/disk usage, load average, uptime, CPU temperature, and camera availability via `vcgencmd`/`libcamera`) to the API's heartbeat endpoint every 10 minutes, so device health is visible from the web dashboard without SSHing in.

Every prediction is auto-tagged with its origin (`server_web`, `server_edge`, `device_offline`) and originating device ID, so cloud-submitted and edge-submitted predictions live in the same searchable history.

## Tech stack

| Layer | Technology |
|---|---|
| API framework | FastAPI (Python 3.11), Uvicorn |
| ML / inference | PyTorch, Torchvision (custom ResNet9) |
| Database | PostgreSQL via SQLAlchemy 2.0 |
| Auth | JWT (python-jose), Passlib/bcrypt, per-device API keys |
| Validation | Pydantic v2 / pydantic-settings |
| Image handling | Pillow |
| Edge scripts | Plain Python + `requests` + `psutil` (Raspberry Pi target) |
| Infra | Docker, GitHub Actions CI |

## API surface

- `POST /v1/predict` — run inference on an uploaded image (user or device auth)
- `POST /v1/devices/{id}/predictions` — sync an offline-computed prediction from a device
- `POST /v1/predictions/{id}/feedback` — confirm or correct a prediction
- `GET /v1/history`, `/v1/review/queue`, `/v1/stats`, `/v1/tags`, `/v1/crops`, `/v1/diseases`
- `POST/GET/PATCH /v1/devices`, `POST /v1/devices/{id}/heartbeat` — device fleet management
- `GET /v1/model/info` — active model metadata
- Full interactive docs at `/docs` (Swagger UI)

## Getting started

```bash
pip install -r requirements.txt
cp .env.example .env   # configure DATABASE_URL, SECRET_KEY, MODELS_PATH, etc.
uvicorn app.main:app --reload
```

Or with Docker:

```bash
docker build -t crop-vision-backend .
docker run -p 8000:8000 --env-file .env crop-vision-backend
```

The trained model weights and `model_info.json` live under `models/`; the training notebook is in `notebook/`.
