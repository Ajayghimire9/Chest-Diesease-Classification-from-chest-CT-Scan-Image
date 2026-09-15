# MedVision — Computer Vision Inference Platform

> Production-oriented ML engineering for medical-image classification research.

This repository started as a notebook-driven chest CT classification project. It has been reorganized around a clearer engineering boundary: **image validation and preprocessing → model inference → confidence policy → API response → operational telemetry and audit metadata**.

The goal is not to present a small CNN as a clinical product. The goal is to demonstrate how a computer-vision model can be packaged and operated as a reliable service, while keeping the model itself replaceable.

## What this project demonstrates

- Typed request/response contracts with Pydantic
- Deterministic image decoding and normalization
- Explicit serving policies for dimensions and payload size
- Replaceable PyTorch model runner
- Confidence-aware routing (`accepted` vs `review`)
- Request IDs for traceability
- Privacy-aware audit events that do not store image bytes
- Prometheus request, error, and latency metrics
- FastAPI inference service
- Dockerized deployment
- Docker Compose development stack with Prometheus
- Automated Ruff and Pytest quality gates
- `src/` package layout and modern Python packaging

## Architecture

```text
                         ┌─────────────────────┐
                         │  Image Client / UI   │
                         └──────────┬──────────┘
                                    │ image bytes
                                    ▼
                         ┌─────────────────────┐
                         │  FastAPI Gateway    │
                         │ request validation  │
                         └──────────┬──────────┘
                                    ▼
                         ┌─────────────────────┐
                         │ Preprocessing Layer │
                         │ decode / normalize  │
                         └──────────┬──────────┘
                                    ▼
                         ┌─────────────────────┐
                         │   Model Runner      │
                         │     PyTorch CNN     │
                         └──────────┬──────────┘
                                    ▼
                         ┌─────────────────────┐
                         │ Confidence Policy   │
                         │ accept / human      │
                         │ review               │
                         └───────┬─────┬───────┘
                                 │     │
                    ┌────────────┘     └────────────┐
                    ▼                               ▼
             JSON prediction                 Audit metadata
                                                    │
                                                    ▼
                                             Prometheus metrics
```

## Why the serving layer matters

A model is only one component of an ML system. In production, failures often happen around the model: malformed inputs, unexpected image dimensions, excessive payloads, missing observability, ambiguous confidence, or an inability to trace a request to the model version that produced it.

MedVision makes those boundaries explicit. The API does not persist uploaded images, and low-confidence predictions are routed to `review` rather than being presented as a definitive result.

## API

Start the service locally:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
uvicorn medvision.api:app --reload
```

Health check:

```bash
curl http://localhost:8000/health
```

Prediction endpoint:

```bash
curl -X POST http://localhost:8000/v1/predict \
  -F 'file=@sample.png'
```

Operational metrics:

```bash
curl http://localhost:8000/metrics
```

The service returns a request ID, model version, confidence, latency, and an explicit serving status. A low-confidence result is returned as `review` instead of silently treating it as an accepted prediction.

## Local infrastructure

Run the API and Prometheus together:

```bash
docker compose up --build
```

- API: `http://localhost:8000`
- Prometheus: `http://localhost:9090`

## Model configuration

The serving layer is deliberately separated from the model implementation. A compatible PyTorch state dictionary can be supplied through:

```bash
export MODEL_WEIGHTS=/path/to/model.pt
export MODEL_VERSION=cnn-v1
```

Without weights, the service uses the small reference network included in the repository. **It is an engineering reference model, not a validated clinical model.**

## Testing and quality

```bash
ruff check src tests
pytest -q
```

CI runs these checks automatically for changes targeting `main`.

## Repository structure

```text
.
├── src/medvision/
│   ├── api.py             # HTTP inference boundary
│   ├── audit.py           # minimal operational audit events
│   ├── contracts.py       # typed API and serving policies
│   ├── model.py           # replaceable PyTorch model runner
│   ├── observability.py   # Prometheus instrumentation
│   └── preprocessing.py   # image validation and normalization
├── tests/
│   └── test_preprocessing.py
├── monitoring/
│   └── prometheus.yml
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

## Engineering roadmap

The repository is intentionally honest about what is and is not implemented. A production research deployment could add:

1. DVC-backed dataset versioning and reproducible training pipelines
2. MLflow experiment tracking and model registry
3. GPU inference workers and model warm-up
4. ONNX/TorchScript optimization and latency benchmarking
5. Calibration and threshold selection on a held-out validation set
6. Dataset/model drift monitoring
7. Structured OpenTelemetry traces
8. Kubernetes deployment with resource limits and autoscaling
9. Secure object storage for authorized datasets
10. Model evaluation gates before promotion

These are engineering extensions, not claims about the current reference service.

## Scope and limitations

This repository is an ML engineering demonstration. The included reference model has not been established as clinically safe or diagnostically accurate, and the API must not be used for medical diagnosis or patient-care decisions.

## Portfolio signal

This project complements the other portfolio systems by focusing on **computer vision serving and ML reliability** rather than another tabular prediction exercise. It shows the ability to move from a research-style model toward a service with contracts, validation, observability, deployment, and explicit uncertainty handling.
