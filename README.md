# MedVision

Research image-classification serving.

MedVision packages an image classifier behind a typed HTTP interface. The engineering work focuses on input validation, model readiness, confidence handling and traceable responses.

## Run locally

Use Python 3.11 or newer in a virtual environment.

```bash
pip install -e ".[dev]"
# Set MODEL_WEIGHTS to a compatible CompactCNN state dictionary.
uvicorn medvision.api:app --host 127.0.0.1 --port 8000
```

## Design decisions

The process can start without weights, but readiness and inference return 503 until a configured model is loaded. Randomly initialized weights are never used as accepted predictions.

Upload reads are bounded before decoding. Image size and format policies are shared by the service and tests.

Low-confidence predictions route to review. Audit records keep request metadata without storing the uploaded image; Prometheus tracks service behavior.

## Technology

Python, PyTorch, FastAPI, Pydantic, Pillow, Prometheus, Docker.

## Validation

Run `python -m pytest tests -q` from the repository root. CI runs the maintained test suite and lint checks. Tests use local fixtures or mocks and do not deploy cloud resources.

## Scope and limitations

This is a research serving example with generic class labels. No diagnostic accuracy, clinical validation or patient-care suitability has been established. A compatible model artifact and its label semantics must be supplied by the operator. Training and model promotion are separate work.
