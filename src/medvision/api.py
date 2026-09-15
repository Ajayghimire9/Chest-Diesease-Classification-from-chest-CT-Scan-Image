from __future__ import annotations

import os
import time
import uuid

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from prometheus_client import make_asgi_app

from .audit import append_event
from .contracts import ImagePolicy, PredictionResponse
from .model import ModelRunner
from .observability import observe_request
from .preprocessing import decode_and_normalize

app = FastAPI(title="MedVision Inference Gateway", version="3.0.0")
app.mount("/metrics", make_asgi_app())

POLICY = ImagePolicy()
MODEL_VERSION = os.getenv("MODEL_VERSION", "reference-cnn-v1")
WEIGHTS = os.getenv("MODEL_WEIGHTS")
runner = ModelRunner(WEIGHTS) if WEIGHTS else None
LABELS = ("class_0", "class_1")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.post("/v1/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:  # noqa: B008 - FastAPI request marker
    request_id = str(uuid.uuid4())
    started = time.perf_counter()
    with observe_request():
        if runner is None:
            raise HTTPException(status_code=503, detail="Configure MODEL_WEIGHTS before inference")
        payload = await file.read(POLICY.max_bytes + 1)
        try:
            array = decode_and_normalize(payload, POLICY)
            index, probability = runner.predict(torch.from_numpy(array))
        except ValueError as exc:
            append_event(
                "artifacts/audit.jsonl",
                request_id=request_id,
                model_version=MODEL_VERSION,
                status="rejected",
            )
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        status = "accepted" if probability >= POLICY.review_threshold else "review"
        append_event(
            "artifacts/audit.jsonl",
            request_id=request_id,
            model_version=MODEL_VERSION,
            status=status,
        )
        return PredictionResponse(
            request_id=request_id,
            model_version=MODEL_VERSION,
            predicted_class=LABELS[index] if status == "accepted" else None,
            probability=probability,
            status=status,
            latency_ms=(time.perf_counter() - started) * 1000,
        )


@app.get("/ready")
def ready():
    if runner is None:
        raise HTTPException(status_code=503, detail="Model weights unavailable")
    return {"ready": True, "model_version": MODEL_VERSION}
