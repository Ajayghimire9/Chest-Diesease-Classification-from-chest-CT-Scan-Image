from __future__ import annotations

import time
from contextlib import contextmanager

try:
    from prometheus_client import Counter, Histogram
except ImportError:  # keeps preprocessing/model modules usable without Prometheus
    Counter = Histogram = None

REQUESTS = Counter("medvision_inference_requests_total", "Inference requests") if Counter else None
ERRORS = Counter("medvision_inference_errors_total", "Inference errors") if Counter else None
LATENCY = (
    Histogram("medvision_inference_latency_seconds", "Inference latency") if Histogram else None
)


@contextmanager
def observe_request():
    started = time.perf_counter()
    if REQUESTS:
        REQUESTS.inc()
    try:
        yield
    except Exception:
        if ERRORS:
            ERRORS.inc()
        raise
    finally:
        if LATENCY:
            LATENCY.observe(time.perf_counter() - started)
