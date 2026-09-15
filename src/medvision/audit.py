from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path


def append_event(path: str | Path, *, request_id: str, model_version: str, status: str) -> None:
    """Write minimal operational audit metadata; never persist image bytes."""
    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "request_id": request_id,
        "model_version": model_version,
        "status": status,
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
