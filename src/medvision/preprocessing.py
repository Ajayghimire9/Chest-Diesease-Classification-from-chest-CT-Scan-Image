from __future__ import annotations

import io

import numpy as np
from PIL import Image

from .contracts import ImagePolicy


def decode_and_normalize(payload: bytes, policy: ImagePolicy) -> np.ndarray:
    """Decode a 2-D image and return a normalized CHW float32 tensor."""
    if len(payload) > policy.max_bytes:
        raise ValueError("image exceeds configured size limit")
    try:
        image = Image.open(io.BytesIO(payload)).convert("L")
    except Exception as exc:
        raise ValueError("payload is not a supported image") from exc
    width, height = image.size
    if min(width, height) < policy.min_size or max(width, height) > policy.max_size:
        raise ValueError("image dimensions violate the serving policy")
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - 0.5) / 0.5
    return array[None, ...].astype(np.float32)
