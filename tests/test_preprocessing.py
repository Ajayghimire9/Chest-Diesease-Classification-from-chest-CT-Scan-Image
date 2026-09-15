import numpy as np
import pytest
from PIL import Image
from io import BytesIO

from medvision.contracts import ImagePolicy
from medvision.preprocessing import decode_and_normalize


def make_image(size=(64, 64)) -> bytes:
    buf = BytesIO()
    Image.fromarray(np.zeros(size, dtype=np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def test_preprocessing_returns_normalized_chw_tensor():
    result = decode_and_normalize(make_image(), ImagePolicy())
    assert result.shape == (1, 64, 64)
    assert result.dtype == np.float32


def test_size_limit_is_enforced():
    with pytest.raises(ValueError, match="size limit"):
        decode_and_normalize(make_image(), ImagePolicy(max_bytes=10))
