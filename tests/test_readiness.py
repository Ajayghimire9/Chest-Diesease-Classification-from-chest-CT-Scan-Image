from fastapi.testclient import TestClient

from medvision import api


def test_service_requires_weights(monkeypatch):
    monkeypatch.setattr(api, "runner", None)
    client = TestClient(api.app)
    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 503
    response = client.post("/v1/predict", files={"file": ("image.png", b"x", "image/png")})
    assert response.status_code == 503
