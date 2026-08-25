from fastapi.testclient import TestClient

from api.main import app


def test_app_starts_and_connects_to_db():
    assert TestClient(app).get("/health").json() == {"status": "ok"}
