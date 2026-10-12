from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from plate_app.config import Settings
from plate_app.inference import Prediction, normalize_text, ordered_characters
from plate_app.main import create_app


class MockPredictor:
    def __init__(self):
        self.ready = False
        self.calls = 0
        self.predictions = [Prediction([10, 10, 110, 50], "KL 07 AB 1234", 0.92, 0.88)]

    def status(self):
        return {"ready": self.ready, "model_mode": "test", "detector": "test", "recognizer": "test", "device": "cpu", "message": "Test predictor"}

    def warmup(self):
        self.ready = True
        return self.status()

    def predict(self, image, confidence=0.35):
        self.calls += 1
        self.ready = True
        return self.predictions


def image_bytes(size=(160, 100), format="PNG"):
    buffer = BytesIO()
    Image.new("RGB", size, (125, 130, 135)).save(buffer, format=format)
    return buffer.getvalue()


@pytest.fixture
def client(tmp_path):
    app = create_app(MockPredictor(), data_dir=tmp_path)
    with TestClient(app) as client:
        yield client


def analyze(client, **fields):
    return client.post("/api/analyze", files={"file": ("vehicle.png", image_bytes(), "image/png")}, data=fields)


def test_status_does_not_load_models(client):
    assert client.get("/api/status").json()["ready"] is False
    assert client.app.state.predictor.calls == 0
    assert client.post("/api/warmup").json()["ready"] is True


def test_upload_produces_real_contract_and_history(client):
    response = analyze(client)
    assert response.status_code == 200
    payload = response.json()
    assert (payload["width"], payload["height"]) == (160, 100)
    detection = payload["detections"][0]
    assert detection["text"] == "KL07AB1234"
    assert detection["raw_text"] == "KL 07 AB 1234"
    assert detection["status"] == "confirmed"
    assert detection["box"] == [10, 10, 110, 50]
    assert detection["crop_data_url"].startswith("data:image/jpeg;base64,")
    items = client.get("/api/history").json()["items"]
    assert len(items) == 1 and items[0]["id"] == detection["id"]
    assert items[0]["source"] == "upload"
    capture = client.get(items[0]["image_url"])
    thumbnail = client.get(items[0]["thumbnail_url"])
    assert capture.headers["content-type"] == thumbnail.headers["content-type"] == "image/jpeg"
    assert capture.headers["cache-control"] == "no-store"
    assert Image.open(BytesIO(capture.content)).size == (160, 100)
    assert Image.open(BytesIO(thumbnail.content)).size == (100, 40)


def test_live_requires_agreement_and_deduplicates(client):
    for index in range(5):
        response = analyze(client, source="live", session_id="camera-one")
        assert response.status_code == 200
        status = response.json()["detections"][0]["status"]
        assert status == ("reading" if index < 2 else "confirmed")
    assert len(client.get("/api/history").json()["items"]) == 1
    response = analyze(client, source="live", session_id="camera-two")
    assert response.json()["detections"][0]["status"] == "reading"


def test_history_correction_csv_and_clear(client):
    identity = analyze(client).json()["detections"][0]["id"]
    response = client.patch(f"/api/history/{identity}", json={"text": "kl-07 ab 1235"})
    assert response.json()["text"] == "KL07AB1235"
    assert response.json()["raw_text"] == "KL 07 AB 1234"
    exported = client.get("/api/history.csv")
    assert exported.status_code == 200 and "KL07AB1235" in exported.text
    assert "attachment" in exported.headers["content-disposition"]
    assert "image_url" not in exported.text
    assert client.get(f"/api/history/{identity}/thumbnail").status_code == 200
    assert client.delete("/api/history").json()["ok"]
    assert client.get("/api/history").json() == {"items": []}
    assert client.patch(f"/api/history/{identity}", json={"text": "KL071234"}).status_code == 404
    assert client.get(f"/api/history/{identity}/thumbnail").status_code == 404
    assert client.get(f"/api/history/{identity}/image").status_code == 404
    with client.app.state.history._connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM captures").fetchone()[0] == 0


def test_history_has_its_own_page(client):
    recognition = client.get("/")
    history = client.get("/history")
    assert recognition.status_code == history.status_code == 200
    assert 'id="historyBody"' not in recognition.text
    assert 'class="example-section"' not in recognition.text
    assert 'href="/history" target="_blank"' in recognition.text
    assert 'id="historyBody"' in history.text and 'id="imageDialog"' in history.text
    assert client.get("/static/history.js").status_code == 200


def test_large_history_capture_is_resized_without_stretching(client):
    response = client.post("/api/analyze", files={"file": ("large.png", image_bytes((2400, 1200)), "image/png")})
    assert response.status_code == 200
    item = client.get("/api/history").json()["items"][0]
    capture = client.get(item["image_url"])
    assert Image.open(BytesIO(capture.content)).size == (1600, 800)


def test_no_plate_is_valid_and_not_logged(client):
    client.app.state.predictor.predictions = []
    assert analyze(client).json()["detections"] == []
    assert client.get("/api/history").json()["items"] == []


def test_bounds_low_confidence_and_threshold(client):
    client.app.state.predictor.predictions = [Prediction([-5, -5, 200, 200], "AB01234", 0.9, 0.20), Prediction([20, 20, 5, 5], "BAD1", 0.9, 0.9)]
    result = analyze(client).json()["detections"]
    assert len(result) == 1
    assert result[0]["box"] == [0, 0, 160, 100]
    assert result[0]["status"] == "needs_review"
    assert analyze(client, confidence="0.95").json()["detections"] == []


def test_invalid_file_and_parameter_failures(client):
    response = client.post("/api/analyze", files={"file": ("fake.jpg", b"not an image", "image/jpeg")})
    assert response.status_code == 400
    assert analyze(client, source="live").status_code == 422
    assert analyze(client, source="unknown").status_code == 422
    assert analyze(client, confidence="2").status_code == 422
    assert client.post("/api/analyze", files={"file": ("image.gif", image_bytes(format="GIF"), "image/gif")}).status_code == 415


def test_oversize_body_and_pixels_rejected_before_inference(tmp_path):
    predictor = MockPredictor()
    app = create_app(predictor, data_dir=tmp_path, settings=Settings(max_upload_bytes=500, max_pixels=5000))
    with TestClient(app) as client:
        assert analyze(client).status_code == 413
        assert client.post("/api/analyze", content=b"x" * 67000).status_code == 413
        assert predictor.calls == 0


def test_missing_samples_and_path_traversal(client):
    assert client.get("/api/samples").json() == {"items": []}
    assert client.get("/api/sample/..%2Fhistory.db").status_code == 404
    assert client.get("/api/sample/unknown").status_code == 404


def test_samples_only_return_allowed_project_images(client, tmp_path):
    directory = tmp_path / "samples"
    directory.mkdir()
    (directory / "car.png").write_bytes(image_bytes())
    (directory / "metadata.json").write_text("{}")
    samples = client.get("/api/samples").json()["items"]
    assert len(samples) == 1
    response = client.get(samples[0]["url"])
    assert response.status_code == 200 and response.headers["content-type"] == "image/png"


def test_normalization_does_not_guess_ambiguous_characters():
    assert normalize_text("kl 0O-1I b8") == "KL0O1IB8"


def test_two_line_character_order_and_duplicate_suppression():
    chars = [
        {"box": [20, 20, 30, 35], "label": "4", "confidence": 0.9},
        {"box": [20, 0, 30, 15], "label": "B", "confidence": 0.9},
        {"box": [0, 20, 10, 35], "label": "3", "confidence": 0.9},
        {"box": [0, 0, 10, 15], "label": "A", "confidence": 0.9},
        {"box": [1, 0, 11, 15], "label": "4", "confidence": 0.5},
    ]
    text, confidence = ordered_characters(chars)
    assert text == "AB34" and confidence == pytest.approx(0.9)
