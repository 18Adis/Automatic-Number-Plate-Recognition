from types import SimpleNamespace

import numpy as np
import pytest

from plate_app.config import Settings
from plate_app.inference import InferenceEngine
from plate_app.ocr import OCR_METHODS, OCRReaders, edit_distance, ordered_words


class FakeDetector:
    def __init__(self):
        self.calls = 0

    def predict(self, image):
        self.calls += 1
        return [SimpleNamespace(bounding_box=SimpleNamespace(x1=2, y1=3, x2=12, y2=8), confidence=0.9)]


@pytest.fixture
def engine(tmp_path):
    instance = InferenceEngine(Settings(model_dir=tmp_path / "models"))
    instance._alpr = SimpleNamespace(detector=FakeDetector())
    return instance


def test_comparison_detects_once_and_passes_identical_isolated_crops(engine, monkeypatch):
    seen = []
    def read(crop, method):
        seen.append((method, crop.copy()))
        crop[:] = 0  # A third-party adapter must not modify another method's input.
        return "KL 07 AB 1234", 0.8
    monkeypatch.setattr(engine._readers, "prepare", lambda method: None)
    monkeypatch.setattr(engine, "_read_crop", read)
    image = np.full((12, 16, 3), 125, dtype=np.uint8)
    result = engine.compare(image, 0.35, list(OCR_METHODS))
    assert engine._alpr.detector.calls == 1
    assert [method for method, _ in seen] == list(OCR_METHODS)
    assert all(crop.shape == (5, 10, 3) and np.all(crop == 125) for _, crop in seen)
    assert np.all(image == 125)
    readings = result["detections"][0]["readings"]
    assert all(reading["text"] == "KL07AB1234" and reading["ocr_ms"] >= 0 for reading in readings)
    assert result["initialization_ms"] == 0


def test_unavailable_method_is_reported_without_fallback(engine, monkeypatch):
    def prepare(method):
        if method == "paddleocr":
            raise RuntimeError("Paddle model unavailable")
    monkeypatch.setattr(engine._readers, "prepare", prepare)
    monkeypatch.setattr(engine, "_read_crop", lambda crop, method: ("AB1234", 0.9))
    result = engine.compare(np.zeros((12, 16, 3), np.uint8), 0.35, ["cct", "paddleocr"])
    first, second = result["detections"][0]["readings"]
    assert first["text"] == "AB1234" and first["error"] is None
    assert second["error"] == "Paddle model unavailable"
    assert second["text"] == "" and second["ocr_ms"] is None


def test_predict_uses_only_selected_ocr(engine, monkeypatch):
    seen = []
    monkeypatch.setattr(engine._readers, "prepare", lambda method: seen.append(method))
    monkeypatch.setattr(engine, "_read_crop", lambda crop, method: (method, 0.75))
    prediction = engine.predict(np.zeros((12, 16, 3), np.uint8), ocr_method="paddleocr")[0]
    assert seen == ["paddleocr"] and prediction.ocr_method == "paddleocr"
    assert prediction.text == "paddleocr"


def test_easyocr_words_follow_plate_rows():
    words = [([[40, 20], [60, 20], [60, 30], [40, 30]], "1234", 0.8),
             ([[20, 0], [40, 0], [40, 10], [20, 10]], "07", 0.9),
             ([[0, 0], [10, 0], [10, 10], [0, 10]], "KL", 0.95),
             ([[0, 20], [20, 20], [20, 30], [0, 30]], "AB", 0.85)]
    text, confidence = ordered_words(words)
    assert text == "KL 07 AB 1234" and confidence == pytest.approx(0.875)


def test_tesseract_parses_tsv_and_times_out_safely(tmp_path, monkeypatch):
    readers = OCRReaders(tmp_path, tmp_path)
    readers.tesseract = "tesseract"
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout=b"level\tconf\ttext\n5\t90\tKL07\n5\t80\tAB1234\n", stderr=b"")
    monkeypatch.setattr("plate_app.ocr.subprocess.run", run)
    text, confidence = readers.read(np.full((30, 100, 3), 200, np.uint8), "tesseract")
    assert text == "KL07 AB1234" and confidence == pytest.approx(0.85)
    assert calls[0][1]["timeout"] == 20 and calls[0][1]["input"].startswith(b"\x89PNG")


def test_character_error_rate_uses_edit_distance():
    assert edit_distance("KL07AB1234", "KL07AB1235") == 1
    assert edit_distance("KL07AB1234", "") == 10
    assert edit_distance("AB12", "AAB12X") == 2
