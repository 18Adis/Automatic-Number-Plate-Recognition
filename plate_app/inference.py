"""Real inference, loaded only after warmup or an image request."""

from __future__ import annotations

import math
import os
import re
import threading
import time
import hashlib
from dataclasses import dataclass
from statistics import mean, median
from typing import Any

import numpy as np

from .config import PROJECT_ROOT, Settings
from .ocr import OCRReaders, OCR_METHODS


def normalize_text(text: str) -> str:
    """Remove OCR separators without guessing or replacing ambiguous characters."""
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def probability(value: Any) -> float:
    if isinstance(value, (list, tuple, np.ndarray)):
        value = mean(float(item) for item in value) if len(value) else 0.0
    result = float(value or 0.0)
    return min(1.0, max(0.0, result)) if math.isfinite(result) else 0.0


@dataclass
class Prediction:
    box: list[int]
    text: str
    detection_confidence: float
    recognition_confidence: float
    ocr_method: str = "cct"
    ocr_ms: float = 0.0


def ordered_characters(characters: list[dict]) -> tuple[str, float]:
    """Sort character boxes into rows; suppress duplicate overlapping predictions."""
    if not characters:
        return "", 0.0
    accepted = []
    for item in sorted(characters, key=lambda item: item["confidence"], reverse=True):
        x1, y1, x2, y2 = item["box"]
        if x2 <= x1 or y2 <= y1:
            continue
        duplicate = False
        for previous in accepted:
            a, b, c, d = previous["box"]
            overlap = max(0, min(x2, c) - max(x1, a)) * max(0, min(y2, d) - max(y1, b))
            if overlap / min((x2 - x1) * (y2 - y1), (c - a) * (d - b)) > 0.65:
                duplicate = True
                break
        if not duplicate:
            accepted.append(item)
    if not accepted:
        return "", 0.0
    typical_height = median(item["box"][3] - item["box"][1] for item in accepted)
    rows: list[list[dict]] = []
    for item in sorted(accepted, key=lambda item: (item["box"][1] + item["box"][3]) / 2):
        center = (item["box"][1] + item["box"][3]) / 2
        for row in rows:
            row_center = mean((member["box"][1] + member["box"][3]) / 2 for member in row)
            if abs(center - row_center) <= typical_height * 0.55:
                row.append(item)
                break
        else:
            rows.append([item])
    ordered = [item for row in rows for item in sorted(row, key=lambda item: item["box"][0])]
    return "".join(item["label"] for item in ordered), mean(item["confidence"] for item in ordered)


class InferenceEngine:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self._lock = threading.Lock()
        self._alpr = None
        self._custom_detector = None
        self._custom_characters = None
        self._loading = False
        self._error: str | None = None
        self._readers = OCRReaders(self.settings.model_dir, PROJECT_ROOT)

    def status(self) -> dict:
        ready = self._alpr is not None
        if ready:
            message = "Models are ready."
        elif self._loading:
            message = "Preparing models. The first run may download the pretrained weights."
        elif self._error:
            message = self._error
        else:
            message = "Models will load when you analyze an image or choose Prepare models."
        methods = self._readers.methods()
        methods[0]["ready"] = ready
        if self.settings.character_weights:
            methods[0]["label"] = "Custom character model"
        return {
            "ready": ready,
            "model_mode": self.settings.model_mode,
            "detector": self.settings.detector_weights.name if self.settings.detector_weights else self.settings.detector_name,
            "recognizer": self.settings.character_weights.name if self.settings.character_weights else self.settings.recognizer_name,
            "device": "cpu" if self.settings.model_mode == "pretrained" else self.settings.device,
            "message": message,
            "ocr_methods": methods,
        }

    def warmup(self, ocr_method: str = "cct") -> dict:
        if ocr_method not in OCR_METHODS:
            raise ValueError("Unknown OCR method")
        with self._lock:
            self._initialize()
            self._readers.prepare(ocr_method)
        return self.status()

    def _initialize(self) -> None:
        if self._alpr is not None:
            return
        self._loading = True
        self._error = None
        try:
            for path in (self.settings.detector_weights, self.settings.character_weights):
                if path is not None and not path.is_file():
                    raise FileNotFoundError(f"Configured model does not exist: {path}")
            self.settings.model_dir.mkdir(parents=True, exist_ok=True)
            # These libraries expose cache locations as module constants, not environment flags.
            # Set them before constructing either model so all downloaded files stay in this project.
            from fast_plate_ocr.inference import hub as ocr_hub
            from open_image_models.detection.core import hub as detector_hub

            ocr_hub.MODEL_CACHE_DIR = self.settings.model_dir / "fast-plate-ocr"
            detector_hub.MODEL_CACHE_DIR = self.settings.model_dir / "open-image-models"
            from fast_alpr import ALPR

            custom_detector = custom_characters = None
            if self.settings.detector_weights or self.settings.character_weights:
                os.environ.setdefault("YOLO_CONFIG_DIR", str(self.settings.model_dir / "ultralytics-config"))
                from ultralytics import YOLO

                if self.settings.detector_weights:
                    custom_detector = YOLO(str(self.settings.detector_weights))
                    labels = {str(label).lower() for label in custom_detector.names.values()}
                    if len(labels) != 1:
                        raise ValueError("The custom plate detector must have exactly one plate class.")
                if self.settings.character_weights:
                    custom_characters = YOLO(str(self.settings.character_weights))
                    if any(not re.fullmatch(r"[A-Z0-9]", str(label)) for label in custom_characters.names.values()):
                        raise ValueError("Character model labels must be individual A-Z or 0-9 characters.")

            # The baseline stays available for whichever half has not been replaced by custom weights.
            # Both ONNX models use CPU explicitly for predictable laptop compatibility.
            alpr = ALPR(
                detector_model=self.settings.detector_name,
                ocr_model=self.settings.recognizer_name,
                detector_conf_thresh=0.10,
                detector_providers=["CPUExecutionProvider"],
                ocr_device="cpu",
                ocr_providers=["CPUExecutionProvider"],
            )
            self._custom_detector = custom_detector
            self._custom_characters = custom_characters
            self._alpr = alpr
        except Exception as exc:
            self._error = f"Model preparation failed: {exc}. Check installation and internet access, then retry."
            raise RuntimeError(self._error) from exc
        finally:
            self._loading = False

    def _detect(self, image: np.ndarray, confidence: float) -> list[tuple[list[int], float]]:
        if self._custom_detector:
            results = self._custom_detector.predict(image, conf=confidence, device=self.settings.device, verbose=False)[0]
            boxes = [(list(map(int, box.xyxy[0].tolist())), float(box.conf[0])) for box in results.boxes]
        else:
            boxes = [([int(item.bounding_box.x1), int(item.bounding_box.y1), int(item.bounding_box.x2), int(item.bounding_box.y2)],
                      probability(item.confidence)) for item in self._alpr.detector.predict(image)
                     if item.confidence >= confidence]
        height, width = image.shape[:2]
        accepted = []
        for box, score in boxes[:64]:
            x1, y1, x2, y2 = box
            box = [max(0, min(width, x1)), max(0, min(height, y1)), max(0, min(width, x2)), max(0, min(height, y2))]
            x1, y1, x2, y2 = box
            if x2 <= x1 or y2 <= y1:
                continue
            accepted.append((box, probability(score)))
        return accepted

    def _read_crop(self, crop: np.ndarray, method: str) -> tuple[str, float]:
        if method != "cct":
            return self._readers.read(crop, method)
        if self._custom_characters:
            result = self._custom_characters.predict(crop, conf=0.25, device=self.settings.device, verbose=False)[0]
            characters = [{"box": item.xyxy[0].tolist(), "confidence": float(item.conf[0]),
                           "label": str(result.names[int(item.cls[0])])} for item in result.boxes]
            return ordered_characters(characters)
        ocr = self._alpr.ocr.predict(crop)
        return (ocr.text, probability(ocr.confidence)) if ocr else ("", 0.0)

    def predict(self, image: np.ndarray, confidence: float = 0.35, ocr_method: str = "cct") -> list[Prediction]:
        if ocr_method not in OCR_METHODS:
            raise ValueError("Unknown OCR method")
        with self._lock:
            self._initialize()
            self._readers.prepare(ocr_method)
            predictions = []
            for box, score in self._detect(image, confidence):
                x1, y1, x2, y2 = box
                started = time.perf_counter()
                text, ocr_score = self._read_crop(image[y1:y2, x1:x2].copy(), ocr_method)
                elapsed = round((time.perf_counter() - started) * 1000, 2)
                predictions.append(Prediction(box, text, score, probability(ocr_score), ocr_method, elapsed))
            return predictions

    def compare(self, image: np.ndarray, confidence: float, methods: list[str]) -> dict:
        if any(method not in OCR_METHODS for method in methods):
            raise ValueError("Unknown OCR method")
        with self._lock:
            base_ready = self._alpr is not None
            started = time.perf_counter()
            self._initialize()
            initialization_ms = round((time.perf_counter() - started) * 1000, 2) if not base_ready else 0.0
            started = time.perf_counter()
            boxes = self._detect(image, confidence)
            detection_ms = round((time.perf_counter() - started) * 1000, 2)
            preparations, errors = {}, {}
            if boxes:
                for method in methods:
                    started = time.perf_counter()
                    try:
                        self._readers.prepare(method)
                    except Exception as error:
                        errors[method] = str(error)
                    preparations[method] = round((time.perf_counter() - started) * 1000, 2)
            detections = []
            for box, score in boxes:
                x1, y1, x2, y2 = box
                crop = image[y1:y2, x1:x2]
                readings = []
                for method in methods:
                    reading = {"ocr_method": method, "text": "", "raw_text": "", "recognition_confidence": None,
                               "ocr_ms": None, "error": errors.get(method)}
                    if method not in errors:
                        try:
                            started = time.perf_counter()
                            text, ocr_score = self._read_crop(crop.copy(), method)
                            reading.update(text=normalize_text(text), raw_text=str(text)[:64],
                                           recognition_confidence=round(probability(ocr_score), 4),
                                           ocr_ms=round((time.perf_counter() - started) * 1000, 2))
                        except Exception as error:
                            reading["error"] = str(error)
                    readings.append(reading)
                detections.append({"box": box, "detection_confidence": score,
                                   "crop_sha256": hashlib.sha256(crop.tobytes()).hexdigest(), "readings": readings})
            return {"detections": detections, "detection_ms": detection_ms, "initialization_ms": initialization_ms,
                    "preparation_ms": preparations, "methods": methods}
