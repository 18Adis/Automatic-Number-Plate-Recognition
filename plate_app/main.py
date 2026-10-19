"""FastAPI server for uploads, camera frames, model status and local history."""

from __future__ import annotations

import base64
import csv
import hashlib
from io import BytesIO, StringIO
import logging
from pathlib import Path
import threading
import time
from typing import Literal
import uuid

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field

from .config import PROJECT_ROOT, Settings
from .consensus import ConsensusTracker, is_reliable
from .history import HistoryStore
from .inference import InferenceEngine, normalize_text, probability
from .ocr import OCR_LABELS, OCR_METHODS, edit_distance

OCRMethod = Literal["cct", "easyocr", "tesseract", "paddleocr"]

LOGGER = logging.getLogger(__name__)


class LimitedBodyMiddleware:
    """Reject oversize requests before multipart parsing creates temporary files."""

    def __init__(self, app, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] not in {"POST", "PATCH", "PUT"}:
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers", []))
        try:
            declared = int(headers.get(b"content-length", b"0"))
        except ValueError:
            declared = self.max_bytes + 1
        if declared > self.max_bytes:
            return await JSONResponse({"detail": "Image upload must be 12 MB or smaller."}, status_code=413)(scope, receive, send)
        chunks = []
        size = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body = message.get("body", b"")
            size += len(body)
            if size > self.max_bytes:
                return await JSONResponse({"detail": "Image upload must be 12 MB or smaller."}, status_code=413)(scope, receive, send)
            chunks.append(body)
            if not message.get("more_body", False):
                break
        payload = b"".join(chunks)
        delivered = False

        async def bounded_receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": payload, "more_body": False}
            return await receive()

        await self.app(scope, bounded_receive, send)


class Correction(BaseModel):
    text: str = Field(min_length=1, max_length=32)


def decode_image(data: bytes, settings: Settings) -> np.ndarray:
    if not data:
        raise HTTPException(400, "Choose a JPEG, PNG or WebP image.")
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(413, "Image upload must be 12 MB or smaller.")
    try:
        with Image.open(BytesIO(data)) as probe:
            if probe.format not in {"JPEG", "PNG", "WEBP"}:
                raise HTTPException(415, "Supported image formats are JPEG, PNG and WebP.")
            width, height = probe.size
            if width * height > settings.max_pixels or max(width, height) > settings.max_dimension:
                raise HTTPException(413, "Image must be at most 20 megapixels and 10,000 pixels per side.")
            probe.verify()
        with Image.open(BytesIO(data)) as source:
            image = np.asarray(ImageOps.exif_transpose(source).convert("RGB"))
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise HTTPException(400, "This image could not be read. Try a different JPEG, PNG or WebP file.") from exc


def crop_data_url(image: np.ndarray, box: list[int]) -> str:
    x1, y1, x2, y2 = box
    crop = image[y1:y2, x1:x2]
    if crop.shape[1] > 640:
        crop = cv2.resize(crop, (640, max(1, round(crop.shape[0] * 640 / crop.shape[1]))))
    ok, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii") if ok else ""


def capture_jpeg(image: np.ndarray) -> bytes:
    height, width = image.shape[:2]
    scale = min(1.0, 1600 / max(width, height))
    if scale < 1:
        image = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))))
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return encoded.tobytes() if ok else b""


def create_app(predictor=None, data_dir: Path | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    data_dir = data_dir or settings.data_dir
    application = FastAPI(title="PlateSight", version="1.0.0", docs_url="/api/docs", redoc_url=None)
    application.add_middleware(LimitedBodyMiddleware, max_bytes=settings.max_upload_bytes + 64 * 1024)
    application.state.predictor = predictor or InferenceEngine(settings)
    application.state.history = HistoryStore(data_dir / "history.db")
    application.state.tracker = ConsensusTracker()
    application.state.processing = threading.BoundedSemaphore(2)

    @application.get("/api/status")
    def status():
        return application.state.predictor.status()

    @application.post("/api/warmup")
    def warmup(ocr_method: OCRMethod = Form("cct")):
        if not application.state.processing.acquire(blocking=False):
            raise HTTPException(429, "The model is busy. Wait for the current image, then retry.")
        try:
            return application.state.predictor.warmup(ocr_method=ocr_method)
        except Exception as exc:
            LOGGER.exception("Model preparation failed")
            raise HTTPException(503, str(exc)) from exc
        finally:
            application.state.processing.release()

    @application.post("/api/analyze")
    def analyze(
        file: UploadFile = File(...),
        source: Literal["upload", "snapshot", "live", "video"] = Form("upload"),
        session_id: str | None = Form(None, max_length=80),
        confidence: float = Form(0.35, ge=0.10, le=0.95),
        ocr_method: OCRMethod = Form("cct"),
    ):
        if source in {"live", "video"} and not (session_id and session_id.strip()):
            raise HTTPException(422, "Live and video frames need a session_id. Restart the camera and retry.")
        if not application.state.processing.acquire(blocking=False):
            raise HTTPException(429, "The model is busy. Wait for the current image, then retry.")
        try:
            image = decode_image(file.file.read(settings.max_upload_bytes + 1), settings)
            height, width = image.shape[:2]
            started = time.perf_counter()
            try:
                predictions = application.state.predictor.predict(image, confidence=confidence, ocr_method=ocr_method)
            except Exception as exc:
                LOGGER.exception("Inference failed")
                raise HTTPException(503, f"Could not analyze the image: {exc}") from exc
            inference_ms = round((time.perf_counter() - started) * 1000, 1)
            detections = []
            for prediction in predictions[:64]:
                detection_score = probability(prediction.detection_confidence)
                if detection_score < confidence:
                    continue
                x1, y1, x2, y2 = [int(coordinate) for coordinate in prediction.box]
                box = [max(0, min(width, x1)), max(0, min(height, y1)), max(0, min(width, x2)), max(0, min(height, y2))]
                if box[2] <= box[0] or box[3] <= box[1]:
                    continue
                raw_text = str(prediction.text or "")[:64]
                text = normalize_text(raw_text)
                recognition_score = probability(prediction.recognition_confidence)
                detections.append({
                    "id": uuid.uuid4().hex, "text": text, "raw_text": raw_text, "box": box,
                    "detection_confidence": round(detection_score, 4),
                    "recognition_confidence": round(recognition_score, 4),
                    "status": "confirmed" if is_reliable(text, detection_score, recognition_score) else "needs_review",
                    "crop_data_url": crop_data_url(image, box),
                    "ocr_method": ocr_method, "ocr_ms": prediction.ocr_ms,
                })
            if source in {"live", "video"}:
                application.state.tracker.update(f"{source}:{session_id}:{ocr_method}", detections)
            stored_frame = None
            for detection in detections:
                if source in {"upload", "snapshot"} or detection["status"] == "confirmed":
                    if stored_frame is None:
                        stored_frame = capture_jpeg(image)
                    crop_bytes = base64.b64decode(detection["crop_data_url"].partition(",")[2])
                    detection["id"] = application.state.history.record(
                        detection, source, session_id, image_jpeg=stored_frame, crop_jpeg=crop_bytes)
            return {
                "width": width, "height": height, "inference_ms": inference_ms,
                "detections": detections, "model_mode": application.state.predictor.status()["model_mode"],
                "ocr_method": ocr_method,
            }
        finally:
            file.file.close()
            application.state.processing.release()

    @application.post("/api/compare")
    def compare(file: UploadFile = File(...), confidence: float = Form(0.35, ge=0.10, le=0.95),
                methods: list[OCRMethod] = Form(list(OCR_METHODS)), expected_text: str = Form("", max_length=32)):
        methods = list(dict.fromkeys(methods))
        expected = normalize_text(expected_text)
        if len(methods) < 2:
            raise HTTPException(422, "Select at least two different OCR methods.")
        if expected_text.strip() and not 1 <= len(expected) <= 16:
            raise HTTPException(422, "Expected plate must contain 1–16 letters or digits.")
        if not application.state.processing.acquire(blocking=False):
            raise HTTPException(429, "The model is busy. Wait for the current image, then retry.")
        try:
            image = decode_image(file.file.read(settings.max_upload_bytes + 1), settings)
            try:
                result = application.state.predictor.compare(image, confidence, methods)
            except Exception as exc:
                LOGGER.exception("OCR comparison failed")
                raise HTTPException(503, f"Could not compare this image: {exc}") from exc
            if expected and len(result["detections"]) > 1:
                raise HTTPException(422, "Expected plate scoring needs one plate. Crop to one vehicle or leave the expected number empty.")
            for detection in result["detections"]:
                detection["crop_data_url"] = crop_data_url(image, detection["box"])
                for reading in detection["readings"]:
                    reading["label"] = "Custom character model" if reading["ocr_method"] == "cct" and settings.character_weights else OCR_LABELS[reading["ocr_method"]]
                    reading["exact_match"] = reading["text"] == expected if expected and not reading["error"] else None
                    reading["character_error_rate"] = edit_distance(expected, reading["text"]) / len(expected) if expected and not reading["error"] else None
            result.update(width=image.shape[1], height=image.shape[0], expected_text=expected or None,
                          normalization="Uppercase A–Z and 0–9; punctuation removed; no ambiguous-character substitution.")
            return result
        finally:
            file.file.close()
            application.state.processing.release()

    @application.get("/api/history")
    def history():
        return {"items": application.state.history.list()}

    @application.get("/api/history.csv")
    def history_csv():
        buffer = StringIO(newline="")
        columns = ["id", "text", "raw_text", "source", "timestamp", "ocr_method", "ocr_ms", "detection_confidence", "recognition_confidence", "status"]
        writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for record in application.state.history.list(limit=100000):
            # Raw OCR text is untrusted; neutralize formula-leading cells in spreadsheet exports.
            for key, value in record.items():
                if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
                    record[key] = "'" + value
            writer.writerow(record)
        return Response(buffer.getvalue(), media_type="text/csv", headers={"Content-Disposition": 'attachment; filename="plate-detections.csv"'})

    @application.delete("/api/history")
    def clear_history():
        application.state.history.clear()
        application.state.tracker.clear()
        return {"ok": True}

    @application.get("/api/history/{identity}/image")
    def history_image(identity: str):
        content = application.state.history.capture(identity)
        if content is None:
            raise HTTPException(404, "No image was saved for this detection.")
        return Response(content, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @application.get("/api/history/{identity}/thumbnail")
    def history_thumbnail(identity: str):
        content = application.state.history.capture(identity, thumbnail=True)
        if content is None:
            raise HTTPException(404, "No image was saved for this detection.")
        return Response(content, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @application.patch("/api/history/{identity}")
    def correct_history(identity: str, correction: Correction):
        text = normalize_text(correction.text)
        if not 1 <= len(text) <= 16:
            raise HTTPException(422, "Enter between 1 and 16 letters and digits.")
        item = application.state.history.correct(identity, text)
        if item is None:
            raise HTTPException(404, "This history item no longer exists.")
        return item

    def samples_by_id() -> dict[str, Path]:
        directory = data_dir / "samples"
        if not directory.is_dir():
            return {}
        return {hashlib.sha256(path.name.encode()).hexdigest()[:16]: path
                for path in sorted(directory.iterdir())
                if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
                and path.resolve().parent == directory.resolve()}

    @application.get("/api/samples")
    def samples():
        return {"items": [{"id": identity, "name": path.stem, "url": f"/api/sample/{identity}"}
                          for identity, path in samples_by_id().items()]}

    @application.get("/api/sample/{identity}")
    def sample(identity: str):
        path = samples_by_id().get(identity)
        if path is None:
            raise HTTPException(404, "Sample image not found.")
        return FileResponse(path)

    @application.get("/", include_in_schema=False)
    def index():
        return FileResponse(PROJECT_ROOT / "static" / "index.html")

    @application.get("/history", include_in_schema=False)
    def history_page():
        return FileResponse(PROJECT_ROOT / "static" / "history.html")

    application.mount("/static", StaticFiles(directory=PROJECT_ROOT / "static", check_dir=False), name="static")
    return application


app = create_app()
