"""Independent OCR adapters. Every method receives the same unmodified plate crop."""
from __future__ import annotations

import csv
from io import StringIO
import importlib.util
import os
from pathlib import Path
import shutil
from statistics import mean, median
import subprocess

import cv2
import numpy as np

OCR_METHODS = ("cct", "easyocr", "tesseract", "paddleocr")
OCR_LABELS = {"cct": "Plate OCR · CCT", "easyocr": "EasyOCR", "tesseract": "Tesseract", "paddleocr": "PaddleOCR"}
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def tesseract_executable(project_root: Path) -> str | None:
    explicit = os.getenv("PLATE_TESSERACT_CMD")
    if explicit:
        return str(Path(explicit)) if Path(explicit).is_file() else shutil.which(explicit)
    candidates = [project_root / "tools" / "tesseract" / "tesseract.exe",
                  Path(os.getenv("ProgramFiles", "C:/Program Files")) / "Tesseract-OCR" / "tesseract.exe",
                  Path(os.getenv("LOCALAPPDATA", str(Path.home()))) / "Programs" / "Tesseract-OCR" / "tesseract.exe"]
    return next((str(path) for path in candidates if path.is_file()), None) or shutil.which("tesseract")


def ordered_words(results) -> tuple[str, float]:
    """Read EasyOCR words top-to-bottom and left-to-right, including two-line plates."""
    words = []
    for points, text, score in results:
        xs, ys = zip(*points)
        words.append({"x": min(xs), "y": (min(ys) + max(ys)) / 2,
                      "height": max(ys) - min(ys), "text": str(text), "score": float(score)})
    if not words:
        return "", 0.0
    typical = max(1.0, median(word["height"] for word in words))
    rows = []
    for word in sorted(words, key=lambda value: value["y"]):
        for row in rows:
            if abs(word["y"] - mean(member["y"] for member in row)) <= typical * 0.55:
                row.append(word)
                break
        else:
            rows.append([word])
    ordered = [word for row in rows for word in sorted(row, key=lambda value: value["x"])]
    return " ".join(word["text"] for word in ordered), mean(word["score"] for word in ordered)


def edit_distance(expected: str, actual: str) -> int:
    row = list(range(len(actual) + 1))
    for index, left in enumerate(expected, 1):
        new = [index]
        for column, right in enumerate(actual, 1):
            new.append(min(new[-1] + 1, row[column] + 1, row[column - 1] + (left != right)))
        row = new
    return row[-1]


class OCRReaders:
    def __init__(self, model_dir: Path, project_root: Path):
        self.model_dir = model_dir
        self.project_root = project_root
        self.easyocr = None
        self.tesseract = None
        self.paddleocr = None

    def methods(self) -> list[dict]:
        easy = importlib.util.find_spec("easyocr") is not None
        tess = tesseract_executable(self.project_root) is not None
        paddle = importlib.util.find_spec("paddleocr") is not None and importlib.util.find_spec("paddle") is not None
        return [
            {"id": "cct", "label": OCR_LABELS["cct"], "available": True, "ready": False, "message": "Plate-specific transformer recognizer"},
            {"id": "easyocr", "label": OCR_LABELS["easyocr"], "available": easy, "ready": self.easyocr is not None,
             "message": "English scene-text OCR" if easy else "Run Setup OCR.ps1 to install EasyOCR"},
            {"id": "tesseract", "label": OCR_LABELS["tesseract"], "available": tess, "ready": self.tesseract is not None,
             "message": "English LSTM OCR" if tess else "Install Tesseract or set PLATE_TESSERACT_CMD"},
            {"id": "paddleocr", "label": OCR_LABELS["paddleocr"], "available": paddle, "ready": self.paddleocr is not None,
             "message": "PP-OCRv5 English recognition" if paddle else "Run Setup OCR.ps1 to install PaddleOCR"},
        ]

    def prepare(self, method: str) -> None:
        if method == "easyocr" and self.easyocr is None:
            if importlib.util.find_spec("easyocr") is None:
                raise RuntimeError("EasyOCR is not installed. Run Setup OCR.ps1, then restart the app.")
            import torch
            import easyocr
            torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
            directory = self.model_dir / "easyocr"
            directory.mkdir(parents=True, exist_ok=True)
            self.easyocr = easyocr.Reader(["en"], gpu=False, detector=False,
                                         model_storage_directory=str(directory), verbose=False)
        elif method == "tesseract" and self.tesseract is None:
            executable = tesseract_executable(self.project_root)
            if not executable:
                raise RuntimeError("Tesseract is not installed. Run Setup OCR.ps1 or set PLATE_TESSERACT_CMD.")
            check = subprocess.run([executable, "--list-langs"], capture_output=True, timeout=15,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            if check.returncode or "eng" not in check.stdout.decode(errors="replace").split():
                raise RuntimeError("Tesseract needs its English language data (eng.traineddata).")
            self.tesseract = executable
        elif method == "paddleocr" and self.paddleocr is None:
            if importlib.util.find_spec("paddleocr") is None or importlib.util.find_spec("paddle") is None:
                raise RuntimeError("PaddleOCR is not installed. Run Setup OCR.ps1, then restart the app.")
            os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(self.model_dir / "paddlex"))
            os.environ.setdefault("HF_HOME", str(self.model_dir / "huggingface"))
            from paddleocr import TextRecognition
            self.paddleocr = TextRecognition(model_name="en_PP-OCRv5_mobile_rec", device="cpu", enable_mkldnn=False)

    def read(self, crop: np.ndarray, method: str) -> tuple[str, float]:
        if method == "easyocr":
            return ordered_words(self.easyocr.recognize(crop, detail=1, paragraph=False,
                                                        allowlist=ALPHABET, decoder="greedy", workers=0))
        if method == "paddleocr":
            results = list(self.paddleocr.predict(input=crop, batch_size=1))
            if not results:
                return "", 0.0
            return str(results[0]["rec_text"]), float(results[0]["rec_score"])
        if method != "tesseract":
            raise ValueError(f"Unknown OCR adapter: {method}")
        # These documented, fixed preprocessing steps are part of the Tesseract method's timing.
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        scale = max(1.0, min(3.0, 80 / gray.shape[0]))
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.mean(binary) < 127:
            binary = 255 - binary
        binary = cv2.copyMakeBorder(binary, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
        ok, encoded = cv2.imencode(".png", binary)
        if not ok:
            raise RuntimeError("Could not prepare the plate for Tesseract.")
        result = subprocess.run([self.tesseract, "stdin", "stdout", "-l", "eng", "--psm", "6",
                                 "-c", f"tessedit_char_whitelist={ALPHABET}", "tsv"], input=encoded.tobytes(),
                                capture_output=True, timeout=20, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if result.returncode:
            raise RuntimeError("Tesseract could not read this crop: " + result.stderr.decode(errors="replace")[:250])
        words = [row for row in csv.DictReader(StringIO(result.stdout.decode("utf-8", errors="replace")), delimiter="\t")
                 if row.get("text", "").strip() and float(row.get("conf", -1)) >= 0]
        return " ".join(row["text"] for row in words), mean(float(row["conf"]) / 100 for row in words) if words else 0.0
