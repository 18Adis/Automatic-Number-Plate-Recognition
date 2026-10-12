"""Generate example overlays from actual model output, never annotation hints."""
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
from plate_app.inference import InferenceEngine, normalize_text


def main():
    engine = InferenceEngine()
    engine.warmup()
    output = ROOT / "static" / "examples"
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for path in sorted((ROOT / "data" / "samples").glob("*.jpg")):
        image = cv2.imread(str(path))
        started = time.perf_counter()
        predictions = engine.predict(image)
        elapsed = round((time.perf_counter() - started) * 1000, 1)
        items = []
        for prediction in predictions:
            x1, y1, x2, y2 = prediction.box
            cv2.rectangle(image, (x1, y1), (x2, y2), (50, 160, 80), 3)
            cv2.rectangle(image, (x1, max(0, y1-23)), (min(image.shape[1], x1+80), y1), (50, 110, 55), -1)
            cv2.putText(image, "PLATE", (x1+7, max(14, y1-7)), cv2.FONT_HERSHEY_SIMPLEX, .45, (255,255,255), 1, cv2.LINE_AA)
            items.append({"text": normalize_text(prediction.text), "box": prediction.box,
                          "detection_confidence": prediction.detection_confidence,
                          "recognition_confidence": prediction.recognition_confidence})
        cv2.imwrite(str(output / path.name), image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        records.append({"image": path.name, "inference_ms": elapsed, "detections": items})
    report = ROOT / "data" / "verification.json"
    report.write_text(json.dumps({"model_mode":"pretrained", "notice":"Smoke examples, not an accuracy benchmark. Overlays are actual model predictions.", "images":records}, indent=2), encoding="utf-8")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
