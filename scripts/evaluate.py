"""Evaluate detector boxes or exact plate text from manually transcribed examples."""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def normalize_text(value):
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def edit_distance(left, right):
    row = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        new = [i]
        for j, b in enumerate(right, 1):
            new.append(min(new[-1] + 1, row[j] + 1, row[j - 1] + (a != b)))
        row = new
    return row[-1]


def read_text_csv(path):
    results = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"image", "text"}.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} requires image,text columns")
        for row in reader:
            if not row["image"] or row["image"] in results:
                raise ValueError(f"Empty/duplicate image identifier in {path}: {row['image']}")
            results[row["image"]] = normalize_text(row["text"])
    return results


def text_metrics(truth, predictions):
    if not truth:
        raise ValueError("Ground truth is empty")
    total_chars = sum(len(v) for v in truth.values())
    errors = sum(edit_distance(expected, predictions.get(key, "")) for key, expected in truth.items())
    return {
        "examples": len(truth),
        "exact_registration_accuracy": sum(expected == predictions.get(key, "") for key, expected in truth.items()) / len(truth),
        "character_error_rate": errors / total_chars if total_chars else None,
        "character_edits": errors,
        "missing_predictions": sum(key not in predictions for key in truth),
        "unmatched_prediction_examples": sum(key not in truth for key in predictions),
        "normalization": "Uppercase ASCII letters/digits; whitespace and punctuation removed. No O/0 or I/1 guessing.",
        "scope": "One known target plate per image; for multiple vehicles use separately matched plate identifiers.",
        "mismatches": [{"image": key, "expected": expected, "predicted": predictions.get(key, "")}
                       for key, expected in truth.items() if expected != predictions.get(key, "")],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    detector = sub.add_parser("detector", help="YOLO box precision/recall and mAP")
    detector.add_argument("--weights", type=Path, required=True)
    detector.add_argument("--data", type=Path, required=True)
    detector.add_argument("--split", choices=("val", "test"), default="test")
    detector.add_argument("--device", default="cpu")
    detector.add_argument("--imgsz", type=int, default=640)
    detector.add_argument("--batch", type=int, default=4)
    detector.add_argument("--output", type=Path, default=Path("runs/evaluation/detector.json"))
    text = sub.add_parser("text", help="Exact registration accuracy using manually reviewed truth")
    text.add_argument("--truth", type=Path, required=True, help="CSV with image,text columns")
    text.add_argument("--predictions", type=Path, required=True, help="CSV with image,text columns; missing rows count as failures")
    text.add_argument("--output", type=Path, default=Path("runs/evaluation/text.json"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new --output path")
    if args.mode == "text":
        report = text_metrics(read_text_csv(args.truth), read_text_csv(args.predictions))
    else:
        if not args.weights.is_file() or not args.data.is_file():
            parser.error("Both --weights and --data must be existing local files")
        try:
            from ultralytics import YOLO
        except ImportError:
            parser.error("Install requirements-training.txt to evaluate YOLO models")
        metrics = YOLO(str(args.weights)).val(data=str(args.data.resolve()), split=args.split,
                                              device=args.device, imgsz=args.imgsz, batch=args.batch,
                                              workers=0, project="runs/evaluation", name="detector", exist_ok=False)
        report = {"weights": str(args.weights.resolve()), "split": args.split,
                  "precision": float(metrics.box.mp), "recall": float(metrics.box.mr),
                  "map50": float(metrics.box.map50), "map50_95": float(metrics.box.map),
                  "speed_ms_per_image": metrics.speed,
                  "limitation": "Detector box metrics do not measure exact registration accuracy. Pseudo plate labels are approximate."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "mismatches"}, indent=2))
    print(f"Evaluation saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
