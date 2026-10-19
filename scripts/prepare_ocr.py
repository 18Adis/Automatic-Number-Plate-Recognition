"""Prepare selected OCR engines and verify each with a real image when available."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from plate_app.inference import InferenceEngine
from plate_app.ocr import OCR_METHODS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", choices=OCR_METHODS, default=list(OCR_METHODS))
    args = parser.parse_args()
    engine = InferenceEngine()
    results = []
    for method in args.methods:
        print(f"Preparing {method}...", flush=True)
        try:
            engine.warmup(method)
            results.append({"method": method, "ready": True})
        except Exception as error:
            results.append({"method": method, "ready": False, "error": str(error)})
        print(json.dumps(results[-1]), flush=True)
    raise SystemExit(0 if all(item["ready"] for item in results) else 1)


if __name__ == "__main__":
    main()
