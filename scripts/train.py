"""Fine-tune a YOLO detector on prepared plate or cropped-character data."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("plate", "characters"), required=True)
    parser.add_argument("--data", type=Path, help="Prepared YOLO data.yaml")
    parser.add_argument("--weights", default="yolo11n.pt", help="Local weights or Ultralytics pretrained model name (may download)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu", help="cpu or CUDA device index, e.g. 0")
    parser.add_argument("--workers", type=int, default=0, help="0 is the most portable Windows default")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", type=Path, default=Path("runs/train"))
    parser.add_argument("--name", help="Run name; Ultralytics increments existing names")
    parser.add_argument("--export-to", type=Path, help="Copy best weights here after training; refuses to overwrite")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch < 1 or args.imgsz < 32 or args.imgsz % 32:
        parser.error("epochs/batch must be positive and imgsz must be a positive multiple of 32")
    data = args.data or Path("data/prepared") / ("plates" if args.task == "plate" else "characters") / "data.yaml"
    if not data.is_file():
        parser.error(f"Dataset YAML not found: {data}. Run scripts/prepare_dataset.py first.")
    if args.export_to and args.export_to.exists():
        parser.error(f"Export destination exists: {args.export_to}; choose a new path")
    try:
        from ultralytics import YOLO
    except ImportError:
        parser.error("Training needs ultralytics. Install requirements-training.txt first.")
    print("Training uses approximate plate labels and existing source augmentations. Review the dataset before interpreting accuracy.")
    model = YOLO(args.weights)
    model.train(
        data=str(data.resolve()), epochs=args.epochs, batch=args.batch,
        imgsz=args.imgsz, device=args.device, workers=args.workers,
        seed=args.seed, deterministic=True, project=str(args.project),
        name=args.name or args.task, exist_ok=False, patience=15,
        fliplr=0.0, flipud=0.0, degrees=5.0, shear=0.0,
        perspective=0.0, mosaic=0.0, mixup=0.0, copy_paste=0.0,
        translate=0.05, scale=0.15, hsv_h=0.0, hsv_s=0.0, hsv_v=0.15,
    )
    best = Path(model.trainer.best)
    print(f"Best weights: {best.resolve()}")
    if args.export_to:
        args.export_to.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation protects a destination created during training.
        with best.open("rb") as source, args.export_to.open("xb") as destination:
            shutil.copyfileobj(source, destination)
        print(f"Exported: {args.export_to.resolve()}")


if __name__ == "__main__":
    main()
