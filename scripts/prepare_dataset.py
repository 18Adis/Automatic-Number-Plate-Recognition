"""Create isolated YOLO datasets for plate detection and cropped character detection."""
from __future__ import annotations

import argparse
import csv
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image

try:
    from .dataset_utils import *
except ImportError:
    from dataset_utils import *


def label_line(class_id, coords):
    return f"{class_id} " + " ".join(f"{v:.8f}" for v in coords) + "\n"


def write_yaml(folder, names):
    # JSON strings are valid YAML scalars and safely preserve Windows paths.
    lines = [f"path: {json.dumps(str(folder.resolve()))}", "train: images/train", "val: images/val", "test: images/test", "names:"]
    lines.extend(f"  {i}: {json.dumps(name)}" for i, name in enumerate(names))
    (folder / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare(zip_path, destination, seed=42, include_flagged=False, max_images=None):
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Output is not empty; use a new --output path: {destination}")
    records, issues, metadata = load_records(zip_path)
    assignments = grouped_split(records, seed)
    destination.mkdir(parents=True, exist_ok=True)
    plate_dir, char_dir = destination / "plates", destination / "characters"
    for folder in (plate_dir, char_dir):
        for split in ("train", "val", "test"):
            (folder / "images" / split).mkdir(parents=True)
            (folder / "labels" / split).mkdir(parents=True)
    write_yaml(plate_dir, ["number_plate"])
    write_yaml(char_dir, CHARACTERS)
    manifest, review, skipped, counts = [], [], Counter(), Counter()
    # Sort so outputs stay deterministic, independent of ZIP member ordering.
    ordered = sorted(records, key=lambda r: r["member"])
    with zipfile.ZipFile(zip_path) as archive:
        for record in ordered:
            split, group = assignments[record["member"]]
            flags = record["review_flags"]
            if flags:
                review.append({"member": record["member"], "split": split, "flags": ";".join(flags)})
            if not record["characters"]:
                skipped["no_valid_characters"] += 1
                continue
            if flags and not include_flagged:
                skipped["geometry_review_required"] += 1
                continue
            if max_images is not None and len(manifest) >= max_images:
                skipped["max_images_limit"] += 1
                continue
            name = unique_stem(record)
            content = bounded_read(archive, record["member"], MAX_IMAGE_BYTES)
            crop = plate_crop([c["box"] for c in record["characters"]], record["width"], record["height"])
            crop_width, crop_height = crop[2] - crop[0], crop[3] - crop[1]
            with Image.open(io.BytesIO(content)) as image:
                image = image.convert("RGB")
                image.save(plate_dir / "images" / split / f"{name}.jpg", quality=95)
                image.crop(crop).save(char_dir / "images" / split / f"{name}.jpg", quality=95)
            plate_label = label_line(0, yolo_bbox(crop, record["width"], record["height"]))
            char_labels = "".join(label_line(c["class_id"], yolo_bbox(c["box"], crop_width, crop_height, crop[:2]))
                                  for c in record["characters"])
            (plate_dir / "labels" / split / f"{name}.txt").write_text(plate_label, encoding="utf-8")
            (char_dir / "labels" / split / f"{name}.txt").write_text(char_labels, encoding="utf-8")
            counts[split] += 1
            manifest.append({"file": f"{name}.jpg", "source_member": record["member"], "sha256": record["sha256"],
                             "source_group": record["source_group"], "leakage_group": group,
                             "split": split, "crop_xyxy": crop, "character_count": len(record["characters"]),
                             "review_flags": flags, "plate_label_status": "approximate_unreviewed"})
    with (destination / "manual_review.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["member", "split", "flags"])
        writer.writeheader()
        writer.writerows(review)
    summary = {"source_zip": str(zip_path.resolve()), "seed": seed, "source_url": SOURCE_URL, "license": "CC BY 4.0",
               "output_images": dict(counts), "skipped": dict(skipped), "include_flagged": include_flagged,
               "geometry_flagged_images": len(review), "integrity_issue_counts": dict(Counter(i["issue"] for i in issues)),
               "single_plate_assumption": "Each retained image is assumed to contain one annotated plate; this requires manual verification.",
               "plate_labels": "Padded character enclosure: approximate unreviewed pseudo labels; not verified whole-plate ground truth.",
               "existing_augmentations": "Flips and 90-degree rotations were baked into the archive and cannot all be identified by geometry.",
               "character_classes": CHARACTERS, "images": manifest}
    write_json(destination / "manifest.json", summary)
    print(json.dumps({"output_images": dict(counts), "skipped": dict(skipped)}, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--output", type=Path, default=Path("data/prepared"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-flagged", action="store_true", help="Also export suspicious geometry for manual review")
    parser.add_argument("--max-images", type=int, help="Small conversion smoke test only; may leave some splits empty")
    args = parser.parse_args()
    if args.max_images is not None and args.max_images < 1:
        parser.error("--max-images must be positive")
    prepare(args.zip, args.output, args.seed, args.include_flagged, args.max_images)
    print(f"Prepared datasets: {args.output.resolve()}")


if __name__ == "__main__":
    main()
