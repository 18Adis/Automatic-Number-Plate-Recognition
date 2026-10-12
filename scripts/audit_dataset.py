"""Audit the original ZIP without extraction; optionally copy attributed examples."""
from __future__ import annotations

import argparse
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

try:
    from .dataset_utils import *
except ImportError:
    from dataset_utils import *


def build_report(zip_path, records, issues, metadata, seed=42):
    original_groups, original_sizes = defaultdict(set), Counter()
    hash_members = defaultdict(list)
    flags = Counter()
    for record in records:
        original_groups[record["original_split"]].add(record["source_group"])
        original_sizes[record["original_split"]] += 1
        hash_members[record["sha256"]].append(record["member"])
        flags.update(record["review_flags"])
    overlap = {}
    names = sorted(original_groups)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            overlap[f"{left} / {right}"] = len(original_groups[left] & original_groups[right])
    assignments = grouped_split(records, seed)
    duplicates = [members for members in hash_members.values() if len(members) > 1]
    return {
        "archive": str(zip_path.resolve()), "source_dataset": "ANPRIND Character-Detectionv1",
        "source_url": SOURCE_URL, "license": "CC BY 4.0", "seed": seed,
        "usable_images": len(records), "original_split_images": dict(original_sizes),
        "original_split_source_groups": {s: len(v) for s, v in original_groups.items()},
        "original_source_group_overlap": overlap,
        "distinct_source_groups": len({r["source_group"] for r in records}),
        "exact_duplicate_sets": len(duplicates),
        "redundant_exact_duplicate_images": sum(len(members) - 1 for members in duplicates),
        "duplicate_examples": duplicates[:20],
        "valid_character_annotations": sum(len(r["characters"]) for r in records),
        "unused_categories": {cid: name for cid, name in metadata["categories"].items()
                              if metadata["annotation_counts"].get(name, 0) == 0},
        "integrity_issue_counts": dict(Counter(i["issue"] for i in issues)),
        "integrity_issue_examples": issues[:100],
        "geometry_review_flag_counts": dict(flags),
        "geometry_flagged_images": sum(bool(r["review_flags"]) for r in records),
        "proposed_group_split_images": dict(Counter(s for s, _ in assignments.values())),
        "proposed_group_split_groups": dict(Counter(s for s, _ in set(assignments.values()))),
        "proposed_group_split_source_overlap": 0,
        "split_method": "80/10/10 by shuffled original-source groups; exact-byte duplicates join groups. Image ratios vary.",
        "limitations": [
            "Original export includes horizontal/vertical flips and 90-degree rotations. Existing transformations cannot be reliably reversed from COCO alone.",
            "Whole-plate boxes are absent. Enclosing characters creates approximate pseudo labels, not verified plate boundaries.",
            "Geometry flags find some rotations and disconnected text, but cannot reliably detect mirrored text, upside-down plates, or every multiple-plate image.",
            "Source-name and exact-byte grouping cannot detect every near-duplicate across different original filenames.",
            "Grayscale images resized to 640x640 do not cover all real camera lighting, color, aspect ratio, or blur conditions.",
            "Select clean upright originals and manually reviewed plate boundaries for a credible final benchmark.",
        ],
        **metadata,
    }


def copy_samples(zip_path, records, folder, count):
    folder.mkdir(parents=True, exist_ok=True)
    if any(folder.iterdir()):
        raise FileExistsError(f"Sample folder is not empty; choose a new folder: {folder}")
    candidates = [r for r in records if r["original_split"].endswith("test") and not r["review_flags"]]
    candidates.sort(key=lambda r: r["member"])
    selected, used_groups = [], set()
    with zipfile.ZipFile(zip_path) as archive:
        for record in candidates:
            if len(selected) >= count:
                break
            if record["source_group"] in used_groups:
                continue
            boxes = [c["box"] for c in record["characters"]]
            dx = max(b[2] for b in boxes) - min(b[0] for b in boxes)
            dy = max(b[3] for b in boxes) - min(b[1] for b in boxes)
            if dx / max(dy, 1) < 2.0:
                continue
            filename = f"sample-{len(selected) + 1:02d}.jpg"
            (folder / filename).write_bytes(bounded_read(archive, record["member"], MAX_IMAGE_BYTES))
            used_groups.add(record["source_group"])
            chars = sorted(record["characters"], key=lambda c: c["box"][0])
            selected.append({"file": filename, "source_archive_member": record["member"],
                             "source_group": record["source_group"], "annotation_count": len(chars),
                             "annotation_text_hint": "".join(c["name"] for c in chars),
                             "notice": "Unmodified source image. Text hint is a simple annotation ordering, not a model prediction or verified transcription."})
    manifest = {"source_dataset": "ANPRIND Character-Detectionv1", "source_url": SOURCE_URL,
                "license": "CC BY 4.0", "license_url": "https://creativecommons.org/licenses/by/4.0/",
                "attribution": "ANPRIND dataset, provided by a Roboflow user (number-plate-infqe); exported September 15, 2026.",
                "images": selected}
    write_json(folder / "manifest.json", manifest)
    return len(selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--output", type=Path, default=Path("data/dataset_audit.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=0, help="Copy up to this many original test examples (max 20)")
    parser.add_argument("--sample-dir", type=Path, default=Path("data/samples"))
    args = parser.parse_args()
    if not 0 <= args.samples <= 20:
        parser.error("--samples must be between 0 and 20")
    if args.output.exists():
        parser.error("Output already exists; choose a new --output path to preserve previous audit")
    records, issues, metadata = load_records(args.zip)
    report = build_report(args.zip, records, issues, metadata, args.seed)
    write_json(args.output, report)
    if args.samples:
        print(f"Copied {copy_samples(args.zip, records, args.sample_dir, args.samples)} attributed samples")
    print(json.dumps({k: report[k] for k in ("usable_images", "valid_character_annotations", "original_source_group_overlap",
                                             "exact_duplicate_sets", "geometry_flagged_images", "proposed_group_split_images")}, indent=2))
    print(f"Audit saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
