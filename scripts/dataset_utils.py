"""Safe COCO archive reading, character geometry, and leakage-resistant splitting.

All archive paths are data. Files are read directly and never extracted by name.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import random
import re
import statistics
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

from PIL import Image

CHARACTERS = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
MAX_JSON_BYTES = 64 * 1024 * 1024
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_TOTAL_BYTES = 2 * 1024 * 1024 * 1024
MAX_PIXELS = 25_000_000
DEFAULT_ZIP = Path.home() / "Downloads" / "IURS Character Detection Dataset.zip"
SOURCE_URL = "https://universe.roboflow.com/number-plate-infqe/anprind"


def safe_member(name: str) -> str:
    path = PurePosixPath(name)
    if not name or "\\" in name or path.is_absolute() or ".." in path.parts or ":" in name:
        raise ValueError(f"Unsafe archive member: {name!r}")
    return str(path)


def bounded_read(archive: zipfile.ZipFile, name: str, limit: int) -> bytes:
    name = safe_member(name)
    info = archive.getinfo(name)
    if info.file_size > limit:
        raise ValueError(f"Archive member exceeds {limit} byte limit: {name}")
    with archive.open(info) as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"Archive member expanded beyond limit: {name}")
    return data


def source_group(image: dict) -> str:
    """Collapse Roboflow variants of the same original source photograph."""
    extra = image.get("extra") or {}
    name = extra.get("name") or image["file_name"]
    name = PurePosixPath(str(name).replace("\\", "/")).name.lower()
    name = re.split(r"\.rf\.", name)[0]
    name = re.sub(r"(?:[._](?:jpe?g|png|webp))$", "", name)
    return name


def clipped_bbox(box, width: int, height: int):
    """COCO xywh -> clipped xyxy; reject malformed/nonfinite/empty boxes."""
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ValueError("bbox must contain four numbers")
    try:
        x, y, w, h = (float(v) for v in box)
    except (TypeError, ValueError) as exc:
        raise ValueError("bbox must contain four numbers") from exc
    if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
        raise ValueError("bbox has nonfinite coordinates or nonpositive area")
    if width <= 0 or height <= 0:
        raise ValueError("invalid image dimensions")
    clipped = (max(0.0, x), max(0.0, y), min(float(width), x + w), min(float(height), y + h))
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        raise ValueError("bbox is outside image")
    return clipped


def yolo_bbox(box, width: int, height: int, origin=(0.0, 0.0)):
    """Normalize xyxy against full image or a crop with the supplied origin."""
    x1, y1, x2, y2 = box
    ox, oy = origin
    clipped = clipped_bbox([x1 - ox, y1 - oy, x2 - x1, y2 - y1], width, height)
    x1, y1, x2, y2 = clipped
    return ((x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
            (x2 - x1) / width, (y2 - y1) / height)


def plate_crop(boxes, width: int, height: int):
    if not boxes:
        raise ValueError("Cannot derive plate without character boxes")
    x1, y1 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x2, y2 = max(b[2] for b in boxes), max(b[3] for b in boxes)
    px, py = max(3.0, (x2 - x1) * .10), max(3.0, (y2 - y1) * .20)
    return (max(0, math.floor(x1 - px)), max(0, math.floor(y1 - py)),
            min(width, math.ceil(x2 + px)), min(height, math.ceil(y2 + py)))


def plate_issues(boxes):
    """Conservative review flags, not a reliable orientation/plate classifier."""
    if not boxes:
        return ["no_valid_characters"]
    issues = []
    if len(boxes) < 4 or len(boxes) > 14:
        issues.append("unusual_character_count")
    dx = max(b[2] for b in boxes) - min(b[0] for b in boxes)
    dy = max(b[3] for b in boxes) - min(b[1] for b in boxes)
    if dx / max(dy, 1) < 1.1:
        issues.append("possibly_rotated_or_tall_plate")
    if dx / max(dy, 1) > 10:
        issues.append("unusually_wide_character_region")
    # Connected components separated by several character heights are likely
    # multiple plates, unrelated text, or distant incorrect annotations.
    scale = statistics.median(max(b[2] - b[0], b[3] - b[1]) for b in boxes)
    unseen, components = set(range(len(boxes))), 0
    while unseen:
        components += 1
        pending = [unseen.pop()]
        while pending:
            a = boxes[pending.pop()]
            for idx in list(unseen):
                b = boxes[idx]
                gapx = max(a[0] - b[2], b[0] - a[2], 0)
                gapy = max(a[1] - b[3], b[1] - a[3], 0)
                if math.hypot(gapx, gapy) <= 3 * scale:
                    unseen.remove(idx)
                    pending.append(idx)
    if components > 1:
        issues.append("disconnected_character_groups")
    return issues


class UnionFind:
    def __init__(self, values):
        self.parent = {v: v for v in values}

    def find(self, value):
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left, right):
        a, b = sorted((self.find(left), self.find(right)))
        self.parent[b] = a


def grouped_split(records, seed=42, ratios=(.8, .1, .1)):
    """Keep original-source variants AND exact duplicate bytes in one split.

    The ratios apply to groups; image ratios may differ because group sizes vary.
    """
    if len(ratios) != 3 or any(r <= 0 for r in ratios) or abs(sum(ratios) - 1) > 1e-8:
        raise ValueError("train/val/test ratios must be positive and sum to one")
    union = UnionFind(r["source_group"] for r in records)
    hashes = {}
    for record in records:
        digest = record.get("sha256")
        if digest:
            if digest in hashes:
                union.union(record["source_group"], hashes[digest])
            else:
                hashes[digest] = record["source_group"]
    groups = sorted({union.find(r["source_group"]) for r in records})
    random.Random(seed).shuffle(groups)
    train_end = int(len(groups) * ratios[0])
    val_end = train_end + int(len(groups) * ratios[1])
    mapping = {g: "train" if i < train_end else "val" if i < val_end else "test"
               for i, g in enumerate(groups)}
    return {r["member"]: (mapping[union.find(r["source_group"])], union.find(r["source_group"]))
            for r in records}


def load_records(zip_path: Path, verify_images=True):
    records, issues, categories, annotation_totals = [], [], {}, Counter()
    with zipfile.ZipFile(zip_path) as archive:
        infos = archive.infolist()
        if len(infos) > 50_000 or sum(i.file_size for i in infos) > MAX_TOTAL_BYTES:
            raise ValueError("Archive exceeds file-count or uncompressed-size safety limit")
        if len({i.filename for i in infos}) != len(infos):
            raise ValueError("Archive has ambiguous duplicate member names")
        json_members = sorted(i.filename for i in infos if i.filename.endswith("_annotations.coco.json"))
        if not json_members:
            raise ValueError("No COCO _annotations.coco.json files found")
        for member in json_members:
            data = json.loads(bounded_read(archive, member, MAX_JSON_BYTES))
            by_id = {}
            for cat in data.get("categories", []):
                by_id[cat["id"]] = cat["name"]
                categories[str(cat["id"])] = cat["name"]
            annotations = defaultdict(list)
            image_ids = {image["id"] for image in data["images"]}
            if len(image_ids) != len(data["images"]):
                raise ValueError(f"Duplicate image IDs in {member}")
            for ann in data["annotations"]:
                annotation_totals[by_id.get(ann.get("category_id"), "<unknown>")] += 1
                if ann.get("image_id") not in image_ids:
                    issues.append({"member": member, "annotation_id": ann.get("id"), "issue": "orphan_annotation"})
                else:
                    annotations[ann["image_id"]].append(ann)
            for image in data["images"]:
                filename = safe_member(str(image["file_name"]))
                path = safe_member(str(PurePosixPath(member).parent / filename))
                try:
                    width, height = int(image["width"]), int(image["height"])
                    if width <= 0 or height <= 0 or width * height > MAX_PIXELS:
                        raise ValueError("invalid/oversized image dimensions")
                    content = bounded_read(archive, path, MAX_IMAGE_BYTES)
                    if verify_images:
                        with Image.open(io.BytesIO(content)) as decoded:
                            if decoded.width * decoded.height > MAX_PIXELS:
                                raise ValueError("decoded image exceeds pixel limit")
                            if decoded.size != (width, height):
                                raise ValueError("COCO and decoded image dimensions differ")
                            decoded.verify()
                except (KeyError, OSError, ValueError) as exc:
                    issues.append({"member": path, "issue": "unusable_image", "detail": str(exc)})
                    continue
                chars = []
                seen = set()
                for ann in annotations[image["id"]]:
                    name = by_id.get(ann.get("category_id"))
                    if name not in CHARACTERS:
                        issues.append({"member": path, "annotation_id": ann.get("id"), "issue": "non_character_category"})
                        continue
                    try:
                        box = clipped_bbox(ann.get("bbox"), width, height)
                    except ValueError as exc:
                        issues.append({"member": path, "annotation_id": ann.get("id"), "issue": "invalid_bbox", "detail": str(exc)})
                        continue
                    raw = ann["bbox"]
                    if raw[0] < 0 or raw[1] < 0 or raw[0] + raw[2] > width or raw[1] + raw[3] > height:
                        issues.append({"member": path, "annotation_id": ann.get("id"), "issue": "clipped_bbox"})
                    key = (name, box)
                    if key in seen:
                        issues.append({"member": path, "annotation_id": ann.get("id"), "issue": "duplicate_annotation"})
                        continue
                    seen.add(key)
                    chars.append({"name": name, "class_id": CHARACTERS.index(name), "box": box})
                records.append({"member": path, "original_split": str(PurePosixPath(member).parent),
                                "source_group": source_group(image), "width": width, "height": height,
                                "sha256": hashlib.sha256(content).hexdigest(), "characters": chars,
                                "review_flags": plate_issues([c["box"] for c in chars])})
        metadata = {"categories": categories, "annotation_counts": dict(annotation_totals),
                    "uncompressed_bytes": sum(i.file_size for i in infos),
                    "archive_members": len(infos), "coco_files": json_members}
    return records, issues, metadata


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def unique_stem(record):
    return f"{re.sub('[^a-z0-9_-]', '_', record['source_group'])[:64]}_{hashlib.sha256(record['member'].encode()).hexdigest()[:16]}"
