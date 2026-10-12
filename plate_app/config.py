"""Project-local paths and explicit model configuration."""

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _optional_path(name: str) -> Path | None:
    value = os.getenv(name, "").strip()
    if not value:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    model_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "models")
    detector_weights: Path | None = field(default_factory=lambda: _optional_path("PLATE_DETECTOR_WEIGHTS"))
    character_weights: Path | None = field(default_factory=lambda: _optional_path("PLATE_CHARACTER_WEIGHTS"))
    device: str = field(default_factory=lambda: os.getenv("PLATE_DEVICE", "cpu"))
    detector_name: str = "yolo-v9-t-384-license-plate-end2end"
    recognizer_name: str = "cct-xs-v2-global-model"
    max_upload_bytes: int = 12 * 1024 * 1024
    max_pixels: int = 20_000_000
    max_dimension: int = 10_000

    @property
    def model_mode(self) -> str:
        return "custom" if self.detector_weights or self.character_weights else "pretrained"
