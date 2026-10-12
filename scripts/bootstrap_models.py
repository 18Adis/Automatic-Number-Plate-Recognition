"""Download the two baseline models into this project and verify inference."""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plate_app.inference import InferenceEngine


def main() -> None:
    engine = InferenceEngine()
    print('Preparing the plate detector and character recognizer...', flush=True)
    print(json.dumps(engine.warmup(), indent=2))
    import numpy as np
    engine.predict(np.zeros((640, 640, 3), dtype=np.uint8))
    print('Model loading and inference verified. PlateSight is ready.')


if __name__ == '__main__':
    main()
