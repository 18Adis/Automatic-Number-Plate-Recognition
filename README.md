# PlateSight — Automatic Number Plate Recognition

Local number plate detection and recognition with image upload, webcam snapshots,
continuous camera frames, and video playback. A FastAPI service runs detection and
OCR on your computer; the browser displays plate crops, boxes, confidence scores
and editable detection history. No account or cloud inference is required.

## Start

On Windows, clone or download this repository, install Python 3.10–3.13, then
run `Setup.ps1` in PowerShell from the project folder:

```powershell
git clone https://github.com/18Adis/Automatic-Number-Plate-Recognition.git
cd Automatic-Number-Plate-Recognition
powershell -ExecutionPolicy Bypass -File .\Setup.ps1
```

Double-click **Start PlateSight.cmd**, then open http://127.0.0.1:8765.
Keep the launcher open while using the app. Close it or press Ctrl+C to stop.
Setup downloads packages and two pretrained models. After
setup, ordinary recognition uses the cached models without internet access.

## Use

- **Upload image:** choose a vehicle photo.
- **OCR method:** select Plate OCR (CCT), EasyOCR, Tesseract, or PaddleOCR above
  the image preview. The selected method applies to images, snapshots, live camera,
  and video. Switching methods pauses analysis; select Analyze or Resume to continue.
- **Compare OCR:** compares all installed methods on the same detected plate crops.
  For camera/video it freezes the current frame. Enter the correct plate number
  optionally to show exact match and character error rate, then export the comparison
  as CSV. Comparisons do not add duplicate entries to detection history.
- **Live camera:** start the webcam, allow browser camera access, and point it at a plate.
  Pause stops analysis; Stop releases the camera. The snapshot control freezes one frame.
- **Video:** choose a browser-supported video, then Play & analyze. Video stays in the browser;
  sampled JPEG frames go to the local service. This is playback analysis, not a complete
  frame-by-frame batch export.
- **History:** opens in a separate tab from the sidebar. Click a plate thumbnail to
  enlarge the crop, view the vehicle image, zoom, or adjust contrast for viewing.
  Edit an incorrect number or download CSV. Live readings require at least
  three matching reliable readings in five observations and are deduplicated for 30 seconds.

The frontend resizes image/camera inputs to at most 1280 pixels per side. Browser
image selection allows 30 MB; each transmitted frame must be at most 12 MB. The
API accepts JPEG, PNG and WebP, up to 20 megapixels and 10,000 pixels per side.

## Additional OCR methods

The baseline setup installs CCT. On another computer, add the other engines with
the following commands after `Setup.ps1` (Python 3.12 is the tested configuration):

```powershell
winget install --id UB-Mannheim.TesseractOCR --exact
powershell -ExecutionPolicy Bypass -File ".\Setup OCR.ps1"
```

EasyOCR and PaddleOCR download their models once to `models/`. Tesseract needs
English language data; an existing installation can be selected with
`PLATE_TESSERACT_CMD`. The dropdown marks missing installations as needing setup.
All four methods run locally on CPU. History and its CSV export record the OCR
method and per-plate OCR time. See [OCR comparison guide](docs/OCR_COMPARISON.md)
for the comparison protocol, model details, and timing definitions.

## Models and limits

The running baseline uses FastALPR, the YOLOv9-t 384 plate detector, and the
CCT-XS-v2 global plate recognizer through CPU ONNX Runtime. It has **not been
trained on the IURS dataset**. A compatible GPU can be used for optional custom training.
Confidence scores are model estimates, not proof that a registration is correct.
Inspect readings especially for small, rotated, blurred, night or two-line plates.
Perspective correction is not implemented in this version.

Real-model smoke predictions and timings are in `data/verification.json`.
These five examples are not an accuracy benchmark.

## Dataset preparation

The original ZIP is preserved. `data/dataset_audit.json` records 8,168 images,
79,792 character annotations, 1,385 source-name groups and 21 exact duplicate pairs.
The original splits share source images. The preparation tool assigns related
source names and exact byte duplicates to the same new train/validation/test split.

The local preparation run produced 4,266 retained examples (3,451 train, 402 validation,
413 test) for each of plate detection and cropped character detection. It flags
3,902 examples for geometry review. The manifest and `manual_review.csv` record
the decisions. The dataset ZIP, prepared images, installed packages, model cache,
and personal detection history are excluded from Git. Use the preparation scripts
with your own copy of the dataset. Plate boxes are approximate padded enclosures of character labels;
they need visual review before serious training or evaluation. Existing flipped
and rotated images cannot all be detected by geometry. See [training guide](docs/TRAINING.md).

## Development

```powershell
.\.venv\Scripts\python.exe run.py
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\bootstrap_models.py
```

API documentation: http://127.0.0.1:8765/api/docs.
Recognition models are in `models/`; detection history is in `data/history.db`.
Each saved detection includes a compressed JPEG capture (up to 1600 pixels per side)
and plate crop stored locally in that database. Unconfirmed live frames are not saved.
Clearing history deletes the saved readings and their images. Entries created before
image storage was added remain available with no image. Contrast adjustment only
changes the viewer; it does not alter the stored capture or recover missing detail.
This server is
bound to localhost and is intended for one local user; remote hosting would need
authentication, HTTPS, and deployment work.

## References and attribution

- [FastALPR](https://github.com/ankandrew/fast-alpr), MIT.
- [Open Image Models](https://github.com/ankandrew/open-image-models).
- [Fast Plate OCR](https://github.com/ankandrew/fast-plate-ocr).
- [ANPRIND dataset](https://universe.roboflow.com/number-plate-infqe/anprind), provided by
  a Roboflow user (number-plate-infqe), CC BY 4.0. Exported September 15, 2026.
  The local smoke tests use sample images from this dataset. Dataset images are
  excluded from Git; supply your own images for recognition.
- Optional Ultralytics training has its own AGPL-3.0 / commercial licensing terms.

The repository's original notebook is preserved in `legacy/`. The runnable
application is now the FastAPI service and browser dashboard described above.
