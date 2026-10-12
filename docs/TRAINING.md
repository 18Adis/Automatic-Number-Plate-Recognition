# Train on the IURS / ANPRIND character dataset

The app already runs with pretrained models. Custom training is a separate step.
Review the prepared labels before starting a long GPU run.

## Audit and prepare

These tools read the ZIP directly and preserve it. Output directories must be empty.

```powershell
.\.venv\Scripts\python.exe scripts\audit_dataset.py --help
.\.venv\Scripts\python.exe scripts\prepare_dataset.py --zip "C:\datasets\IURS Character Detection Dataset.zip" --output data/prepared-new
```

Change the ZIP path to your local copy. The audit report is included; prepared
images and labels are generated locally and excluded from Git.
The new split groups source-name variants and exact duplicate bytes before
assigning 80% / 10% / 10% of groups. Image percentages differ with group size.
This prevents detected source-name and exact-byte leakage; related images with
different names or only perceptual similarity may still need manual grouping.

Plate labels are padded character enclosures, assuming one annotated plate per
image. They are unreviewed pseudo labels, not verified whole-plate ground truth.
Character training uses crops and transforms boxes into crop coordinates.
Review `manual_review.csv` and `manifest.json`. Geometry filtering does not
reliably detect horizontal flips or upside-down text. A fresh original export
with realistic augmentations is preferable for final accuracy measurements.
Review both retained images and excluded two-line plates so exclusions do not
remove an important deployment condition.

## Install training support

```powershell
uv pip install --python .venv/Scripts/python.exe -r requirements-training.txt
```

Install a CUDA-enabled PyTorch build from the official PyTorch selector if GPU
training is desired; verify `torch.cuda.is_available()` first. CUDA support is
not included in the lightweight baseline setup. The RTX 3050 has 4 GB VRAM;
start with batch 4 and lower it if memory runs out.

## Train

```powershell
.\.venv\Scripts\python.exe scripts\train.py --task plate --device 0 --batch 4 --epochs 50 --export-to models/plate-custom.pt
.\.venv\Scripts\python.exe scripts\train.py --task characters --device 0 --batch 4 --epochs 50 --export-to models/characters-custom.pt
```

Use `--device cpu` if CUDA is unavailable. Initial YOLO weights may download.
New training disables horizontal/vertical flips and mosaic, but cannot undo
augmentations baked into the archive. Scripts refuse to overwrite exported weights.

## Use custom weights

In PowerShell, from the project directory:

```powershell
$env:PLATE_DETECTOR_WEIGHTS = 'models/plate-custom.pt'
$env:PLATE_CHARACTER_WEIGHTS = 'models/characters-custom.pt'
$env:PLATE_DEVICE = '0'
.\.venv\Scripts\python.exe run.py --open-browser
```

Either model can be replaced independently. Character labels must be single
letters A–Z or digits 0–9; the plate detector must have one class. Keep this
PowerShell window open; these variables apply to its launched server process.
The app labels the active mode as custom. It sorts character boxes by row then
left to right, with overlap suppression. It does not guess substitutions for
ambiguous O/0 or I/1.

## Evaluate

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py detector --weights models/plate-custom.pt --data data/prepared/plates/data.yaml --device 0
.\.venv\Scripts\python.exe scripts\evaluate.py text --truth truth.csv --predictions predictions.csv
```

Text CSV files use `image,text` columns, one manually transcribed plate per image.
Missing predictions count as failures. Detector precision/recall/mAP do not
measure exact registration accuracy; report exact text accuracy and character
error rate separately on held-out, manually reviewed originals. Avoid claiming
real-world accuracy from approximate labels or a few smoke-test examples.
