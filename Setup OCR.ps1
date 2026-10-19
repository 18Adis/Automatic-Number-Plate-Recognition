$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$platePython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $platePython)) { throw 'Run Setup.ps1 first.' }
$plateUv = Get-Command uv -ErrorAction SilentlyContinue
if ($plateUv) {
    & $plateUv.Source pip install --python $platePython torch torchvision --index-url https://download.pytorch.org/whl/cpu
    if ($LASTEXITCODE -ne 0) { throw 'CPU PyTorch installation failed.' }
    & $plateUv.Source pip install --python $platePython -r requirements-ocr.txt --link-mode copy
} else {
    & $platePython -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    if ($LASTEXITCODE -ne 0) { throw 'CPU PyTorch installation failed.' }
    & $platePython -m pip install -r requirements-ocr.txt
}
if ($LASTEXITCODE -ne 0) { throw 'Additional OCR dependency installation failed.' }
Write-Host 'Tesseract must also be installed separately (including English language data).'
Write-Host 'Windows command: winget install --id UB-Mannheim.TesseractOCR --exact'
Write-Host 'Alternatively set PLATE_TESSERACT_CMD to your existing tesseract.exe path.'
& $platePython scripts\prepare_ocr.py
if ($LASTEXITCODE -ne 0) { throw 'One or more engines need attention. See the messages above.' }
Write-Host 'OCR engines ready. Restart PlateSight.' -ForegroundColor Green
