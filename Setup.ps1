$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$platePython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $platePython)) {
    $plateBasePython = $null
    $plateCandidates = @(
        (Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'),
        (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe'),
        (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python311\python.exe')
    )
    foreach ($plateCandidate in $plateCandidates) {
        if (Test-Path -LiteralPath $plateCandidate) { $plateBasePython = $plateCandidate; break }
    }
    if (-not $plateBasePython) {
        $plateCommand = Get-Command python -ErrorAction SilentlyContinue
        if ($plateCommand -and $plateCommand.Source -notlike '*WindowsApps*') { $plateBasePython = $plateCommand.Source }
    }
    if (-not $plateBasePython) { throw 'Install Python 3.10 or newer from python.org, then run Setup.ps1 again.' }
    & $plateBasePython -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the Python environment.' }
}
$plateUv = Get-Command uv -ErrorAction SilentlyContinue
$plateRequirements = if (Test-Path -LiteralPath 'requirements-lock.txt') { 'requirements-lock.txt' } else { 'requirements.txt' }
if ($plateUv) {
    & $plateUv.Source pip install --python $platePython -r $plateRequirements
} else {
    & $platePython -m ensurepip --upgrade
    & $platePython -m pip install -r $plateRequirements
}
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed. Check your internet connection and the error above.' }
& $platePython scripts\bootstrap_models.py
if ($LASTEXITCODE -ne 0) { throw 'Model download failed. Re-run Setup.ps1 when the connection is available.' }
Write-Host 'Ready. Double-click Start PlateSight.cmd.' -ForegroundColor Green
