# MoreNeck setup for Windows. Run from the repo folder:  .\setup.bat   (or: powershell -ExecutionPolicy Bypass -File setup.ps1)
# Creates .venv, installs CPU-only PyTorch + the project, downloads the YOLO pose weights, then checks the hardware.
# CPU-only on purpose: works the same on every laptop, no GPU drivers needed.
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$ModelUrl  = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt"
$ModelPath = "data\models\yolo11n-pose.pt"
$Venv      = ".venv"
$Py        = "$Venv\Scripts\python.exe"

# Windows needs Python 3.13+: before 3.13, time.monotonic() there ticks in 15.6 ms steps, too coarse
# for the 50 Hz neck loop and the perception timestamps.
$VersionOk = "import sys, struct; sys.exit(0 if (3,13) <= sys.version_info[:2] <= (3,14) and struct.calcsize('P') == 8 else 1)"

function Find-Python {
    foreach ($c in @("py -3.13", "py -3.14", "py -3", "python", "python3")) {
        $parts = $c.Split(" ")
        $exe = $parts[0]
        $pyArgs = @($parts | Select-Object -Skip 1)
        if (-not (Get-Command $exe -ErrorAction SilentlyContinue)) { continue }
        try {
            & $exe @pyArgs -c $VersionOk 2>$null
            if ($LASTEXITCODE -eq 0) { return $c }
        } catch { }
    }
    return $null
}

if (Test-Path $Py) {
    & $Py -c $VersionOk 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Existing .venv uses an unsupported Python; recreating it."
        Remove-Item -Recurse -Force $Venv
    }
}

if (-not (Test-Path $Py)) {
    $python = Find-Python
    if ($null -eq $python) {
        Write-Host "Python 3.13 (64-bit) not found." -ForegroundColor Red
        Write-Host "Install it with:  winget install Python.Python.3.13   (or from python.org), then re-run setup.bat"
        exit 1
    }
    $parts = $python.Split(" "); $exe = $parts[0]; $pyArgs = @($parts | Select-Object -Skip 1)
    Write-Host "Creating virtual environment with: $python"
    & $exe @pyArgs -m venv $Venv
    if ($LASTEXITCODE -ne 0) { throw "venv creation failed" }
}

Write-Host "`n== Installing CPU-only PyTorch ==" -ForegroundColor Cyan
& $Py -m pip install --upgrade pip
& $Py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { throw "PyTorch install failed" }

Write-Host "`n== Installing MoreNeck and its dependencies ==" -ForegroundColor Cyan
& $Py -m pip install -e ".[dev,vision]"
if ($LASTEXITCODE -ne 0) { throw "project install failed" }

if (-not (Test-Path $ModelPath)) {
    Write-Host "`n== Downloading YOLO11n pose weights (~6 MB) ==" -ForegroundColor Cyan
    New-Item -ItemType Directory -Force -Path (Split-Path $ModelPath) | Out-Null
    $ProgressPreference = "SilentlyContinue"
    Invoke-WebRequest -Uri $ModelUrl -OutFile $ModelPath
}

Write-Host "`n== Checking hardware ==" -ForegroundColor Cyan
& $Py tools\check_devices.py

Write-Host "`nSetup done. Start the head with:  run.bat" -ForegroundColor Green
