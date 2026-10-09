# 一键构建（Windows / 本地冒烟）
$ErrorActionPreference = "Stop"
$Root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$Bd = Join-Path $Root "scripts\latet_bulid_phase1"
$Py = if ($env:PYTHON) { $env:PYTHON } else { "python" }

Set-Location $Root
Write-Host "=== ConVerSeg dataset rebuild ==="

& $Py (Join-Path $Bd "build_group1.py") --coconut-target 140000
& $Py (Join-Path $Bd "build_group2.py")
& $Py (Join-Path $Bd "build_group3.py")
& $Py (Join-Path $Bd "build_group4.py")
& $Py (Join-Path $Bd "mix_phase1_train.py") --g3-cap 48529 --coconut-cap 140000 --paco-cap 100000
& $Py (Join-Path $Bd "mix_phase2_train.py") --g3-cap 48529

Write-Host "Done. See data/phase1 and data/phase2"
