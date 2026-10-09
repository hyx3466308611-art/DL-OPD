# Phase-1 build (Windows). From repo root:
#   pip install pycocotools tqdm pillow numpy
#   powershell -ExecutionPolicy Bypass -File scripts/build_phase1/run_build.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location $Root

$Py = "python"
$BD = "scripts/build_phase1"

Write-Host "== Group3 (HF paths only) =="
& $Py "$BD/import_group3_hf.py"

Write-Host "== Group2 Ref (~321K, long) =="
& $Py "$BD/export_group2_ref.py"

Write-Host "== Group1 COCO =="
& $Py "$BD/export_group1_coco.py" --max-rows 200000

Write-Host "== Group1 COCONut =="
& $Py "$BD/export_group1_coconut.py" --max-rows 140000

Write-Host "== Group1 PACO =="
& $Py "$BD/export_group1_paco.py" --max-rows 100000

Write-Host "== Mix =="
& $Py "$BD/mix_phase1.py"

Write-Host "Done. Train with --data-dir $Root\data --dataset-jsonl phase1/phase1_mixed.jsonl"
