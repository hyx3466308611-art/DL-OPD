# COCONut 两档导出 + 全量 mix（主 segment >=768px，小 segment [512,768) 最多 2.5 万）
# 从仓库根目录: powershell -ExecutionPolicy Bypass -File scripts/build_phase1/rebuild_coconut_and_mix.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location $Root

python scripts/build_phase1/export_group1_coconut.py `
  --target-rows 140000 `
  --min-area 768 `
  --min-area-small 512 `
  --small-quota 8000 `
  --resume

python scripts/build_phase1/mix_phase1.py `
  --ref-cap 321327 `
  --coconut-cap 140000

Write-Host "Check: Get-Content data\phase1\phase1_mixed.meta.json"
