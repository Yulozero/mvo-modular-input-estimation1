param(
  [string]$Suite = "configs/experiment_design/suites/design_suite_full.yaml",
  [string]$Group = "",
  [switch]$Resume,
  [string]$PythonCmd = "py -3.12",
  [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$pythonParts = $PythonCmd.Split(" ", [System.StringSplitOptions]::RemoveEmptyEntries)
if ($pythonParts.Count -eq 0) {
  throw "PythonCmd is empty."
}

$pythonExe = $pythonParts[0]
$pythonArgsPrefix = @()
if ($pythonParts.Count -gt 1) {
  $pythonArgsPrefix = $pythonParts[1..($pythonParts.Count - 1)]
}

$moduleName = if ($Resume) { "src.runners.run_suite_resume" } else { "src.runners.run_suite" }
$argsList = @()
$argsList += $pythonArgsPrefix
$argsList += "-m"
$argsList += $moduleName
$argsList += "--suite"
$argsList += $Suite

if (-not $Resume -and $Group.Trim().Length -gt 0) {
  $argsList += "--group"
  $argsList += $Group
}

Write-Host "[INFO] ProjectRoot = $ProjectRoot"
Write-Host "[INFO] Python      = $PythonCmd"
Write-Host "[INFO] Suite       = $Suite"
if ($Group.Trim().Length -gt 0) {
  Write-Host "[INFO] Group       = $Group"
}
Write-Host "[INFO] Mode        = $(if ($Resume) { 'resume' } else { 'normal' })"
Write-Host "[INFO] DryRun      = $DryRun"
Write-Host "[INFO] Command     = $pythonExe $($argsList -join ' ')"

if ($DryRun) {
  exit 0
}

& $pythonExe @argsList
if ($LASTEXITCODE -ne 0) {
  throw "Experiment run failed with exit code $LASTEXITCODE"
}
