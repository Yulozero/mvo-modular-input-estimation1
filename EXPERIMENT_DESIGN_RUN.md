# Experiment Design Offline Run

This repository now includes a design-first configuration set under:

- `configs/experiment_design/base_design.yaml`
- `configs/experiment_design/presets/`
- `configs/experiment_design/suites/design_suite_full.yaml`

All outputs from this new design-first setup are written to:

- `outputs_experiment_design/`

## One-click in VSCode

Open the workspace in VSCode and run one of these tasks:

- `Experiment Design: Full Suite`
- `Experiment Design: Resume Full Suite`
- `Experiment Design: Run Group`

## One-click in terminal

Run the full suite:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_design_suite_offline.ps1
```

Resume the full suite:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_design_suite_offline.ps1 -Resume
```

Run one group only:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_design_suite_offline.ps1 -Group "Mean Comparison"
```

Or use the batch wrapper:

```bat
run_design_suite.bat
run_design_suite.bat -Resume
run_design_suite.bat -Group "Generalization"
```

Preview the command only without running training:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_design_suite_offline.ps1 -Group "Main Result" -DryRun
```

## Notes

- The old `fusion_no_kg_*` configs are untouched.
- The new config tree is design-first and uses experiment names aligned with the experiment-design document.
- The default Python command is `py -3.12`. You can override it:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_design_suite_offline.ps1 -PythonCmd "D:\ana\python.exe"
```
