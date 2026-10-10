# Results and reproducibility

M0 establishes the harness only. No benchmark, drift, reliability, or serving experiment has been executed.

| Metric | Measurement | Evidence |
|---|---|---|
| PR-AUC | TBD (not yet measured) | M2 |
| Calibration / ECE | TBD (not yet measured) | M2 |
| Worst-slice quality | TBD (not yet measured) | M2 |
| Inference latency | TBD (not yet measured) | M3 |
| Hot-reload race errors | TBD (not yet measured) | M3 |
| Canary failure/rollback | TBD (not yet measured) | M4 |
| Drift detection / false alarms | TBD (not yet measured) | M5 |
| Gate accept + reject | TBD (not yet measured) | M6 |

Future artifacts must live under results/*.json, be emitted by scripts in bench/ or eval/, include OS, CPU/GPU, RAM, package versions, data hash, seed and git commit, and be rendered into README by another script. Negative results will be retained.

Environment probe (M0) can be reproduced with:

    python -m mlp doctor --json

Probe output is diagnostic metadata, not a benchmark result.
