# Results

The project enforces an honest-numbers policy. Only a script-produced measurement artifact counts as a result.

| Evidence | Current status |
|---|---|
| Training PR-AUC, Brier, ECE, slices | MLflow run artifact emitted by training command; no committed report yet |
| Drift injection detection/false-alarm rates | Generate using `python eval/drift_injection.py`; no report committed yet |
| Smoke serving / reload race | CI acceptance pending |
| Canary auto-rollback | Deterministic behavior tests, not a sustained live load result |
| Promotion gate accept/reject | Deterministic paired-bootstrap tests, not a production promotion |
| One-command Docker Compose | Local reproduction instructions, CI verification pending |

Reports under `results/*.json` should contain measured values, hardware metadata, git commit, seed, environment versions and methodology. Failures must be retained as negative evidence, not deleted to improve the narrative.
