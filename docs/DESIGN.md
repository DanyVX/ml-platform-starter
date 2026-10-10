# Design and decision log

## Objective

Single-node reference MLOps platform for an internship portfolio. The differentiator is governance, reproducibility, safe model serving and observed behavior, not sophisticated prediction quality.

## Decision log

| Date | Question / options | Decision and rationale |
|---|---|---|
| 2026-10-10 | Real churn dataset, health/biometric dataset, or deterministic synthetic tabular data? | **Synthetic binary classification pilot.** No personal records, no external dataset license, deterministic small runs. Trade-off: no external validity; switch only after license/data provenance review. |
| 2026-10-10 | Full cluster or one node? | **Single-node local reference.** Docker Compose is enough to exhibit platform primitives without infrastructure cost or fragile cloud assumptions. |
| 2026-10-10 | GPU-first or CPU-first? | **CPU-first.** The runner has no detectable GPU; synthetic tiny configuration must complete quickly on modest laptops. |
| 2026-10-10 | DVC or content-hash manifests? | **Hash manifests in M1.** The pilot data is generated, not downloaded. DVC adds operational surface with little benefit here. |
| 2026-10-10 | Registry choice: custom database vs MLflow? | **MLflow planned for M2.** Registry versions and aliases are standard operational capabilities; aliases permit switch/rollback without changing serving code. Operational downside: services + artifact integrity need hardening. |
| 2026-10-10 | Drift: third-party monitor only or explicit implementation? | **PSI + KS planned for M5.** We can directly test false positives and detection rates; thresholds and multiple-testing strategy need evidence. |
| 2026-10-10 | CLI/task interface: Bash or Python? | **Python task runner with optional Makefile.** Works on Windows and Linux; no bash-only release path. |

## Metrics and guardrails (planned; not measured)

- **Primary:** PR-AUC on an untouched final test partition. Tune only on training/validation.
- **Secondary:** ROC-AUC and class-frequency baseline, plus calibration (ECE and Brier score).
- **Guardrails:** worst-slice quality, paired-bootstrap support for promotion, per-request p95 latency, artifact size, prediction validity, and smoke-serving test.
- **Promotion threshold:** TBD, to be specified and justified from validation/evaluation before M6. No implicit zero-margin promotion.
- **Drift detector:** numeric PSI + KS and categorical PSI/chi-square when M5 implements them; false-positive control and small-window insufficient-data state.

No model has been trained. Model claims, subgroup results and latency budgets are intentionally not fabricated.

## Working conventions

Every milestone is a separate PR. Commits represent one logical change. Logs must avoid secrets, raw PII, biometrics and embeddings. Real identity documents never enter fixtures. Results belong in JSON artifacts with environment + revision metadata. Model deserialization from untrusted files is forbidden. SQLite/MLflow/serving ports must not be exposed publicly without authentication and TLS.

## M1 implementation notes (2026-10-10)

The six-feature synthetic fixture is CPU-generated with an intentionally imbalanced binary label and a bounded synthetic channel. Strict Pandera schemas are shared between training and planned serving calls. M1 rejects unknown categories rather than silently changing model input dimensions; a later serving milestone can implement an explicit unknown-bucket counter. Dataset manifests use canonical CSV SHA-256, a schema fingerprint, split row counts and library versions. No normalization is fitted at data-generation time: scalers and encoders must be fitted on training rows only in M2.
