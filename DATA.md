# Data provenance and use

## M0 dataset decision

**Dataset:** deterministic synthetic binary tabular classification data (generated on demand in M1). No dataset file is downloaded, bundled or committed. No real-world customers, IDs, faces, medical information, audio, images or other private records are used.

**External dataset license:** not applicable to algorithmically generated M1 examples. Generated fixture output remains disposable; M1 records its seed, git revision, Python and library versions, schema SHA-256, full-data SHA-256, and per-partition SHA-256. Synthetic datasets do not validate real-world fairness, base rates, or business performance.

## Implemented schema and split policy

Six bounded float64 features (f_0 to f_5), a three-valued synthetic channel (web/mobile/kiosk), and a binary int64 target. No personal identifiers or demographic claims. The generator uses scikit-learn make_classification with a fixed seed and an imbalanced target. Pandera rejects unexpected columns, unknown channels, NaN/inf, wrong dtypes, and out-of-range features.

The samples are independent, with no real entity groups or timestamp semantics. M1 uses stratified train/validation/test splits and rejects exact duplicate feature vectors to prevent overlap. Time-aware, group-aware, and wide/sparse validation is deferred pending a relevant dataset.

## Dependencies and provenance

Runtime: pydantic-settings (MIT), pydantic (MIT), pandas (BSD-3-Clause), NumPy (BSD-3-Clause), scikit-learn (BSD-3-Clause), and Pandera (MIT). Developer checks: ruff (MIT), mypy (MIT), pytest (MIT), pytest-cov (MIT), pre-commit (MIT), pip-audit (Apache-2.0), and gitleaks (MIT). Upstream license/NOTICE and transitive dependency audits must be reverified against the final locked versions before release. Do not infer a model or dataset license from a library license.

## Data handling

No raw production input retention. Only synthetic smoke data may be generated during tests. No data or weights committed to git. M1 generates content-hash manifests alongside disposable CSV outputs; future data deletion and artifact retention must be documented before deployment.

## Platform library and image license notes (M2–M8)

- MLflow: Apache-2.0 (upstream mlflow/mlflow).
- Pandera: MIT (upstream unionai-oss/pandera).
- **Grafana OSS Docker image: AGPL-3.0-only**, independently deployed, unmodified. Read its upstream LICENSING.md and check obligations before redistributing or adapting.
- Prometheus and the FastAPI/scikit-learn ecosystem: review pinned binary/wheel metadata before any distributable release; transitive licenses are not implied by this summary.

No external training dataset or pretrained model is shipped. Synthetic fixture creation depends only on the pinned scikit-learn generator and seed.
