# Limitations and unresolved checks

## M0 boundaries

- No actual data validation, training, model registry, model serving, canary, drift, promotion, Compose platform, or dashboard yet. They are separate milestones.
- Dataset is **planned deterministic synthetic data**. No empirical evidence of predictive value, fairness or robustness to real-world class shifts.
- Python CI covers 3.11/3.12 on Linux. Windows is structurally supported by pathlib and the Python task runner but is not verified in CI yet.
- This execution environment runs Python 3.13.5 with no detected GPU/CUDA, no Docker and no external package-index DNS; dependency resolution, Docker workflows and clean-clone quickstart could not be confirmed locally.
- The project does not yet include a uv.lock. Direct dependencies are pinned in pyproject.toml, but transitive reproducibility still requires a locked resolution on a networked runner.
- No published performance, coverage or CI passing claim until an actual workflow returns success.
- The logger restricts which record attributes are emitted, but callers can still put private data in a free-text message. APIs must use fixed event names and sanitized bounded labels.
- CUDA probes report unknown when tooling is absent; they do not certify that the hardware has no GPU.
- No pretrained model, sensitive data, or artifact caching exists.

## Explicitly deferred edge-case coverage

All Project 07 M1–M8 edge cases, including split leakage, schema drift, unseen categories, NaN inputs, background reload races, canary stability, rollback, tiny drift windows, multiple testing, delayed labels, DB contention, model pickle safety, reproducibility and erase flows, are **not yet covered**. Each must acquire a test or an explicit unresolved limitation in its own milestone PR.

## Security defaults

No external services run in M0. Future Compose endpoints must bind locally by default; authentication, TLS, artifact checksums and storage cleanup are M3/M7 work. Never load arbitrary serialized model files.
