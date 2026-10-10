# Limitations and unresolved acceptance items

This is a local **single-node reference**, not production infrastructure.

- **Deployment:** MLflow, Grafana and Prometheus bind to loopback at host level but lack production-ready authentication, TLS, network isolation, HA, or robust backups. Never expose to the internet.
- **Dataset:** deterministic, non-sensitive synthetic data. No external validity, demographic fairness claim, real fraud/KYC/biometric performance, or calibrated business threshold.
- **Model security:** internal joblib/MLflow sklearn loading uses pickle. SHA-256 on a local cache does not authenticate malicious tampering when the checksum manifest itself is writable by an attacker; restrict volume permissions or migrate to signed safer formats.
- **Tracking:** SQLite file backend has single-node write-lock limits; workflow_dispatch CI stores an ephemeral demonstration registry rather than shipping to shared cloud.
- **Monitoring:** PSI/KS thresholds may alert on seasonality; multiple-feature hypothesis testing, durable production reference refresh, and deployment-specific sensitivity are not yet calibrated. Synthetic injection is not a real-world alert guarantee.
- **Quality:** reproducible data hashing works, but full model-weight bit-for-bit reproduction and feature-version compatibility across arbitrary dependency upgrades are not assured.
- **Serving:** background reload runs within a process only; distributed cross-replica atomicity and verified no-error race benchmark remain unproven. Drift loop and delayed-label storage are example primitives and are not currently wired to a durable label service.
- **Rollback:** canary guard routes back to champion in-process; coordinated multi-instance rollback and persistent alerting require external orchestration.
- **CI:** release tag must not be created until all mandatory checks, full Compose demo, promoted/rejected paths and replay evidence pass.
- **Repo hygiene:** no generated inputs, weights, .env secrets, MLflow DB or large binaries may be committed.
- **Security audit:** every dependency upgrade requires rerunning pip-audit; dependency licenses require verification from installed wheel metadata before release.
