# Phase 7: MLOps & Production

## 7.1 CI/CD for AI
**Why it matters:** repeatable release process.
**What it is:** automated tests, builds, container pushes.
**Goal:** one-click pipeline from commit to staging.

### Steps
1. Add test jobs for lint, unit, integration.
2. Add model training / evaluation automation.
3. Add Docker + registry publish.
4. Add deployment workflow.

### Status
- [ ] planned
- [ ] in progress
- [ ] done

### Resources
- `.github/workflows/ci.yml`

---

## 7.2 Monitoring and Drift Detection
**Why it matters:** catches outages and model decay.
**What it is:** telemetry, metrics, alert rules.
**Goal:** dashboard + alert policy in place.

### Steps
1. Add app metrics (latency, error rate, model performance).
2. Hook Prometheus/Grafana or Datadog.
3. Add drift checks on prediction distribution.
4. Set alert thresholds and test.

### Status
- [ ] planned
- [ ] in progress
- [ ] done

### Resources
- `infra/monitoring/README.md`
