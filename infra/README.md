# infra

Phase 5 territory — **empty skeleton in Phase 1**. Directories here exist to
reserve layout only.

| Directory     | Intended contents (Phase 5)                                            |
| ------------- | ---------------------------------------------------------------------- |
| `k8s/`        | Raw Kubernetes manifests — Deployments, Services, Ingress, HPA, PDBs    |
| `helm/`       | Helm charts packaging the above, with per-environment values files      |
| `prometheus/` | Prometheus scrape config (populated in Phase 1, see below) and alerts   |

`prometheus/prometheus.yml` is the one real file here in Phase 1: a minimal,
valid scrape configuration targeting `backend-core`.
