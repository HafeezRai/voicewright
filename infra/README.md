# infra

Phase 5. Kubernetes manifests, a Helm chart, Prometheus scrape + alert rules,
Grafana dashboards and the CI pipeline for the Voicewright Voice Agent.

## What is validated vs. what has been executed

This host is **arm64 macOS with no Kubernetes cluster and no NVIDIA GPU**. What
follows is deliberately blunt about which category each artifact is in.

| Artifact | Status | Evidence (exact command → result) |
|---|---|---|
| `k8s/*.yaml` (raw manifests) | **EXECUTED (validate)** | `kubectl kustomize infra/k8s` → 18 resources; `kubeconform -strict` → **18 valid, 0 invalid, 0 errors** |
| `helm/voicewright/` chart | **EXECUTED (lint + render)** | `helm lint` → 0 failed (4 value sets); `helm template` → 13/11/17/20 resources |
| Rendered Helm output | **EXECUTED (validate)** | `kubeconform -strict` over all rendered sets → **58 valid, 0 invalid, 3 skipped** (ServiceMonitor CRDs) |
| `prometheus/prometheus.yml` | **EXECUTED (validate)** | `promtool check config` → SUCCESS |
| `prometheus/rules/voicewright.rules.yml` | **EXECUTED (validate)** | `promtool check rules` → SUCCESS, 16 rules |
| `grafana/dashboards/*.json` | **EXECUTED (validate)** | `infra/tests/test_observability.py` → 5 passed; every metric cross-checked against the apps' `metrics.py` |
| `.github/workflows/ci.yml` | **EXECUTED (parse)** | Parses as YAML; job commands are the ones run by hand in this repo |
| `.github/workflows/ci.yml` — actually run on GitHub | **NOT EXECUTED** | No runner here; the workflow has never run |
| **Kubernetes cluster** | **NEVER RUN** | No cluster available on this host; nothing was deployed, no `kubectl apply` |
| **Prometheus / Grafana servers** | **NEVER RUN** | Configs are validated, not served |
| **Triton** | **NEVER RUN** | No NVIDIA GPU (Docker reports only the `runc` runtime, no `nvidia`) |

A manifest that passes `kubeconform` is a manifest that is *schema-valid*. It is
not a workload that schedules, starts, or serves. No claim here goes further than
what the table says.

## Layout

```
infra/
├── k8s/                  raw manifests + kustomization.yaml
├── helm/voicewright/        the chart (values.yaml + per-env overlays)
├── prometheus/           scrape config + rules/ (+ Alertmanager placeholder)
├── grafana/              dashboards (JSON) + provisioning
└── tests/                dashboard/alert ↔ metric cross-checks
```

## Deploying

Raw manifests:

```bash
kubectl apply -k infra/k8s
```

Helm (preferred — everything is parameterised):

```bash
helm install voicewright infra/helm/voicewright -f infra/helm/voicewright/values-staging.yaml
helm install voicewright infra/helm/voicewright -f infra/helm/voicewright/values-production.yaml
```

Both paths expect a `voicewright-secrets` Secret to already exist (see
`k8s/secret.example.yaml`); the chart does not create one unless
`secrets.create=true`, which is only for throwaway environments.

## Components and their real endpoints

Probes are wired to the **actual** health endpoints, not placeholders:

| Component | Ports | Probes |
|---|---|---|
| `backend-core` | 8000 HTTP/`/metrics`, 50051 gRPC | `/health` (frozen Phase 1 contract) |
| `ai-engine` | 8001 HTTP/`/metrics`, 50052 gRPC | `/health`, with a startup probe for cold model loads |
| `frontend-dashboard` | 5173 | `/` |
| `triton` (optional) | 8000/8001/8002 | `/v2/health/ready` |

## Observability

* **Prometheus** scrapes `backend-core:8000/metrics` and `ai-engine:8001/metrics`
  (plus an optional `triton:8002`). The rule file pre-computes stage p95s and
  error ratios and raises alerts for service-down, high pipeline latency, error
  ratios, signalling/VAD anomalies and VAD backend degradation.
* **Grafana** ships two dashboards — *Voice Gateway* and *STT/LLM/TTS Pipeline* —
  built **only** from series that exist in the two `metrics.py` files. The
  `infra/tests` suite enforces that: a panel querying a metric nobody emits fails
  the build.

## Validation commands

```bash
kubectl kustomize infra/k8s | kubeconform -strict -summary -ignore-missing-schemas -
helm lint infra/helm/voicewright
helm template voicewright infra/helm/voicewright | kubeconform -strict -summary -ignore-missing-schemas -
promtool check config infra/prometheus/prometheus.yml
promtool check rules  infra/prometheus/rules/voicewright.rules.yml
ai-engine/.venv/bin/python -m pytest -o addopts="" infra/tests/
```

## Known gaps / risks

* **Nothing here has been deployed.** There is no cluster to deploy to.
* **HPAs scale on CPU.** The signal that matters is concurrent sessions / p95
  latency; those need a custom-metrics adapter and are called out as a follow-up
  in `k8s/hpa.yaml` rather than quietly approximated.
* **`secret.example.yaml` is a template.** Real secrets come from an external
  secret store; the file contains no values.
* **Ingress carries signalling, not media.** WebRTC media is peer-to-peer or
  through a TURN relay, not through the ingress — documented in the manifest.
* **The AI engine's weights PVC is `ReadWriteMany`.** On a cluster without an RWX
  storage class this will stay `Pending`; staging overrides to `ReadWriteOnce`
  for one replica.
