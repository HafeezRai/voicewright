# infra/k8s

Raw Kubernetes manifests for the CloudPSO Voice Agent.

| File | Contents |
|---|---|
| `namespace.yaml` | `cloudpso` namespace, restricted Pod Security Admission |
| `configmap.yaml` | non-secret config, mirroring docker-compose.yml's env |
| `secret.example.yaml` | **template only** — `secretKeyRef` contract, no values |
| `backend-core.yaml` | Deployment + Service (HTTP 8000, gRPC 50051) |
| `ai-engine.yaml` | PVC + Deployment + Service (HTTP 8001, gRPC 50052) |
| `frontend-dashboard.yaml` | Deployment + Service (5173) |
| `triton.yaml` | Optional GPU Deployment + Service + PVC |
| `hpa.yaml` | HPAs for backend-core and ai-engine |
| `pdb.yaml` | PodDisruptionBudgets |
| `ingress.yaml` | Dashboard + API ingress (signalling only, not media) |
| `kustomization.yaml` | `kubectl apply -k infra/k8s` |

Probes target the real endpoints: `/health` on both services (the frozen Phase 1
contract) and `/` on the dashboard.

## Status

**Validated, never deployed.** There is no cluster on the verification host.

```bash
kubectl kustomize infra/k8s | kubeconform -strict -summary -ignore-missing-schemas -
# 18 resources - Valid: 18, Invalid: 0, Errors: 0
```

`kubeconform` proves schema validity, not that a pod schedules or serves.

`secret.example.yaml` is deliberately excluded from `kustomization.yaml`: apply it
manually with real values (or, preferably, have an external secret store create
the Secret) — never commit a filled-in copy.
