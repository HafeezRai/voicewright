# infra/helm

A single Helm chart, `cloudpso`, packaging the whole stack.

A single chart rather than one-per-service: the components share labels, a
ConfigMap and an ingress, and a chart-per-service would mean three releases that
must be installed in order to be coherent. `values.yaml` groups settings per
component so `--set backendCore.autoscaling.maxReplicas=12` still reads clearly.

```
cloudpso/
├── Chart.yaml
├── values.yaml                 # the contract: every template reads from here
├── values-staging.yaml         # 1 replica, RWO volume, no GPU, DEBUG logs
├── values-production.yaml      # HA, autoscaling, RWX volume, Triton on
└── templates/
    ├── _helpers.tpl            # names + the standard label set
    ├── configmap.yaml          # shared env
    ├── secret.yaml             # opt-in only (secrets.create)
    ├── backend-core.yaml       # Deployment + Service
    ├── ai-engine.yaml          # PVC + Deployment + Service
    ├── frontend.yaml           # Deployment + Service
    ├── triton.yaml             # optional GPU Deployment + Service + PVC
    ├── hpa.yaml, pdb.yaml, ingress.yaml, servicemonitor.yaml
    └── NOTES.txt
```

## Status

**Linted and rendered. Never installed.** There is no cluster on the
verification host.

```bash
helm lint infra/helm/cloudpso                                              # 0 failed
helm lint infra/helm/cloudpso -f infra/helm/cloudpso/values-staging.yaml  # 0 failed
helm template cloudpso infra/helm/cloudpso | \
  kubeconform -strict -summary -ignore-missing-schemas -
```

Rendered resource counts: default 13, staging 11, production 17,
all-options-enabled 20. `kubeconform` over every set: 58 valid, 0 invalid, 3
skipped (the Prometheus Operator `ServiceMonitor` CRDs, which have no public
schema).

## Installing

```bash
helm install cloudpso infra/helm/cloudpso -f infra/helm/cloudpso/values-production.yaml
```

`secrets.create` defaults to **false**: credentials belong in an external secret
store, and a Secret rendered from `values.yaml` ends up in `helm get values` and
the release history. Point `secrets.existingSecret` at a Secret you manage.
