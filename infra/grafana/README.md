# infra/grafana

Two dashboards, provisioned as code.

| Dashboard | UID | Panels |
|---|---|---|
| `dashboards/cloudpso-overview.json` | `cloudpso-overview` | 9 — sessions, signalling, barge-in, VAD |
| `dashboards/cloudpso-ai-pipeline.json` | `cloudpso-ai-pipeline` | 11 — STT/LLM/TTS latency, throughput, errors |

Provisioning lives in `provisioning/`: a Prometheus datasource with uid
`prometheus` (which every panel references) and a file-based dashboard provider.

## Panels are built only from metrics that exist

Every query names a series declared in `backend-core/app/metrics.py` or
`ai-engine/app/metrics.py` — or a recording rule from
`infra/prometheus/rules/cloudpso.rules.yml`. Nothing is invented; a panel
querying a metric that is never emitted renders an empty graph forever and looks
like "no traffic", which is the failure this avoids.

`infra/tests/test_observability.py` enforces it:

```bash
ai-engine/.venv/bin/python -m pytest -o addopts="" infra/tests/
# 5 passed — including: every metric reference resolves to a declaration
```

## Status

**Dashboards are validated JSON with verified metric references. Grafana itself
was never run** — there is no Grafana instance on the verification host, so no
panel has been rendered against live data.

## Running it

Add to the compose stack (not wired in by default):

```yaml
grafana:
  image: grafana/grafana:11.2.0
  ports: ["3000:3000"]
  volumes:
    - ./infra/grafana/provisioning:/etc/grafana/provisioning:ro
    - ./infra/grafana/dashboards:/var/lib/grafana/dashboards:ro
  environment:
    GF_AUTH_ANONYMOUS_ENABLED: "true"
  networks: [cloudpso-net]
```
