# infra/prometheus

Scrape configuration, alerting rules and recording rules.

| File | Contents |
|---|---|
| `prometheus.yml` | Scrape jobs: `prometheus`, `backend-core:8000`, `ai-engine:8001`, `triton:8002` (optional) |
| `rules/cloudpso.rules.yml` | 3 recording-rule groups + 8 alerts |

`docker-compose.yml` mounts both into the `prometheus` service — the `rules/`
mount is **required**, because `prometheus.yml` declares a `rule_files` glob and
Prometheus refuses to start if it matches nothing.

## Every series is real

The rules reference only metrics declared in `backend-core/app/metrics.py`
(`cloudpso_active_webrtc_sessions`, `cloudpso_webrtc_sessions_total`,
`cloudpso_bargein_events_total`, `cloudpso_vad_frames_processed_total`,
`cloudpso_signaling_connections`) and `ai-engine/app/metrics.py` (`cloudpso_ai_*`).
`infra/tests/test_observability.py` enforces this — a rule naming a metric nobody
emits fails the suite.

## Recording rules

`cloudpso:stt_latency_p95_seconds`, `cloudpso:nlu_latency_p95_seconds`,
`cloudpso:tts_latency_p95_seconds`, `cloudpso:pipeline_latency_p95_seconds`,
`cloudpso:stt_audio_seconds_rate`, `cloudpso:tts_audio_seconds_rate`,
`cloudpso:stt_error_ratio`, `cloudpso:pipeline_error_ratio`.

Pre-computing the quantiles once per interval is cheaper than recomputing them in
every dashboard panel on every refresh.

## Alerts

`CloudPSOServiceDown` (critical) · `CloudPSOPipelineHighLatency` (p95 > 1.5 s) ·
`CloudPSOSTTHighErrorRatio` · `CloudPSOPipelineHighErrorRatio` ·
`CloudPSONoActiveSessions` · `CloudPSOSignalingChurn` · `CloudPSOHighBargeInRate` ·
`CloudPSOVADBackendDegraded` (the energy fallback is running).

`triton` is excluded from `CloudPSOServiceDown` on purpose: it is an optional
workload and its absence in the local stack is normal.

## Status

**Validated, never served.**

```bash
promtool check config infra/prometheus/prometheus.yml   # SUCCESS
promtool check rules  infra/prometheus/rules/cloudpso.rules.yml  # SUCCESS: 16 rules
```

No Alertmanager configuration is present: alert routing (receivers, inhibition)
is deployment-specific and would be invented here.
