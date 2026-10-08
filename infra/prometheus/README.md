# infra/prometheus

Scrape configuration, alerting rules and recording rules.

| File | Contents |
|---|---|
| `prometheus.yml` | Scrape jobs: `prometheus`, `backend-core:8000`, `ai-engine:8001`, `triton:8002` (optional) |
| `rules/voicewright.rules.yml` | 3 recording-rule groups + 9 alerts |

`docker-compose.yml` mounts both into the `prometheus` service — the `rules/`
mount is **required**, because `prometheus.yml` declares a `rule_files` glob and
Prometheus refuses to start if it matches nothing.

## Every series is real

The rules reference only metrics declared in `backend-core/app/metrics.py`
(`voicewright_active_webrtc_sessions`, `voicewright_webrtc_sessions_total`,
`voicewright_bargein_events_total`, `voicewright_vad_frames_processed_total`,
`voicewright_signaling_connections`) and `ai-engine/app/metrics.py` (`voicewright_ai_*`).
`infra/tests/test_observability.py` enforces this — a rule naming a metric nobody
emits fails the suite.

## Recording rules

`voicewright:stt_latency_p95_seconds`, `voicewright:llm_latency_p95_seconds`,
`voicewright:llm_first_token_p95_seconds`, `voicewright:tts_latency_p95_seconds`,
`voicewright:pipeline_latency_p95_seconds`, `voicewright:stt_audio_seconds_rate`,
`voicewright:tts_audio_seconds_rate`, `voicewright:stt_error_ratio`,
`voicewright:pipeline_error_ratio`.

Pre-computing the quantiles once per interval is cheaper than recomputing them in
every dashboard panel on every refresh.

## Alerts

`VoicewrightServiceDown` (critical) · `VoicewrightPipelineHighLatency` (p95 > 6 s) ·
`VoicewrightLLMFirstTokenHighLatency` (first-token p95 > 3 s) ·
`VoicewrightSTTHighErrorRatio` · `VoicewrightPipelineHighErrorRatio` ·
`VoicewrightNoActiveSessions` · `VoicewrightSignalingChurn` · `VoicewrightHighBargeInRate` ·
`VoicewrightVADBackendDegraded` (the energy fallback is running).

The pipeline-latency budget is 6 s rather than the old pre-LLM 1.5 s because the
generative stage replaced a ~30 ms intent classifier; the first-token alert is
the one that tracks whether the agent *feels* responsive.

`triton` is excluded from `VoicewrightServiceDown` on purpose: it is an optional
workload and its absence in the local stack is normal.

## Status

**Validated, never served.**

```bash
promtool check config infra/prometheus/prometheus.yml   # SUCCESS
promtool check rules  infra/prometheus/rules/voicewright.rules.yml  # SUCCESS: 18 rules
```

No Alertmanager configuration is present: alert routing (receivers, inhibition)
is deployment-specific and would be invented here.
