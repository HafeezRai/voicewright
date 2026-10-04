# infra/prometheus

Phase 1 provides `prometheus.yml` — a minimal, valid scrape configuration that
targets `backend-core` over the compose network. Phase 5 extends this directory
with alerting rules (`rules/*.yml`), Alertmanager configuration, and recording
rules for the STT/NLU/TTS latency SLOs.

The compose stack mounts `prometheus.yml` read-only into the `prometheus`
service at `/etc/prometheus/prometheus.yml`.
