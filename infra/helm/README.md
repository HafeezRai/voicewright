# infra/helm

**Empty skeleton — Phase 5 owns this.**

Helm charts packaging the Kubernetes manifests: one chart per deployable
(`backend-core`, `ai-engine`, `frontend-dashboard`), a shared library chart for
common labels/probes, and `values-<env>.yaml` overlays for development,
staging and production.
