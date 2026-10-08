"""Static validation of the observability assets.

Two things are checked, neither of which needs a running Prometheus or Grafana:

1. Every dashboard is valid JSON and every panel has a query.
2. Every ``voicewright_*`` metric referenced by a dashboard or an alert/recording
   rule **actually exists** -- it is declared in backend-core/app/metrics.py or
   ai-engine/app/metrics.py (histograms expanded to their ``_bucket``/``_sum``/
   ``_count`` series), or it is a recording rule defined in the rules file.

(2) is the point: a dashboard panel querying a metric nobody emits renders an
empty graph forever and looks like "no traffic". This turns that into a failing
test.

Run from the repository root:

    ai-engine/.venv/bin/python -m pytest -o addopts="" infra/tests/
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

_INFRA = Path(__file__).resolve().parents[1]
_REPO = _INFRA.parent
_DASHBOARDS = sorted((_INFRA / "grafana" / "dashboards").glob("*.json"))
_RULES = _INFRA / "prometheus" / "rules" / "voicewright.rules.yml"
_METRIC_SOURCES = [
    _REPO / "backend-core" / "app" / "metrics.py",
    _REPO / "ai-engine" / "app" / "metrics.py",
]

#: Histogram base metrics expose these derived series.
_HIST_SUFFIXES = ("_bucket", "_sum", "_count")

_NAME_RE = re.compile(r"\b(voicewright_[a-z0-9_]+)")
_RECORD_RE = re.compile(r"^\s*-\s*record:\s*(voicewright:[a-z0-9_]+)", re.M)


def declared_metrics() -> set[str]:
    """Every ``voicewright_*`` series the two services declare, histograms expanded."""
    names: set[str] = set()
    for path in _METRIC_SOURCES:
        source = path.read_text()
        for name in _NAME_RE.findall(source):
            names.add(name)
            # A Histogram(...) declaration also exposes _bucket/_sum/_count.
            if f'Histogram(\n        "{name}"' in source or f'Histogram("{name}"' in source:
                names.update(name + s for s in _HIST_SUFFIXES)
    return names


def recording_rules() -> set[str]:
    return set(_RECORD_RE.findall(_RULES.read_text()))


def dashboard_queries() -> list[tuple[str, str]]:
    """(dashboard file, expression) for every panel target in every dashboard."""
    out: list[tuple[str, str]] = []
    for path in _DASHBOARDS:
        data = json.loads(path.read_text())
        for panel in data.get("panels", []):
            for target in panel.get("targets", []):
                expr = target.get("expr")
                if expr:
                    out.append((path.name, expr))
    return out


def test_dashboards_present():
    assert _DASHBOARDS, "no dashboard JSON found"


def test_dashboards_are_valid_json_with_panels():
    for path in _DASHBOARDS:
        data = json.loads(path.read_text())
        assert data.get("panels"), f"{path.name} has no panels"
        assert data.get("uid"), f"{path.name} has no uid"
        assert data.get("title"), f"{path.name} has no title"


def test_every_metric_reference_exists():
    declared = declared_metrics()
    recorded = recording_rules()
    assert declared, "no metrics parsed from the services -- check the regex"

    problems: list[str] = []
    for source, expr in dashboard_queries():
        for name in _NAME_RE.findall(expr):
            if name in declared:
                continue
            base = name
            for suffix in _HIST_SUFFIXES:
                if base.endswith(suffix):
                    base = base[: -len(suffix)]
                    break
            if base in declared:
                continue
            problems.append(f"{source}: unknown metric {name!r} in {expr!r}")

    # Recording-rule references (colon namespace) must be defined too.
    for source, expr in dashboard_queries():
        for name in re.findall(r"\b(voicewright:[a-z0-9_]+)", expr):
            if name not in recorded:
                problems.append(f"{source}: unknown recording rule {name!r} in {expr!r}")

    assert not problems, "dashboards reference metrics that do not exist:\n" + "\n".join(problems)


def test_alert_rules_reference_existing_metrics():
    text = _RULES.read_text()
    declared = declared_metrics()
    recorded = recording_rules()
    problems = []
    for expr in re.findall(r"expr:\s*[>|]?(.*)", text):
        for name in _NAME_RE.findall(expr):
            if name in declared:
                continue
            base = next((name[: -len(s)] for s in _HIST_SUFFIXES if name.endswith(s)), name)
            if base not in declared:
                problems.append(f"rules: unknown metric {name!r} in {expr.strip()!r}")
        for name in re.findall(r"\b(voicewright:[a-z0-9_]+)", expr):
            if name not in recorded:
                problems.append(f"rules: unknown recording rule {name!r}")
    assert not problems, "rules reference metrics that do not exist:\n" + "\n".join(problems)


def test_dashboard_datasource_uid_matches_provisioning():
    """Dashboards must point at the uid the datasource provisioning creates."""
    provisioned = (_INFRA / "grafana" / "provisioning" / "datasources" / "prometheus.yml").read_text()
    assert "uid: prometheus" in provisioned
    for path in _DASHBOARDS:
        data = json.loads(path.read_text())
        for panel in data.get("panels", []):
            ds = panel.get("datasource", {})
            if isinstance(ds, dict) and ds:
                assert ds.get("uid") == "prometheus", f"{path.name} panel {panel['id']} bad uid"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
