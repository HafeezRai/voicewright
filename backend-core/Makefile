# backend-core developer targets.
.PHONY: proto test lint

# Regenerate gRPC stubs into app/gen/ (committed). Requires grpcio-tools.
proto:
	./scripts/gen_proto.sh

# Run the test suite (uses the local .venv).
test:
	.venv/bin/python -m pytest

lint:
	.venv/bin/python -m ruff check app tests || true
