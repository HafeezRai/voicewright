# ai-engine developer targets.
.PHONY: proto test lint models venv install

# Regenerate gRPC stubs into app/gen/ (committed). Requires grpcio-tools.
proto:
	./scripts/gen_proto.sh

# Run the test suite (real inference; uses the local .venv).
test:
	.venv/bin/python -m pytest

lint:
	.venv/bin/python -m ruff check app tests || true

# Download the STT (Whisper) and TTS (Piper) weights into ./models.
models:
	./scripts/fetch_models.sh

venv:
	python3.11 -m venv .venv

install: venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt -r requirements-dev.txt
