#!/usr/bin/env bash
# Regenerate gRPC stubs from proto/voice_gateway.proto into app/gen/.
#
# Mirrors backend-core/scripts/gen_proto.sh: stubs are committed so the service
# builds without grpcio-tools at runtime, and this script keeps regeneration
# reproducible. Run from ai-engine/ (or via `make proto`). Requires grpcio-tools.
#
# ai-engine carries its own byte-identical copy of the proto (see proto/README.md)
# because its Docker build context is ./ai-engine and cannot reach the sibling
# backend-core/proto/. tests/test_proto_sync.py fails if the two ever diverge.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT_DIR="app/gen"
PROTO_DIR="proto"

# Use the project venv when it exists. The generated stubs are not
# interpreter-independent: protoc stamps the protobuf runtime version into the
# file, so regenerating with a different `grpc_tools`/`protobuf` produces
# different bytes from the same proto. Resolving the interpreter explicitly is
# what makes this script reproducible -- a bare `python` on PATH is whichever
# Python the developer happens to have first, which here was a system protobuf
# two minor versions off the one the service runs.
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
    PYTHON="$(command -v python3)"
    echo "warning: .venv not found; using $PYTHON" >&2
fi

echo "using $PYTHON ($("$PYTHON" -c 'import google.protobuf as p; print("protobuf", p.__version__)'))"

mkdir -p "$OUT_DIR"
touch "$OUT_DIR/__init__.py"

"$PYTHON" -m grpc_tools.protoc \
    -I "$PROTO_DIR" \
    --python_out="$OUT_DIR" \
    --grpc_python_out="$OUT_DIR" \
    "$PROTO_DIR/voice_gateway.proto"

# grpc_tools emits a top-level `import voice_gateway_pb2`, which does not resolve
# when the stubs live inside a package. Rewrite it to a package-absolute import
# so `from app.gen import voice_gateway_pb2_grpc` works.
PB2_GRPC="$OUT_DIR/voice_gateway_pb2_grpc.py"
if [ -f "$PB2_GRPC" ]; then
    "$PYTHON" - "$PB2_GRPC" <<'PY'
import re, sys
path = sys.argv[1]
src = open(path).read()
src = re.sub(
    r"^import voice_gateway_pb2 as",
    "from app.gen import voice_gateway_pb2 as",
    src,
    flags=re.MULTILINE,
)
open(path, "w").write(src)
PY
fi

echo "generated stubs in $OUT_DIR:"
ls -1 "$OUT_DIR"/voice_gateway_pb2*.py
