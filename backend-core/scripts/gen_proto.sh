#!/usr/bin/env bash
# Regenerate gRPC stubs from proto/voice_gateway.proto into app/gen/.
#
# Stubs are committed so the service builds without grpcio-tools at runtime;
# this script keeps regeneration reproducible. Run from backend-core/ (or via
# `make proto`). Requires grpcio-tools in the active environment.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT_DIR="app/gen"
PROTO_DIR="proto"

mkdir -p "$OUT_DIR"
touch "$OUT_DIR/__init__.py"

python -m grpc_tools.protoc \
    -I "$PROTO_DIR" \
    --python_out="$OUT_DIR" \
    --grpc_python_out="$OUT_DIR" \
    "$PROTO_DIR/voice_gateway.proto"

# grpc_tools emits a top-level `import voice_gateway_pb2`, which does not resolve
# when the stubs live inside a package. Rewrite it to a package-absolute import
# so `from app.gen import voice_gateway_pb2_grpc` works.
PB2_GRPC="$OUT_DIR/voice_gateway_pb2_grpc.py"
if [ -f "$PB2_GRPC" ]; then
    python - "$PB2_GRPC" <<'PY'
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
