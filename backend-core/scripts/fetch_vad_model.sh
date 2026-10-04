#!/usr/bin/env bash
# Re-download the Silero VAD ONNX asset that is committed at
# app/audio/models/silero_vad.onnx (kept in-repo so the VAD tests run offline).
#
# Source: snakers4/silero-vad (MIT). The file is byte-identical to the model
# shipped inside the `silero-vad` PyPI wheel (silero_vad/data/silero_vad.onnx).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/app/audio/models/silero_vad.onnx"
URL="https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
EXPECTED_SHA256="1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"

mkdir -p "$(dirname "$DEST")"
curl -sSL --max-time 120 -o "$DEST" "$URL"

actual="$(shasum -a 256 "$DEST" | awk '{print $1}')"
if [ "$actual" != "$EXPECTED_SHA256" ]; then
    echo "sha256 mismatch: got $actual, expected $EXPECTED_SHA256" >&2
    exit 1
fi
echo "ok: $DEST ($(wc -c < "$DEST") bytes)"
