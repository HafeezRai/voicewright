#!/usr/bin/env bash
# Download the model weights ai-engine needs into ./models (gitignored).
#
# Mirrors backend-core/scripts/fetch_vad_model.sh: weights are NOT committed --
# they are hundreds of MB to ~1 GB and are fetched into the /models volume (or
# ./models on a dev host) rather than the git tree.
#
#   STT  faster-whisper base.en        ~150 MB  (MIT; OpenAI Whisper weights)
#   TTS  Piper en_US-amy-low (VITS)     ~80 MB  (MIT voice, Apache-2.0 runtime)
#   NLU  Laya English checkpoint       ~840 MB  (Apache-2.0) -- OPTIONAL
#
# Laya is skipped by default: the HuggingFace fallback NLU backend works without
# it, and 840 MB is a lot to pull for a component that is swappable. Pass
# --with-laya to fetch it (AI_ENGINE_NLU_BACKEND=auto will then select Laya).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODELS_DIR="${AI_ENGINE_MODEL_CACHE:-$ROOT/models}"
STT_MODEL="${AI_ENGINE_STT_MODEL:-base.en}"
TTS_MODEL="${AI_ENGINE_TTS_MODEL:-vits-piper-en_US-amy-low}"
WITH_LAYA=false
for arg in "$@"; do
    [ "$arg" = "--with-laya" ] && WITH_LAYA=true
done

# All HuggingFace downloads live under <models>/hf/hub so the layout matches the
# running service (see app/config.py: hf_home / hf_hub_cache).
export HF_HOME="$MODELS_DIR/hf"
mkdir -p "$HF_HOME/hub" "$MODELS_DIR/tts"

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
    PYTHON="$(command -v python3)"
fi

echo "==> STT: $STT_MODEL"
"$PYTHON" - "$STT_MODEL" "$HF_HOME/hub" <<'PY'
import sys
from faster_whisper import WhisperModel
name, download_root = sys.argv[1], sys.argv[2]
WhisperModel(name, device="cpu", compute_type="int8", download_root=download_root)
print(f"    ok: {name}")
PY

echo "==> TTS: $TTS_MODEL"
TTS_DIR="$MODELS_DIR/tts/$TTS_MODEL"
if [ -f "$TTS_DIR/model.onnx" ] || ls "$TTS_DIR"/*.onnx >/dev/null 2>&1; then
    echo "    already present: $TTS_DIR"
else
    URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/${TTS_MODEL}.tar.bz2"
    TMP="$(mktemp -d)"
    echo "    downloading $URL"
    curl -fsSL "$URL" -o "$TMP/voice.tar.bz2"
    tar xjf "$TMP/voice.tar.bz2" -C "$MODELS_DIR/tts"
    rm -rf "$TMP"
    echo "    ok: $TTS_DIR"
fi

if [ "$WITH_LAYA" = true ]; then
    echo "==> NLU: Laya (this is ~840 MB and takes a while)"
    "$PYTHON" - <<'PY'
import laya
laya.load("convaiinnovations/laya")
print("    ok: convaiinnovations/laya")
PY
else
    echo "==> NLU: Laya skipped (pass --with-laya to fetch it)."
    echo "    The pipeline will use the HuggingFace fallback (~90 MB, downloaded on first use)."
fi

echo "==> done. Models under $MODELS_DIR"
