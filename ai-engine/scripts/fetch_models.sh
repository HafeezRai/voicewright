#!/usr/bin/env bash
# Download the model weights ai-engine needs into ./models (gitignored).
#
# Mirrors backend-core/scripts/fetch_vad_model.sh: weights are NOT committed --
# they are hundreds of MB to ~1 GB and are fetched into the /models volume (or
# ./models on a dev host) rather than the git tree.
#
#   STT  faster-whisper base.en        ~150 MB  (MIT; OpenAI Whisper weights)
#   TTS  Piper en_US-amy-low (VITS)     ~80 MB  (MIT voice, Apache-2.0 runtime)
#   LLM  llama3:8b via Ollama          ~4.7 GB  (fetched by Ollama, not here)
#
# The LLM is not a weight file this script downloads: it is served by Ollama,
# which manages its own model store. If `ollama` is on PATH the configured
# AI_ENGINE_LLM_MODEL is pulled here; otherwise the step is skipped with a note
# rather than failing the whole fetch.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODELS_DIR="${AI_ENGINE_MODEL_CACHE:-$ROOT/models}"
STT_MODEL="${AI_ENGINE_STT_MODEL:-base.en}"
TTS_MODEL="${AI_ENGINE_TTS_MODEL:-vits-piper-en_US-amy-low}"
LLM_MODEL="${AI_ENGINE_LLM_MODEL:-llama3:8b}"

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

echo "==> LLM: $LLM_MODEL (served by Ollama)"
if command -v ollama >/dev/null 2>&1; then
    ollama pull "$LLM_MODEL"
else
    echo "    ollama not found on PATH; skipping the LLM pull."
    echo "    Install Ollama and run:  ollama pull $LLM_MODEL"
fi

echo "==> done. Models under $MODELS_DIR"
