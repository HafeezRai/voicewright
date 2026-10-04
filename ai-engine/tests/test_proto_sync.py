"""The ai-engine copy of the proto must stay byte-identical to backend-core's.

ai-engine carries its own copy because its Docker build context is ./ai-engine
and cannot reach the sibling backend-core/proto/. Two copies of a contract drift;
this test is what stops them.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
AI_PROTO = REPO_ROOT / "ai-engine" / "proto" / "voice_gateway.proto"
CORE_PROTO = REPO_ROOT / "backend-core" / "proto" / "voice_gateway.proto"


def test_proto_copies_are_identical() -> None:
    if not CORE_PROTO.is_file():
        pytest.skip(f"backend-core proto not found at {CORE_PROTO}")
    assert AI_PROTO.is_file(), "ai-engine must carry its own proto copy"

    ai = AI_PROTO.read_bytes()
    core = CORE_PROTO.read_bytes()
    assert hashlib.sha256(ai).hexdigest() == hashlib.sha256(core).hexdigest(), (
        "ai-engine/proto/voice_gateway.proto has drifted from "
        "backend-core/proto/voice_gateway.proto -- re-copy it and regenerate stubs"
    )


def test_generated_stubs_are_present() -> None:
    gen_dir = REPO_ROOT / "ai-engine" / "app" / "gen"
    assert (gen_dir / "voice_gateway_pb2.py").is_file()
    assert (gen_dir / "voice_gateway_pb2_grpc.py").is_file()


def test_generated_stubs_use_package_absolute_imports() -> None:
    """grpc_tools emits a bare import; the gen script must have rewritten it."""
    grpc_stub = (REPO_ROOT / "ai-engine" / "app" / "gen" / "voice_gateway_pb2_grpc.py").read_text()
    assert "from app.gen import voice_gateway_pb2 as" in grpc_stub
    assert "\nimport voice_gateway_pb2 as" not in grpc_stub


def test_stubs_expose_the_contract() -> None:
    from app.gen import voice_gateway_pb2 as pb

    # The three downstream event payloads Phase 3 fills in.
    assert pb.VoiceEvent(transcript_final=pb.TranscriptFinal(text="hi")).HasField(
        "transcript_final"
    )
    assert pb.VoiceEvent(synthesis_chunk=pb.SynthesisChunk(audio=b"\x00")).HasField(
        "synthesis_chunk"
    )
    assert pb.AudioFrame().session_id == ""
