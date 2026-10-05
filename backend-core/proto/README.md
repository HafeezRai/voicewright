# backend-core protobuf definitions

Placeholder for the gRPC contract between `backend-core` and `ai-engine`.

Phase 1 contains only `voice_gateway.proto` — an interface sketch that is **not**
compiled or imported by the application. `app/main.py` does not import generated
stubs, so nothing here can break imports.

Phase 3 owns the real STT → LLM → TTS service contract; Phase 2 owns the
gateway-side streaming RPCs. Regenerate stubs with `grpcio-tools` once the
contract is frozen:

```bash
python -m grpc_tools.protoc -I proto \
  --python_out=app/gen --grpc_python_out=app/gen proto/voice_gateway.proto
```
