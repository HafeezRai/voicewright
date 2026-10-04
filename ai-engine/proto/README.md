# proto/

`voice_gateway.proto` is a **byte-identical copy** of
`backend-core/proto/voice_gateway.proto`. It is the single shared contract
between the Phase 2 gateway and this service, and it is duplicated here only
because ai-engine's Docker build context is `./ai-engine` and therefore cannot
reach the sibling directory.

Two copies of a wire contract drift. Two things prevent it:

* `tests/test_proto_sync.py` compares the two files by SHA-256 and fails on any
  difference;
* `scripts/gen_proto.sh` regenerates the committed stubs from *this* copy.

To change the contract: edit `backend-core/proto/voice_gateway.proto`, copy it
here verbatim, then run `make proto` (in both services).
