"""FastAPI application entrypoint for ai-engine.

Phase 1 exposed ``GET /health`` only. Phase 3 added the real STT -> NLU -> TTS
pipeline, its Prometheus metrics, and a gRPC server (the ``VoiceGateway``
contract) started alongside HTTP. The intent-classification stage has since been
replaced by a generative LLM, so the middle stage is now ``POST /llm`` and the
pipeline is ``STT -> LLM -> TTS``.

The ``GET /health`` contract is frozen from Phase 1 and is unchanged.

No model is loaded at import or startup: every engine is constructed lazily and
resolves its weights on first use, so the process starts (and answers ``/health``)
in milliseconds and never downloads anything until a request needs it. The LLM
is the one exception, and deliberately so: ``AI_ENGINE_LLM_WARMUP`` warms it on a
*background thread*, so a cold multi-gigabyte model load is paid before the first
caller arrives without ever delaying ``/health``.
"""

from __future__ import annotations

import asyncio
import base64
import functools
import io
import logging
import threading
import time
from collections import OrderedDict
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path

import grpc
import soundfile as sf
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field, ValidationError

# ``request.form()`` hands back *Starlette* UploadFile instances. FastAPI's own
# ``UploadFile`` is a *subclass* of that, so ``isinstance(part, fastapi.UploadFile)``
# is False for a real upload and every multipart request would be rejected -- the
# check must name the class the parser actually produces.
from starlette.datastructures import UploadFile

from app.agents import (
    AgentCreate,
    AgentOut,
    AgentStore,
    AgentUpdate,
    DocumentCreate,
    DocumentOut,
)
from app.agents.store import DEFAULT_ORG_ID
from app.audio_io import to_wav_bytes
from app.config import get_settings
from app.grpc.server import start_grpc_server
from app.knowledge import Retriever, build_embedder, chunk_text
from app.knowledge.embeddings import EmbeddingUnavailableError
from app.llm.history import ConversationHistory
from app.metrics import (
    KNOWLEDGE_CHUNKS_RETURNED,
    KNOWLEDGE_RETRIEVAL_LATENCY,
    KNOWLEDGE_RETRIEVALS,
)
from app.pipeline import VoicePipeline, build_pipeline

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

logger = logging.getLogger(__name__)

#: The single process-wide pipeline. Constructing it loads no weights.
pipeline: VoicePipeline = build_pipeline(settings)

#: The agent/knowledge store. Opening it creates the file and seeds the default
#: org, so the service has a working control plane with no setup step.
agent_store = AgentStore(settings.agent_db_path)

#: The retrieval encoder and the ranker over it. Neither loads any weights at
#: construction -- MiniLM is paged in on the first retrieve (or the first
#: document that is indexed), and never at all for agents with no documents.
embedder = build_embedder(settings)
retriever = Retriever(
    embedder,
    agent_store,
    top_k=settings.knowledge_top_k,
    min_score=settings.knowledge_min_score,
)

#: How many distinct sessions keep history in memory. A demo has a handful; the
#: cap exists so an unauthenticated caller cannot grow the process without bound
#: by sending a fresh ``session_id`` on every request.
_MAX_SESSIONS = 256


class SessionHistoryStore:
    """The smallest thing that can carry conversation history across requests.

    The pipeline is stateless -- it is shared by every session in the process --
    so history has to live where the session does. This is a locked LRU of
    :class:`ConversationHistory`, one per ``session_id``. A request without a
    ``session_id`` gets no history at all, which makes the default behaviour
    identical to a pipeline with no memory.
    """

    def __init__(self, max_turns: int, max_sessions: int = _MAX_SESSIONS):
        self._max_turns = max_turns
        self._max_sessions = max_sessions
        self._lock = threading.Lock()
        self._sessions: OrderedDict[str, ConversationHistory] = OrderedDict()

    @property
    def enabled(self) -> bool:
        return self._max_turns > 0

    def snapshot(self, session_id: str | None) -> list:
        """Prior messages for this session (empty for no/unknown session)."""
        if not session_id or not self.enabled:
            return []
        with self._lock:
            history = self._sessions.get(session_id)
            if history is None:
                return []
            self._sessions.move_to_end(session_id)
            return history.snapshot()

    def record(self, session_id: str | None, user_text: str, assistant_text: str) -> None:
        """Append one exchange, creating the session if needed."""
        if not session_id or not self.enabled:
            return
        if not user_text.strip() or not assistant_text.strip():
            return
        with self._lock:
            history = self._sessions.get(session_id)
            if history is None:
                history = ConversationHistory(self._max_turns)
                self._sessions[session_id] = history
            else:
                self._sessions.move_to_end(session_id)
            while len(self._sessions) > self._max_sessions:
                self._sessions.popitem(last=False)
            history.add_turn(user_text, assistant_text)


_sessions = SessionHistoryStore(settings.llm_history_turns)


def _warm_llm(target: VoicePipeline) -> None:
    """Load the model off the request path. Failures are logged, never raised."""
    try:
        target.llm.load()
    except Exception:
        logger.warning(
            "LLM warm-up failed; the first turn will pay the model-load cost",
            exc_info=True,
        )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Start the gRPC server and (optionally) warm the LLM; shut down cleanly."""
    grpc_server: grpc.aio.Server | None = None
    if settings.grpc_enabled:
        try:
            grpc_server, port = await start_grpc_server(settings, pipeline)
            logger.info("gRPC server bound on port %s", port)
        except Exception:  # pragma: no cover - port clash etc. must not kill HTTP
            logger.exception("gRPC server failed to start; HTTP continues")
            grpc_server = None

    # A background thread, not an await: /health must answer immediately even
    # while 4.66 GB of weights are being paged in.
    if settings.llm_warmup:
        threading.Thread(
            target=_warm_llm, args=(pipeline,), name="llm-warmup", daemon=True
        ).start()

    try:
        yield
    finally:
        pipeline.stt.close()
        pipeline.llm.close()
        pipeline.tts.close()
        if grpc_server is not None:
            await grpc_server.stop(grace=1.0)


app = FastAPI(
    title="CloudPSO Voice Agent — ai-engine",
    version="0.3.0",
    description="STT -> LLM -> TTS pipeline service.",
    lifespan=lifespan,
)

# CORS for the Phase 4 browser dashboard. Additive and off the frozen contract:
# origins come from CLOUDEPSO_CORS_ALLOW_ORIGINS and default to the local
# dashboard only (never "*").
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness/readiness probe. Contract fixed at Phase 1."""
    return {"status": "ok", "service": settings.service_name}


@app.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    """Prometheus exposition endpoint (scraped by infra/prometheus)."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ---------------------------------------------------------------------------
# Pipeline endpoints
#
# These are additive: the frozen surface is /health. They exist so the
# Phase 4 dashboard (and `curl`) can drive each stage independently without
# standing up a WebRTC peer.
# ---------------------------------------------------------------------------


class LLMRequest(BaseModel):
    """A generation request. ``/nlu`` this is not: the reply is free-form prose."""

    text: str = Field(min_length=1, max_length=4000)
    #: Overrides AI_ENGINE_LLM_SYSTEM_PROMPT for this call.
    system_prompt: str | None = Field(default=None, max_length=4000)
    #: Opt in to conversation memory; omitted means a single stateless turn.
    session_id: str | None = Field(default=None, max_length=200)


class TTSRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    speed: float | None = Field(default=None, gt=0.0, le=4.0)


@app.post("/stt")
async def stt(request: Request) -> dict:
    """Transcribe a WAV body. Returns the text and per-stage timing."""
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="empty request body; send WAV bytes")
    try:
        audio, rate = await _read_wav_bytes(body)
        result = await _run_off_loop(pipeline.transcribe, audio, rate)
    except HTTPException:
        raise
    except Exception as exc:
        _raise_unavailable("STT", exc)
    return {
        "text": result.text,
        "language": result.language,
        "duration_s": round(result.duration_s, 3),
        "backend": result.backend,
    }


@app.post("/llm")
async def llm(payload: LLMRequest) -> dict:
    """Generate a reply from a transcript (no audio involved)."""
    history = _sessions.snapshot(payload.session_id)
    try:
        result = await _run_off_loop(
            lambda: pipeline.generate(
                payload.text,
                system_prompt=payload.system_prompt,
                history=history,
            )
        )
    except Exception as exc:
        _raise_unavailable("LLM", exc)
    _sessions.record(payload.session_id, payload.text, result.text)
    return result.as_dict()


@app.post("/tts")
async def tts(payload: TTSRequest) -> Response:
    """Synthesise text and return a WAV body."""
    try:
        result = await _run_off_loop(
            lambda: pipeline.synthesize(payload.text, speed=payload.speed)
        )
    except Exception as exc:
        _raise_unavailable("TTS", exc)
    wav = to_wav_bytes(result.audio, result.sample_rate)
    return Response(
        content=wav,
        media_type="audio/wav",
        headers={
            "X-Sample-Rate": str(result.sample_rate),
            "X-Duration-Seconds": f"{result.duration_s:.3f}",
        },
    )


@app.post("/pipeline")
async def run_pipeline(
    request: Request,
    include_audio: bool = False,
    system_prompt: str | None = None,
    session_id: str | None = None,
    agent_id: str | None = None,
) -> dict:
    """Full STT -> LLM -> TTS turn from a WAV body.

    ``system_prompt`` overrides the configured persona for this call, and
    ``session_id`` opts in to conversation memory. ``agent_id`` selects a stored
    agent: its system prompt, voice, language and greeting supply the defaults,
    and its documents ground the answer. An explicit ``system_prompt`` still
    wins over the agent's -- the request is the more specific instruction.
    """
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="empty request body; send WAV bytes")

    org_id = _resolve_org(request)
    agent = None
    if agent_id:
        agent = agent_store.get_agent(org_id, agent_id)
        if agent is None:
            raise HTTPException(status_code=404, detail=f"agent not found: {agent_id}")

    # Precedence: explicit query param, then the agent's prompt, then the
    # configured default (applied by the pipeline when both are None).
    effective_prompt = system_prompt
    if effective_prompt is None and agent is not None and agent["system_prompt"]:
        effective_prompt = agent["system_prompt"]

    # Retrieval needs the transcript, so it is a callback the pipeline invokes
    # once it has one -- not something the endpoint can do before process_turn.
    knowledge_provider = (
        functools.partial(_knowledge_for, org_id, agent_id) if agent is not None else None
    )

    history: Sequence = _sessions.snapshot(session_id)
    try:
        audio, rate = await _read_wav_bytes(body)
        result = await _run_off_loop(
            lambda: pipeline.process_turn(
                audio,
                rate,
                system_prompt=effective_prompt,
                history=history,
                knowledge_provider=knowledge_provider,
            )
        )
    except HTTPException:
        raise
    except Exception as exc:
        _raise_unavailable("pipeline", exc)

    _sessions.record(session_id, result.transcript.text, result.reply_text)
    payload = {
        "transcript": result.transcript.text,
        "reply_text": result.reply_text,
        "model": result.llm.model,
        "provider": result.llm.provider,
        "usage": result.llm.usage.as_dict() if result.llm.usage else None,
        "agent_id": agent_id,
        "knowledge_chunks": result.knowledge_chunks,
        "audio": {
            "sample_rate": result.speech.sample_rate,
            "duration_s": round(result.speech.duration_s, 3),
            "cancelled": result.speech.cancelled,
        },
        "timings_ms": {k: round(v * 1000, 2) for k, v in result.timings.items()},
    }
    if include_audio:
        payload["audio"]["wav_base64"] = base64.b64encode(
            to_wav_bytes(result.speech.audio, result.speech.sample_rate)
        ).decode("ascii")
    return payload


# ---------------------------------------------------------------------------
# Agent configuration (CRUD)
#
# A tenant-aware store from day one: every row carries an ``org_id`` and every
# query is scoped by one. There is no auth and no org UI yet -- the org comes
# from ``?org_id=`` or an ``X-Org-Id`` header, defaulting to the seeded default
# org. The important property is the *shape*: an agent that belongs to another
# org is reported as 404, never 403, so a caller cannot probe for the existence
# of another tenant's agents by watching the status code.
#
# These handlers are deliberately synchronous ``def``: the store is blocking
# SQLite, so letting Starlette run them in its worker threadpool keeps the event
# loop free. That is also why the store opens a connection per call rather than
# sharing one -- see ``app.agents.store``.
# ---------------------------------------------------------------------------


def _resolve_org(request: Request) -> str:
    """The org a request acts within.

    Precedence: ``?org_id=`` query, then ``X-Org-Id`` header, then the seeded
    default org. Blank values are treated as absent so ``?org_id=`` does not
    resolve to the empty string (which is a valid, if pointless, org id).
    """
    return (
        request.query_params.get("org_id")
        or request.headers.get("x-org-id")
        or DEFAULT_ORG_ID
    )


def _agent_or_404(org_id: str, agent_id: str) -> dict:
    agent = agent_store.get_agent(org_id, agent_id)
    if agent is None:
        # Deliberately the same 404 whether the agent does not exist or belongs
        # to another org: existence must not leak across tenants.
        raise HTTPException(status_code=404, detail=f"agent not found: {agent_id}")
    return agent


@app.post("/agents", response_model=AgentOut, status_code=201)
def create_agent(payload: AgentCreate, request: Request) -> AgentOut:
    """Create an agent in the caller's org."""
    org_id = _resolve_org(request)
    if not agent_store.org_exists(org_id):
        raise HTTPException(status_code=404, detail=f"unknown org: {org_id}")
    return AgentOut(**agent_store.create_agent(org_id, payload.model_dump()))


@app.get("/agents", response_model=list[AgentOut])
def list_agents(request: Request) -> list[AgentOut]:
    """List the caller's org's agents (empty for an unknown org)."""
    return [AgentOut(**row) for row in agent_store.list_agents(_resolve_org(request))]


@app.get("/agents/{agent_id}", response_model=AgentOut)
def get_agent(agent_id: str, request: Request) -> AgentOut:
    """Fetch one agent, scoped to the caller's org."""
    return AgentOut(**_agent_or_404(_resolve_org(request), agent_id))


@app.put("/agents/{agent_id}", response_model=AgentOut)
def update_agent(agent_id: str, payload: AgentUpdate, request: Request) -> AgentOut:
    """Partially update an agent. Unset fields are left unchanged."""
    row = agent_store.update_agent(
        _resolve_org(request), agent_id, payload.model_dump(exclude_unset=True)
    )
    if row is None:
        raise HTTPException(status_code=404, detail=f"agent not found: {agent_id}")
    return AgentOut(**row)


@app.delete("/agents/{agent_id}", status_code=204, response_class=Response)
def delete_agent(agent_id: str, request: Request) -> Response:
    """Delete an agent and, by cascade, its documents and chunks."""
    if not agent_store.delete_agent(_resolve_org(request), agent_id):
        raise HTTPException(status_code=404, detail=f"agent not found: {agent_id}")
    return Response(status_code=204)


# ---------------------------------------------------------------------------
# Knowledge base (documents)
#
# A document is chunked, each chunk embedded with the MiniLM encoder, and the
# vector stored as a BLOB. Ingestion is the one place the embedder *must* be
# available -- there is nothing to store without a vector -- so a missing model
# is a clean 503 here, unlike retrieval, which degrades to an ungrounded turn.
# ---------------------------------------------------------------------------

#: File extensions accepted for upload. ``""`` allows a filename-less part.
_UPLOAD_SUFFIXES = frozenset({".txt", ".md", ".markdown", ""})


@app.post("/agents/{agent_id}/documents", response_model=DocumentOut, status_code=201)
async def add_document(agent_id: str, request: Request) -> DocumentOut:
    """Add a document to an agent, from pasted JSON text or a txt/md upload."""
    org_id = _resolve_org(request)
    _agent_or_404(org_id, agent_id)

    content_type = request.headers.get("content-type", "")
    if content_type.startswith("multipart/form-data"):
        title, text, source = await _read_upload(request)
    else:
        title, text, source = await _read_json_document(request)

    # Embedding and the insert are blocking; keep them off the event loop.
    stored = await asyncio.to_thread(_ingest_document, org_id, agent_id, title, text, source)
    return DocumentOut(**stored)


@app.get("/agents/{agent_id}/documents", response_model=list[DocumentOut])
def list_documents(agent_id: str, request: Request) -> list[DocumentOut]:
    """List an agent's documents (metadata only -- never the chunks)."""
    org_id = _resolve_org(request)
    _agent_or_404(org_id, agent_id)
    return [DocumentOut(**row) for row in agent_store.list_documents(org_id, agent_id)]


@app.delete("/documents/{document_id}", status_code=204, response_class=Response)
def delete_document(document_id: str, request: Request) -> Response:
    """Delete a document and, by cascade, its chunks."""
    if not agent_store.delete_document(_resolve_org(request), document_id):
        raise HTTPException(status_code=404, detail=f"document not found: {document_id}")
    return Response(status_code=204)


async def _read_json_document(request: Request) -> tuple[str, str, str]:
    try:
        data = await request.json()
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"expected a JSON body or a multipart upload: {exc}",
        ) from exc
    try:
        payload = DocumentCreate.model_validate(data)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc
    return payload.title, payload.text, "text"


async def _read_upload(request: Request) -> tuple[str, str, str]:
    form = await request.form()
    upload = form.get("file")
    if not isinstance(upload, UploadFile):
        raise HTTPException(status_code=422, detail="multipart upload requires a 'file' field")
    filename = upload.filename or ""
    suffix = Path(filename).suffix.lower()
    if suffix not in _UPLOAD_SUFFIXES:
        raise HTTPException(
            status_code=422, detail=f"unsupported file type {suffix!r}; upload .txt or .md"
        )
    # Cap the read itself, so a huge upload cannot be materialised in full before
    # the character check below rejects it.
    raw = await upload.read(settings.knowledge_max_document_chars * 4 + 1)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=422, detail="file is not valid UTF-8 text") from exc
    title = str(form.get("title") or filename)
    return title, text, "upload"


def _ingest_document(org_id: str, agent_id: str, title: str, text: str, source: str) -> dict:
    """Chunk, embed and store one document. Blocking; runs on a worker thread."""
    text = text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="document is empty")
    limit = settings.knowledge_max_document_chars
    if len(text) > limit:
        raise HTTPException(
            status_code=413, detail=f"document exceeds the {limit}-character limit"
        )

    pieces = chunk_text(
        text,
        max_chars=settings.knowledge_chunk_chars,
        overlap_chars=settings.knowledge_chunk_overlap_chars,
    )
    if not pieces:
        raise HTTPException(status_code=422, detail="document produced no chunks")

    try:
        vectors = embedder.encode(pieces)
    except EmbeddingUnavailableError as exc:
        _raise_unavailable("embedding", exc)

    chunks = [
        (piece, vectors[index].astype("float32").tobytes(), int(vectors.shape[1]))
        for index, piece in enumerate(pieces)
    ]
    return agent_store.add_document(
        org_id, agent_id, title=title, source=source, text=text, chunks=chunks
    )


def _knowledge_for(org_id: str, agent_id: str, query: str) -> list[str]:
    """Retrieve grounding snippets for an utterance, or ``[]``.

    The one place the "nothing relevant" and "no knowledge base" cases are
    decided, and the one place an unavailable embedder is swallowed. A turn must
    never fail because retrieval could not run: an ungrounded answer is worse
    than a grounded one, but far better than a 500.
    """
    if not settings.knowledge_enabled:
        return []
    # Fast path: no chunks means no embedding, no scoring, nothing to inject.
    # This is what makes an agent without documents as cheap as before step 2.
    if not agent_store.agent_has_documents(org_id, agent_id):
        KNOWLEDGE_RETRIEVALS.labels(status="empty").inc()
        return []

    started = time.perf_counter()
    try:
        chunks = retriever.retrieve(org_id, agent_id, query)
    except EmbeddingUnavailableError as exc:
        KNOWLEDGE_RETRIEVALS.labels(status="error").inc()
        logger.warning("retrieval unavailable; answering ungrounded: %s", exc)
        return []
    except Exception:
        KNOWLEDGE_RETRIEVALS.labels(status="error").inc()
        logger.exception("retrieval failed; answering ungrounded")
        return []
    finally:
        KNOWLEDGE_RETRIEVAL_LATENCY.observe(time.perf_counter() - started)

    KNOWLEDGE_RETRIEVALS.labels(status="ok").inc()
    KNOWLEDGE_CHUNKS_RETURNED.observe(len(chunks))
    if chunks:
        logger.debug("grounding turn with %d chunk(s) for agent=%s", len(chunks), agent_id)
    return retriever.to_snippets(chunks)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


async def _run_off_loop(fn, *args):
    """Run a blocking, CPU-bound pipeline call on a worker thread."""
    return await asyncio.to_thread(fn, *args)


async def _read_wav_bytes(body: bytes):
    """Decode WAV bytes to (float32 mono, sample_rate) without touching disk."""
    try:
        data, rate = sf.read(io.BytesIO(body), dtype="float32", always_2d=False)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"could not decode WAV: {exc}") from exc
    if getattr(data, "ndim", 1) > 1:
        data = data.mean(axis=1)
    return data, int(rate)


def _raise_unavailable(stage: str, exc: Exception) -> None:
    logger.warning("%s unavailable: %s", stage, exc)
    raise HTTPException(status_code=503, detail=f"{stage} backend unavailable: {exc}")


def main() -> None:
    """Run the service with uvicorn (container entrypoint / `python -m app.main`)."""
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
