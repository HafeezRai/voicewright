"""STT -> LLM -> TTS orchestration.

Three paths over the same three stages:

* :meth:`VoicePipeline.process_turn` -- the **batch** path. Audio in, a
  :class:`TurnResult` (transcript, generated reply, synthesised speech) out.
  This is what the HTTP API, the gRPC servicer and the end-to-end test use.

* :meth:`VoicePipeline.stream_turn` -- the **streaming** path. Same stages, but
  the reply is emitted as it is produced and the turn can be cut short.

* :meth:`VoicePipeline.process_stream` -- the **chunked ingress** path: audio
  arrives as a sequence of chunks (as it does over gRPC / WebRTC), is buffered,
  and is then run through the batch path. Buffering is deliberate: Whisper's
  accuracy on a 300 ms fragment is poor, so the honest design is to accumulate a
  full utterance (the gateway's VAD decides where that ends) and transcribe once.

Where the reply comes from
--------------------------
The middle stage used to be a 7-way intent classifier plus a template table.
It is now a generative LLM: ``STT -> llm.generate(system_prompt, text) -> TTS``.
That is what makes a configurable persona, and later a knowledge base, possible
at all -- a new "intent" is now a prompt, not code.

Message assembly lives in one place, :meth:`VoicePipeline._build_messages`,
because that is the single seam step 2 (retrieval) splices retrieved snippets
into.

History is a per-call *parameter*, never pipeline state
-------------------------------------------------------
The pipeline is stateless -- it is shared by every session in the process, and
``to_thread`` runs turns concurrently. Whoever owns a session owns its history
and hands the pipeline a snapshot; :meth:`_build_messages` never mutates it.

Where barge-in cancels
----------------------
Two cancel points, both driven by the sink's :class:`CancelToken`:

* **between generated deltas**, which is the point a turn abandoned *while the
  model was still thinking* is detected -- and it aborts generation itself rather
  than only the playback. ``sink.start()`` is deliberately called *before* the
  LLM call, so a caller who talks over the silence is heard rather than ignored;
* **between audio chunks**, which ends playback at the current chunk. The token
  is also passed into :meth:`TTSEngine.stream`, so a backend that can interrupt
  synthesis (sherpa-onnx does, via its generation callback) aborts the forward
  pass rather than finishing an utterance nobody will hear.

The trap in the first point: :meth:`TTSPlaybackSink.start` re-arms the cancel
token (``token.reset()``), so a ``stop()`` that races in *before* ``start()`` is
silently undone. Every barge-in test therefore fires ``stop()`` after the sink
has started.

STT is not cancellable: it is a short, bounded call, and killing it mid-flight
would leave the session in a worse state than letting it finish.

Streaming shape
---------------
Deltas from :meth:`LLMEngine.stream` are fed to a :class:`SentenceSegmenter`, and
each *completed sentence* is synthesised and emitted while the rest of the reply
is still being generated. That is what gets the first audio out in about a second
instead of only after the whole reply has been produced. ``REPLY_DELTA`` carries
each delta as it arrives, so ``DONE.text`` is always the concatenation of every
``REPLY_DELTA`` yielded.
"""

from __future__ import annotations

import enum
import logging
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field

import numpy as np

from app.audio_io import duration_s
from app.llm.base import (
    ChatMessage,
    LLMEngine,
    LLMResult,
    LLMTimeoutError,
    LLMUnavailableError,
)
from app.llm.prompts import DEFAULT_SYSTEM_PROMPT
from app.metrics import (
    LLM_CANCELLED,
    LLM_FIRST_TOKEN,
    LLM_LATENCY,
    LLM_REQUESTS,
    LLM_TOKENS,
    PIPELINE_LATENCY,
    PIPELINE_REQUESTS,
    STT_AUDIO_SECONDS,
    STT_LATENCY,
    STT_REQUESTS,
    TTS_AUDIO_SECONDS,
    TTS_CANCELLED,
    TTS_LATENCY,
    TTS_REQUESTS,
)
from app.pipeline.segmentation import SentenceSegmenter, iter_sentences
from app.stt.base import STTEngine, TranscriptionResult
from app.tools import Toolbox, ToolCall, ToolResult
from app.tts.base import CancelToken, SynthesisResult, TTSEngine
from app.tts.sink import TTSPlaybackSink

logger = logging.getLogger(__name__)

#: Errors that mean "the provider did not answer", as opposed to a bug in our
#: own code. Only these may be swallowed by ``fallback_reply``.
_LLM_FAILURES = (LLMUnavailableError, LLMTimeoutError)


class _ToolTurnDone(Exception):
    """Internal: a tool-enabled turn already has its reply; skip the stream."""


class _ToolTurnCancelled(Exception):
    """Internal: barge-in landed during a tool-enabled turn's playback."""

#: Header for the retrieved-knowledge block spliced into the system message.
#: The wording carries two instructions that matter for a *voice* agent:
#: answer only from this material (so the model does not invent a policy), and
#: never mention the material itself (a caller should not hear "according to my
#: documents"). "Say you do not know" is the deliberate failure mode -- better a
#: short admission than a confident fabrication, which is the whole reason to
#: ground at all.
_KNOWLEDGE_TEMPLATE = (
    "Reference material follows. Use it to answer the caller only if it is "
    "relevant to their question. If it does not contain the answer, say you do "
    "not know rather than guessing. Answer in your own words and never mention "
    "the reference material, its source, or that you were given any.\n\n"
    "{snippets}"
)

#: Separator between snippets, so the model can tell one policy from the next.
_KNOWLEDGE_SEPARATOR = "\n---\n"

#: Retrieval is not the pipeline's job -- it is stateless and knows nothing of
#: orgs or stores. A caller that wants grounding passes a provider; the pipeline
#: calls it once, with the finished transcript, and splices whatever comes back.
#: This is the whole coupling between the two: a plain function, called at the
#: one point where the utterance text is known and the prompt is not yet built.
KnowledgeProvider = Callable[[str], Sequence[str]]


def format_knowledge(snippets: Sequence[str] | None) -> str:
    """Build the ground-truth block for the system message.

    Returns ``""`` (not a header with no body) when there is nothing relevant,
    so the ungrounded path leaves the system message byte-identical to what it
    was before retrieval existed. That is what makes "no documents" cheap and
    "nothing above threshold" honest: neither adds a word to the prompt.
    """
    items = [s.strip() for s in (snippets or []) if s and s.strip()]
    if not items:
        return ""
    return _KNOWLEDGE_TEMPLATE.format(snippets=_KNOWLEDGE_SEPARATOR.join(items))



class EventKind(enum.StrEnum):
    """Kinds of :class:`PipelineEvent` emitted by :meth:`stream_turn`."""

    TRANSCRIPT_FINAL = "transcript_final"
    #: A piece of the reply text, as it is generated.
    REPLY_DELTA = "reply_delta"
    #: The model asked to run a tool.
    TOOL_CALL = "tool_call"
    #: What that tool returned.
    TOOL_RESULT = "tool_result"
    AUDIO_CHUNK = "audio_chunk"
    DONE = "done"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class PipelineEvent:
    """One step of a streaming turn."""

    kind: EventKind
    #: The transcript, a reply delta, or the full reply on ``DONE``/``CANCELLED``.
    text: str | None = None
    audio: np.ndarray | None = None
    sample_rate: int = 0
    #: Set on ``DONE``/``CANCELLED`` only. A streaming client cannot inspect a
    #: ``TurnResult`` (there is not one), so the two facts the UI needs to
    #: caption a turn -- which model answered, and whether the answer was
    #: grounded -- travel on the terminal event.
    knowledge_chunks: int = 0
    model: str = ""
    #: Set on ``TOOL_CALL`` / ``TOOL_RESULT``.
    tool_call: ToolCall | None = None
    tool_result: ToolResult | None = None


@dataclass(frozen=True)
class Generation:
    """One completed generation, and any tools it used getting there."""

    result: LLMResult
    tool_calls: tuple[ToolCall, ...] = ()
    tool_results: tuple[ToolResult, ...] = ()


@dataclass
class TurnResult:
    """Everything one batch turn produced."""

    transcript: TranscriptionResult
    llm: LLMResult
    reply_text: str
    speech: SynthesisResult
    stt_latency_s: float = 0.0
    llm_latency_s: float = 0.0
    tts_latency_s: float = 0.0
    total_latency_s: float = 0.0
    cancelled: bool = False
    #: The system prompt this turn actually ran with, after the per-call
    #: override was applied. Recorded because "which persona answered?" is the
    #: first question about any turn.
    system_prompt: str = ""
    #: Number of prior messages replayed into the prompt.
    history_turns: int = 0
    #: How many retrieved knowledge chunks were spliced into the system message.
    knowledge_chunks: int = 0
    #: Tools the model called this turn, and what they returned.
    tool_calls: tuple[ToolCall, ...] = ()
    tool_results: tuple[ToolResult, ...] = ()
    #: Per-stage timings, for the dashboard and for latency accounting.
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def audio_duration_s(self) -> float:
        return self.speech.duration_s


class VoicePipeline:
    """Wires the three stages together. Holds no per-turn state."""

    def __init__(
        self,
        stt: STTEngine,
        llm: LLMEngine,
        tts: TTSEngine,
        *,
        sample_rate: int = 16000,
        chunk_samples: int = 3200,
        tool_rounds: int = 3,
        tool_max_tokens: int = 1024,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        fallback_reply: str = "",
    ):
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.sample_rate = sample_rate
        self.chunk_samples = chunk_samples
        self.tool_rounds = tool_rounds
        self.tool_max_tokens = tool_max_tokens
        #: The configured persona. A caller may override it per turn; blank
        #: disables the system message entirely.
        self.system_prompt = system_prompt
        #: Spoken when the provider is unreachable. Empty means fail loudly.
        self.fallback_reply = fallback_reply

    # -- single stages (public, metric-recording) ---------------------------
    #
    # The HTTP API exposes /stt, /llm and /tts individually. They go through
    # these rather than the engines directly so that every path into a stage
    # records the same metrics -- otherwise calling /llm would be invisible in
    # Prometheus while the identical generation inside a turn was counted.
    def transcribe(self, audio: np.ndarray | str, sample_rate: int | None = None):
        """Transcribe one utterance, recording STT metrics."""
        t0 = time.perf_counter()
        try:
            transcript = self.stt.transcribe(audio, sample_rate)
        except Exception:
            STT_REQUESTS.labels(status="error").inc()
            raise
        STT_REQUESTS.labels(status="ok").inc()
        STT_LATENCY.observe(time.perf_counter() - t0)
        if transcript.duration_s:
            STT_AUDIO_SECONDS.observe(transcript.duration_s)
        return transcript

    def generate(
        self,
        text: str,
        *,
        system_prompt: str | None = None,
        history: Sequence[ChatMessage] | None = None,
        knowledge: Sequence[str] | None = None,
        toolbox: Toolbox | None = None,
        llm: LLMEngine | None = None,
    ) -> Generation:
        """Generate one reply, recording LLM metrics."""
        return self._generate(
            text,
            system_prompt=system_prompt,
            history=history,
            knowledge=knowledge,
            toolbox=toolbox,
            llm=llm,
        )

    def synthesize(self, text: str, *, speed: float | None = None) -> SynthesisResult:
        """Synthesise one utterance, recording TTS metrics."""
        return self._synthesize(text, speed=speed)

    # -- batch path ---------------------------------------------------------
    def process_turn(
        self,
        audio: np.ndarray | str,
        sample_rate: int | None = None,
        *,
        system_prompt: str | None = None,
        history: Sequence[ChatMessage] | None = None,
        knowledge_provider: KnowledgeProvider | None = None,
        toolbox: Toolbox | None = None,
        llm: LLMEngine | None = None,
    ) -> TurnResult:
        """Run one full STT -> LLM -> TTS turn and return the result.

        ``knowledge_provider`` is called once, with the transcript, and its
        snippets are spliced into the system message. Retrieval needs the
        *transcript*, which only exists after STT, so the provider is invoked
        here rather than owned here -- the pipeline stays ignorant of stores.
        """
        started = time.perf_counter()

        t0 = time.perf_counter()
        try:
            transcript = self.transcribe(audio, sample_rate)
        except Exception:
            PIPELINE_REQUESTS.labels(status="error", llm_provider="?").inc()
            raise
        stt_s = time.perf_counter() - t0

        knowledge = knowledge_provider(transcript.text) if knowledge_provider else None
        return self._turn_from_transcript(
            transcript,
            stt_s,
            started,
            system_prompt,
            history,
            knowledge,
            toolbox=toolbox,
            llm=llm,
        )

    def process_stream(
        self,
        chunks: Iterable[np.ndarray],
        sample_rate: int | None = None,
        *,
        system_prompt: str | None = None,
        history: Sequence[ChatMessage] | None = None,
        knowledge_provider: KnowledgeProvider | None = None,
    ) -> TurnResult:
        """Buffer an audio chunk stream into one utterance, then run it.

        See the module docstring: transcription happens once, on the whole
        buffered utterance, not per chunk.
        """
        rate = sample_rate or self.sample_rate
        parts = [np.asarray(c, dtype=np.float32) for c in chunks if c is not None and len(c)]
        audio = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
        logger.debug("ingested %.2fs of audio from %d chunks",
                     duration_s(audio, rate), len(parts))
        return self.process_turn(
            audio,
            rate,
            system_prompt=system_prompt,
            history=history,
            knowledge_provider=knowledge_provider,
        )

    # -- streaming path -----------------------------------------------------
    def stream_turn(
        self,
        audio: np.ndarray | str,
        sample_rate: int | None = None,
        *,
        sink: TTSPlaybackSink | None = None,
        system_prompt: str | None = None,
        history: Sequence[ChatMessage] | None = None,
        knowledge_provider: KnowledgeProvider | None = None,
        toolbox: Toolbox | None = None,
        llm: LLMEngine | None = None,
    ) -> Iterator[PipelineEvent]:
        """Yield the turn as it happens, cutting it off on barge-in.

        The reply is generated and spoken *concurrently*: each delta is fed to a
        :class:`SentenceSegmenter`, and every sentence it completes is synthesised
        and emitted while the rest of the reply is still being generated. See the
        module docstring for the two cancel points.
        """
        started = time.perf_counter()
        t0 = time.perf_counter()
        transcript = self.transcribe(audio, sample_rate)
        stt_s = time.perf_counter() - t0
        yield PipelineEvent(EventKind.TRANSCRIPT_FINAL, text=transcript.text)

        sink = sink or TTSPlaybackSink()
        # Started *before* generation, so a caller who talks over the silence
        # while the model is thinking is heard rather than ignored. Note this
        # re-arms the token: a stop() before this point would be silently undone.
        sink.start()
        token: CancelToken = sink.cancel_token
        provider = self._provider()
        knowledge = knowledge_provider(transcript.text) if knowledge_provider else None
        messages = self._build_messages(transcript.text, system_prompt, history, knowledge)
        logger.debug("stream turn: stt=%.3fs provider=%s", stt_s, provider)

        segmenter = SentenceSegmenter()
        #: Every delta yielded as REPLY_DELTA, so DONE.text is exactly their join.
        parts: list[str] = []
        cancelled = False
        fallback = False
        first_token_s: float | None = None
        first_chunk_s: float | None = None
        produced = 0
        llm_t0 = time.perf_counter()

        def speak(sentences: Iterable[str]) -> Iterator[PipelineEvent]:
            """Synthesise each sentence, stopping the moment barge-in lands.

            This is cancel point B: the check between audio chunks. When it
            trips, ``cancelled`` is set and the caller's outer loop unwinds.
            """
            nonlocal cancelled, first_chunk_s, produced
            for sentence in sentences:
                t_tts = time.perf_counter()
                for chunk in self._speak(sentence, sink, token):
                    if first_chunk_s is None:
                        first_chunk_s = time.perf_counter() - started
                    produced += len(chunk)
                    yield PipelineEvent(
                        EventKind.AUDIO_CHUNK,
                        audio=chunk,
                        sample_rate=self.tts.sample_rate,
                    )
                TTS_LATENCY.observe(time.perf_counter() - t_tts)
                # ---- BARGE-IN CANCEL POINT B: between audio chunks ---------
                if token.cancelled or not sink.is_speaking:
                    cancelled = True
                    return
                # ------------------------------------------------------------

        try:
            try:
                if toolbox:
                    # A tool-enabled turn cannot stream: the model has to finish
                    # deciding, the tool has to run, and only then is there
                    # anything worth saying. The whole reply is generated first
                    # and then spoken a sentence at a time, so the *audio* still
                    # starts early even though the text did not.
                    yield from self._speak_tool_turn(
                        messages, toolbox, llm, sink, token, speak, parts
                    )
                    if cancelled:
                        raise _ToolTurnCancelled
                    raise _ToolTurnDone

                for delta in self.llm.stream(messages, cancel=token):
                    # ---- BARGE-IN CANCEL POINT A: between deltas ------------
                    # The turn was abandoned while the model was still thinking;
                    # this aborts generation, not just playback.
                    if token.cancelled or not sink.is_speaking:
                        cancelled = True
                        break
                    # ---------------------------------------------------------
                    if first_token_s is None:
                        first_token_s = time.perf_counter() - llm_t0
                        LLM_FIRST_TOKEN.labels(provider=provider).observe(first_token_s)
                    parts.append(delta)
                    yield PipelineEvent(EventKind.REPLY_DELTA, text=delta)
                    yield from speak(segmenter.feed(delta))
                    if cancelled:
                        break
                else:
                    # The deltas ran out with a fragment still buffered (a reply
                    # that stopped mid-sentence still gets spoken).
                    yield from speak(segmenter.flush())
            except (_ToolTurnDone, _ToolTurnCancelled):
                # A tool-enabled turn already produced (and spoke) its reply;
                # the delta loop below never ran, so there is nothing to do.
                pass
            except _LLM_FAILURES as exc:
                if not self.fallback_reply:
                    LLM_REQUESTS.labels(provider=provider, status="error").inc()
                    raise
                logger.warning(
                    "LLM unavailable (%s); speaking the configured fallback reply", exc
                )
                fallback = True
                parts = [self.fallback_reply]
                yield PipelineEvent(EventKind.REPLY_DELTA, text=self.fallback_reply)
                if not token.cancelled and sink.is_speaking:
                    yield from speak(iter_sentences(self.fallback_reply))
                else:
                    cancelled = True

            # The delta stream can also end early *because* of the cancel -- and
            # if no delta ever arrived, the in-loop check never ran at all. Catch
            # both here, before the finally below marks the sink finished, or an
            # abandoned turn would be reported as a completed one.
            if not cancelled and (token.cancelled or not sink.is_speaking):
                cancelled = True
        finally:
            LLM_LATENCY.labels(provider=provider).observe(time.perf_counter() - llm_t0)
            if produced and self.tts.sample_rate:
                TTS_AUDIO_SECONDS.observe(produced / float(self.tts.sample_rate))
            if sink.is_speaking:
                sink.finish()

        if first_chunk_s is not None:
            logger.debug("tts first chunk in %.3fs of the turn", first_chunk_s)

        # Caption material for whichever terminal event this turn produces.
        grounded_chunks = len([s for s in (knowledge or []) if s and s.strip()])
        model_name = getattr(self.llm, "model", "") or ""

        if cancelled:
            LLM_CANCELLED.labels(provider=provider).inc()
            LLM_REQUESTS.labels(provider=provider, status="cancelled").inc()
            TTS_CANCELLED.inc()
            TTS_REQUESTS.labels(status="cancelled").inc()
            yield PipelineEvent(
                EventKind.CANCELLED,
                text="".join(parts),
                knowledge_chunks=grounded_chunks,
                model=model_name,
            )
            return

        LLM_REQUESTS.labels(
            provider=provider, status="fallback" if fallback else "ok"
        ).inc()
        TTS_REQUESTS.labels(status="ok").inc()
        PIPELINE_REQUESTS.labels(status="ok", llm_provider=provider).inc()
        PIPELINE_LATENCY.observe(time.perf_counter() - started)
        yield PipelineEvent(
            EventKind.DONE,
            text="".join(parts),
            knowledge_chunks=grounded_chunks,
            model=model_name,
        )

    # -- internals ----------------------------------------------------------
    def _provider(self) -> str:
        """Label for the metrics: the provider, falling back to the engine name."""
        return self.llm.provider or self.llm.name

    def _system_prompt_for(self, override: str | None) -> str:
        return self.system_prompt if override is None else override

    def _build_messages(
        self,
        text: str,
        system_prompt: str | None,
        history: Sequence[ChatMessage] | None,
        knowledge: Sequence[str] | None = None,
    ) -> list[ChatMessage]:
        """Assemble the prompt: system, then replayed turns, then the utterance.

        The single seam step 2's retrieval splices into. Retrieved snippets are
        appended to the *existing* system message rather than sent as a second
        one: a model treats one system turn more reliably than two, and the
        persona and its grounding belong together. When there is no knowledge
        (the common case) the system message is byte-identical to before.
        """
        messages: list[ChatMessage] = []
        prompt = self._system_prompt_for(system_prompt)
        block = format_knowledge(knowledge)
        if block:
            prompt = f"{prompt}\n\n{block}" if prompt else block
        if prompt:
            messages.append(ChatMessage(role="system", content=prompt))
        if history:
            messages.extend(history)
        messages.append(ChatMessage(role="user", content=text))
        return messages

    def _complete_with_tools(
        self,
        messages: list[ChatMessage],
        toolbox: Toolbox | None,
        llm: LLMEngine | None = None,
    ) -> tuple[LLMResult, tuple[ToolCall, ...], tuple[ToolResult, ...]]:
        """Generate, running whatever tools the model asks for, until it answers.

        Bounded by ``tool_rounds``. A small model can ask for a tool, be given
        the result, and ask again -- and a voice agent that thinks forever is
        worse than one that gives up and says so. On the last round the tools
        are withdrawn, so the model has to produce speech rather than another
        call.

        The token budget is the tool budget, not the reply budget. A *thinking*
        model spends its allowance on hidden reasoning before it emits a call:
        measured on qwen3.5:0.8b, a 160-token budget produced a tool call once
        in five and otherwise returned ``finish_reason=length``. At 512 it
        managed four in five.
        """
        engine = llm or self.llm
        if not toolbox:
            return engine.generate(messages), (), ()

        calls: list[ToolCall] = []
        results: list[ToolResult] = []
        for round_index in range(self.tool_rounds):
            last_round = round_index == self.tool_rounds - 1
            result = engine.generate(
                messages,
                tools=None if last_round else toolbox.specs,
                max_tokens=self.tool_max_tokens,
            )
            if not result.tool_calls:
                return result, tuple(calls), tuple(results)

            messages.append(
                ChatMessage("assistant", result.text, tool_calls=result.tool_calls)
            )
            for call in result.tool_calls:
                outcome = toolbox.run(call)
                calls.append(call)
                results.append(outcome)
                messages.append(
                    ChatMessage("tool", outcome.content, tool_call_id=call.id)
                )
                logger.info("tool %s(%s) -> %s", call.name, call.arguments, outcome.content[:80])

        # Unreachable: the last round passes no tools, so it cannot return calls.
        return engine.generate(messages, max_tokens=self.tool_max_tokens), tuple(calls), tuple(results)

    def _generate(
        self,
        text: str,
        *,
        system_prompt: str | None,
        history: Sequence[ChatMessage] | None,
        knowledge: Sequence[str] | None = None,
        toolbox: Toolbox | None = None,
        llm: LLMEngine | None = None,
    ) -> Generation:
        provider = (llm.provider if llm is not None else None) or self._provider()
        messages = self._build_messages(text, system_prompt, history, knowledge)
        started = time.perf_counter()
        try:
            result, calls, tool_results = self._complete_with_tools(messages, toolbox, llm)
        except _LLM_FAILURES as exc:
            LLM_REQUESTS.labels(provider=provider, status="error").inc()
            if not self.fallback_reply:
                raise
            logger.warning(
                "LLM unavailable (%s); speaking the configured fallback reply", exc
            )
            LLM_LATENCY.labels(provider=provider).observe(time.perf_counter() - started)
            LLM_REQUESTS.labels(provider=provider, status="fallback").inc()
            fallback = LLMResult(
                text=self.fallback_reply,
                model=self.llm.model,
                provider=provider,
                finish_reason="fallback",
                fallback=True,
                latency_s=time.perf_counter() - started,
            )
            return Generation(result=fallback)

        if not result.text.strip():
            # A model that used a tool and then said nothing leaves the caller in
            # silence -- the worst outcome in a voice loop, and one no amount of
            # UI can paper over. Common with small models, which sometimes answer
            # a tool result with an empty string.
            logger.warning(
                "model %s returned an empty reply%s",
                self.llm.model,
                f" after {len(calls)} tool call(s)" if calls else "",
            )
            if self.fallback_reply:
                result = LLMResult(
                    text=self.fallback_reply,
                    model=result.model,
                    provider=result.provider,
                    finish_reason="fallback",
                    fallback=True,
                    latency_s=result.latency_s,
                )

        LLM_REQUESTS.labels(provider=provider, status="ok").inc()
        LLM_LATENCY.labels(provider=provider).observe(time.perf_counter() - started)
        if result.first_token_s is not None:
            LLM_FIRST_TOKEN.labels(provider=provider).observe(result.first_token_s)
        if result.usage is not None:
            if result.usage.prompt_tokens:
                LLM_TOKENS.labels(provider=provider, kind="prompt").inc(
                    result.usage.prompt_tokens
                )
            if result.usage.completion_tokens:
                LLM_TOKENS.labels(provider=provider, kind="completion").inc(
                    result.usage.completion_tokens
                )
        return Generation(result=result, tool_calls=calls, tool_results=tool_results)

    def _speak_tool_turn(
        self,
        messages: list[ChatMessage],
        toolbox: Toolbox,
        llm: LLMEngine | None,
        sink: TTSPlaybackSink,
        token: CancelToken,
        speak: Callable[[Iterable[str]], Iterator[PipelineEvent]],
        parts: list[str],
    ) -> Iterator[PipelineEvent]:
        """Run the tool loop, publish what it did, and speak the answer.

        The tool events are emitted *before* any audio. A caller only hears the
        answer, but anyone watching the console sees which tools ran to produce
        it, in order -- and that ordering is the difference between an agent
        that claims to have booked something and one you can watch book it.
        """
        result, calls, outcomes = self._complete_with_tools(messages, toolbox, llm)
        for call, outcome in zip(calls, outcomes, strict=True):
            yield PipelineEvent(EventKind.TOOL_CALL, tool_call=call)
            yield PipelineEvent(EventKind.TOOL_RESULT, tool_result=outcome)

        text = result.text
        if not text:
            return
        parts.append(text)
        # One delta carrying the whole reply, so DONE.text is still exactly the
        # join of every REPLY_DELTA a client received.
        yield PipelineEvent(EventKind.REPLY_DELTA, text=text)
        if not token.cancelled and sink.is_speaking:
            yield from speak(iter_sentences(text))

    def _speak(
        self,
        sentence: str,
        sink: TTSPlaybackSink,
        token: CancelToken,
    ) -> Iterator[np.ndarray]:
        """Yield one sentence's audio chunks, stopping the moment barge-in lands.

        The token is handed to :meth:`TTSEngine.stream` (a backend that can
        interrupt synthesis aborts the forward pass), and re-checked here so a
        backend that cannot still stops between chunks.
        """
        for chunk in self.tts.stream(
            sentence, chunk_samples=self.chunk_samples, cancel=token
        ):
            if token.cancelled or not sink.is_speaking:
                return
            yield chunk

    def _synthesize(self, text: str, *, speed: float | None = None) -> SynthesisResult:
        t2 = time.perf_counter()
        try:
            speech = self.tts.synthesize(text, speed=speed)
        except Exception:
            TTS_REQUESTS.labels(status="error").inc()
            raise
        TTS_LATENCY.observe(time.perf_counter() - t2)
        TTS_REQUESTS.labels(status="ok").inc()
        if speech.duration_s:
            TTS_AUDIO_SECONDS.observe(speech.duration_s)
        return speech

    def _turn_from_transcript(
        self,
        transcript: TranscriptionResult,
        stt_s: float,
        started: float,
        system_prompt: str | None,
        history: Sequence[ChatMessage] | None,
        knowledge: Sequence[str] | None = None,
        toolbox: Toolbox | None = None,
        llm: LLMEngine | None = None,
    ) -> TurnResult:
        t1 = time.perf_counter()
        generation = self._generate(
            transcript.text,
            system_prompt=system_prompt,
            history=history,
            knowledge=knowledge,
            toolbox=toolbox,
            llm=llm,
        )
        llm_result = generation.result
        llm_s = time.perf_counter() - t1

        t2 = time.perf_counter()
        speech = self._synthesize(llm_result.text)
        tts_s = time.perf_counter() - t2
        total = time.perf_counter() - started

        PIPELINE_REQUESTS.labels(status="ok", llm_provider=self._provider()).inc()
        PIPELINE_LATENCY.observe(total)
        return TurnResult(
            transcript=transcript,
            llm=llm_result,
            tool_calls=generation.tool_calls,
            tool_results=generation.tool_results,
            reply_text=llm_result.text,
            speech=speech,
            stt_latency_s=stt_s,
            llm_latency_s=llm_s,
            tts_latency_s=tts_s,
            total_latency_s=total,
            system_prompt=self._system_prompt_for(system_prompt),
            history_turns=len(history) if history else 0,
            knowledge_chunks=len([s for s in (knowledge or []) if s and s.strip()]),
            timings={"stt": stt_s, "llm": llm_s, "tts": tts_s, "total": total},
        )
