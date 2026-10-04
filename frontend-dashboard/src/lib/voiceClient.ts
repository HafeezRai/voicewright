/**
 * The signalling client: microphone -> RTCPeerConnection -> backend-core's
 * `/ws/signaling` WebSocket.
 *
 * It speaks the exact wire protocol implemented in
 * `backend-core/app/ws/signaling.py` — nothing invented here:
 *
 *   client -> server
 *     {"type":"offer","sdp":...}                   // candidates inline (vanilla ICE)
 *     {"type":"playback","state":"start"|"stop"}   // drives the Phase-2 stub sink
 *     {"type":"ping"} / {"type":"close"}
 *
 * The protocol also defines a client -> server `{"type":"ice-candidate", ...}`
 * message; this client does not send it, and `attachPeerHandlers` explains why.
 *
 *   server -> client
 *     {"type":"answer","sdp":...,"session_id":...}
 *     {"type":"ice-candidate","candidate":...}
 *     {"type":"state","state":"connected"|...}
 *     {"type":"barge-in","duration_ms":N}
 *     {"type":"pong"} / {"type":"error","message":...}
 *
 * Everything the client touches that a test cannot create — getUserMedia, the
 * peer connection, the WebSocket — is injected via `VoiceClientDeps`, so the
 * unit tests drive the real logic against fakes and no browser is required.
 *
 * Media note: backend-core (Phase 2) adds no outbound track, so normally **no
 * remote audio arrives** — the client still wires `ontrack` so that if a
 * server-side track does appear it plays. The agent's audible voice in this
 * build comes from ai-engine's `/tts` (see the Agent turn panel).
 */

import type { IceState, VoiceEvent } from "../state/voiceMachine";
import { config } from "../config";

export interface VoiceClientDeps {
  getUserMedia(constraints: MediaStreamConstraints): Promise<MediaStream>;
  createPeerConnection(config: RTCConfiguration): RTCPeerConnection;
  createWebSocket(url: string): WebSocket;
}

export function defaultDeps(): VoiceClientDeps {
  return {
    getUserMedia: (constraints) => navigator.mediaDevices.getUserMedia(constraints),
    createPeerConnection: (rtcConfig) => new RTCPeerConnection(rtcConfig),
    createWebSocket: (url) => new WebSocket(url),
  };
}

export interface VoiceClientOptions {
  signalingUrl?: string;
  iceServers?: RTCIceServer[];
  deps?: VoiceClientDeps;
  /** Called for every event the client wants folded into the state machine. */
  dispatch(event: VoiceEvent): void;
  /** Called when a remote (agent) audio track arrives. */
  onRemoteStream?(stream: MediaStream): void;
  /** Called with the raw parsed message, for logging/telemetry. */
  onMessage?(message: Record<string, unknown>): void;
}

const ICE_MAP: Record<string, IceState> = {
  new: "new",
  checking: "checking",
  connected: "connected",
  completed: "completed",
  disconnected: "disconnected",
  failed: "failed",
  closed: "closed",
};

export class VoiceClient {
  private readonly deps: VoiceClientDeps;
  private readonly signalingUrl: string;
  private readonly iceServers: RTCIceServer[];
  private readonly dispatch: (event: VoiceEvent) => void;
  private readonly onRemoteStream?: (stream: MediaStream) => void;
  private readonly onMessage?: (message: Record<string, unknown>) => void;

  private pc: RTCPeerConnection | null = null;
  private ws: WebSocket | null = null;
  private localStream: MediaStream | null = null;
  private remoteStream: MediaStream | null = null;
  private active = false;

  constructor(options: VoiceClientOptions) {
    this.deps = options.deps ?? defaultDeps();
    this.signalingUrl = options.signalingUrl ?? config.signalingUrl;
    this.iceServers = options.iceServers ?? [];
    this.dispatch = options.dispatch;
    this.onRemoteStream = options.onRemoteStream;
    this.onMessage = options.onMessage;
  }

  /** Current microphone stream (exposed so teardown can be asserted in tests). */
  get microphone(): MediaStream | null {
    return this.localStream;
  }

  get peerConnection(): RTCPeerConnection | null {
    return this.pc;
  }

  async start(): Promise<void> {
    if (this.active) return;
    this.active = true;

    // 1. Microphone first: a permission failure should fail before any socket.
    let stream: MediaStream;
    try {
      stream = await this.deps.getUserMedia({ audio: true, video: false });
    } catch (error) {
      this.active = false;
      this.dispatch({ type: "mic-denied", error: describeError(error) });
      return;
    }
    this.localStream = stream;
    this.dispatch({ type: "mic-granted" });

    // 2. Peer connection with echo cancellation on (barge-in relies on it).
    const pc = this.deps.createPeerConnection({ iceServers: this.iceServers });
    this.pc = pc;
    this.attachPeerHandlers(pc);

    for (const track of stream.getAudioTracks()) {
      pc.addTrack(track, stream);
    }

    // 3. Signalling socket; the offer goes out once it is open.
    const ws = this.deps.createWebSocket(this.signalingUrl);
    this.ws = ws;
    this.attachSocketHandlers(ws, pc);
  }

  private attachPeerHandlers(pc: RTCPeerConnection): void {
    // NOTE — vanilla ICE, deliberately.
    //
    // We do *not* trickle candidates to backend-core. Its
    // `WebRTCSession.add_ice_candidate` constructs `RTCIceCandidate(candidate=…)`,
    // which aiortc >= 1.10 rejects with `TypeError: unexpected keyword argument
    // 'candidate'`; the exception escapes the signalling handler and the session
    // is torn down. That is a Phase 2 defect, out of scope for this phase, so the
    // client instead gathers fully and carries its candidates in the offer SDP.
    // Once the server is fixed, switching back to trickle is a two-line change
    // (re-add an `onicecandidate` send here and drop the gather wait below).
    pc.onconnectionstatechange = () => {
      this.dispatch({ type: "ice", state: ICE_MAP[pc.connectionState] ?? "none" });
    };
    pc.oniceconnectionstatechange = () => {
      const mapped = ICE_MAP[pc.iceConnectionState];
      if (mapped) this.dispatch({ type: "ice", state: mapped });
    };
    pc.ontrack = (event) => {
      const [stream] = event.streams;
      this.remoteStream = stream ?? new MediaStream([event.track]);
      if (this.onRemoteStream) this.onRemoteStream(this.remoteStream);
    };
  }

  private attachSocketHandlers(ws: WebSocket, pc: RTCPeerConnection): void {
    ws.onopen = () => {
      this.dispatch({ type: "signaling-open" });
      void this.sendOffer(pc);
    };
    ws.onmessage = (event) => {
      void this.handleMessage(event.data, pc);
    };
    ws.onerror = () => {
      this.dispatch({ type: "error", message: "signalling socket error" });
    };
    ws.onclose = () => {
      if (this.active) {
        // Remote-initiated close (or a transport drop) ends the call.
        this.active = false;
        this.dispatch({ type: "closed" });
      }
    };
  }

  private async sendOffer(pc: RTCPeerConnection): Promise<void> {
    try {
      const offer = await pc.createOffer({ offerToReceiveAudio: true });
      await pc.setLocalDescription(offer);
      // Vanilla ICE: wait for gathering so the offer SDP already carries our
      // candidates (see the note in attachPeerHandlers).
      await waitForIceGathering(pc);
      const sdp = pc.localDescription?.sdp ?? offer.sdp;
      this.send({ type: "offer", sdp });
    } catch (error) {
      this.dispatch({ type: "error", message: `offer failed: ${describeError(error)}` });
    }
  }

  private async handleMessage(raw: unknown, pc: RTCPeerConnection): Promise<void> {
    let message: Record<string, unknown>;
    try {
      message = JSON.parse(String(raw)) as Record<string, unknown>;
    } catch {
      return;
    }
    if (this.onMessage) this.onMessage(message);

    switch (message.type) {
      case "answer": {
        try {
          await pc.setRemoteDescription({
            type: "answer",
            sdp: String(message.sdp ?? ""),
          });
          this.dispatch({ type: "answer", sessionId: String(message.session_id ?? "") });
        } catch (error) {
          this.dispatch({ type: "error", message: `setRemoteDescription: ${describeError(error)}` });
        }
        break;
      }
      case "ice-candidate": {
        const candidate = message.candidate as RTCIceCandidateInit | null | undefined;
        if (candidate && candidate.candidate) {
          try {
            await pc.addIceCandidate(candidate);
          } catch (error) {
            this.dispatch({ type: "error", message: `addIceCandidate: ${describeError(error)}` });
          }
        }
        break;
      }
      case "state":
        this.dispatch({ type: "ice", state: (message.state as IceState) ?? "none" });
        break;
      case "barge-in":
        this.dispatch({ type: "barge-in", durationMs: Number(message.duration_ms ?? 0) });
        break;
      case "pong":
        break;
      case "error":
        this.dispatch({ type: "error", message: String(message.message ?? "unknown error") });
        break;
      default:
        break;
    }
  }

  /**
   * Tell backend-core whether the agent is "playing" audio. This drives the
   * Phase-2 stub sink, which is what makes barge-in observable end-to-end: the
   * controller only interrupts while `sink.is_speaking`.
   */
  setAgentSpeaking(speaking: boolean): void {
    this.send({ type: "playback", state: speaking ? "start" : "stop" });
    this.dispatch({ type: "agent-speaking", speaking });
  }

  ping(): void {
    this.send({ type: "ping" });
  }

  private send(payload: Record<string, unknown>): void {
    const ws = this.ws;
    if (!ws || ws.readyState !== 1 /* OPEN */) return;
    try {
      ws.send(JSON.stringify(payload));
    } catch {
      /* socket went away between the check and the send */
    }
  }

  /** Tear everything down: mic, tracks, peer connection, socket. Idempotent. */
  close(): void {
    const wasActive = this.active;
    this.active = false;

    this.send({ type: "close" });
    if (this.ws) {
      try {
        this.ws.close();
      } catch {
        /* already closing */
      }
      this.ws = null;
    }

    if (this.localStream) {
      for (const track of this.localStream.getTracks()) track.stop();
      this.localStream = null;
    }
    if (this.remoteStream) {
      for (const track of this.remoteStream.getTracks()) track.stop();
      this.remoteStream = null;
    }

    if (this.pc) {
      this.pc.onicecandidate = null;
      this.pc.onconnectionstatechange = null;
      this.pc.oniceconnectionstatechange = null;
      this.pc.ontrack = null;
      try {
        this.pc.close();
      } catch {
        /* already closed */
      }
      this.pc = null;
    }

    if (wasActive) this.dispatch({ type: "closed" });
  }
}

/**
 * Resolve once `RTCPeerConnection.iceGatheringState` is `complete`, or after
 * `timeoutMs` (so a hostile network cannot hang the call setup forever).
 */
export function waitForIceGathering(pc: RTCPeerConnection, timeoutMs = 3000): Promise<void> {
  if (pc.iceGatheringState === "complete") return Promise.resolve();
  return new Promise((resolve) => {
    let settled = false;
    const finish = (): void => {
      if (settled) return;
      settled = true;
      pc.removeEventListener("icegatheringstatechange", onChange);
      clearTimeout(timer);
      resolve();
    };
    const onChange = (): void => {
      if (pc.iceGatheringState === "complete") finish();
    };
    const timer = setTimeout(finish, timeoutMs);
    pc.addEventListener("icegatheringstatechange", onChange);
  });
}

function describeError(error: unknown): string {
  if (error instanceof Error) return `${error.name}: ${error.message}`;
  return String(error);
}
