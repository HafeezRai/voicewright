/**
 * React binding for {@link VoiceClient}: owns the session reducer, keeps one
 * client instance alive for the component's lifetime, and tears it down on
 * unmount (so a route change cannot leak a microphone).
 */

import { useCallback, useEffect, useReducer, useRef, useState } from "react";

import { VoiceClient } from "../lib/voiceClient";
import { type VoiceState, initialVoiceState, voiceReducer } from "../state/voiceMachine";

export interface UseVoiceClient {
  state: VoiceState;
  remoteStream: MediaStream | null;
  start(): Promise<void>;
  stop(): void;
  setAgentSpeaking(speaking: boolean): void;
}

/** Test/E2E handle published on `window.__voicewrightVoice`. */
declare global {
  interface Window {
    __voicewrightVoice?: {
      status: VoiceState["status"];
      ice: VoiceState["ice"];
      sessionId: string | null;
      bargeInCount: number;
    };
  }
}

export function useVoiceClient(): UseVoiceClient {
  const [state, dispatch] = useReducer(voiceReducer, initialVoiceState);
  const [remoteStream, setRemoteStream] = useState<MediaStream | null>(null);
  const clientRef = useRef<VoiceClient | null>(null);

  const start = useCallback(async () => {
    clientRef.current?.close();
    dispatch({ type: "start" });
    const client = new VoiceClient({
      dispatch,
      onRemoteStream: (stream) => setRemoteStream(stream),
    });
    clientRef.current = client;
    setRemoteStream(null);
    await client.start();
  }, []);

  const stop = useCallback(() => {
    clientRef.current?.close();
    clientRef.current = null;
    setRemoteStream(null);
  }, []);

  const setAgentSpeaking = useCallback((speaking: boolean) => {
    clientRef.current?.setAgentSpeaking(speaking);
  }, []);

  // Release the microphone and peer connection if the component unmounts.
  useEffect(() => {
    return () => {
      clientRef.current?.close();
      clientRef.current = null;
    };
  }, []);

  // Publish a compact snapshot for the Playwright end-to-end test.
  useEffect(() => {
    window.__voicewrightVoice = {
      status: state.status,
      ice: state.ice,
      sessionId: state.sessionId,
      bargeInCount: state.bargeInCount,
    };
  }, [state.status, state.ice, state.sessionId, state.bargeInCount]);

  return { state, remoteStream, start, stop, setAgentSpeaking };
}
