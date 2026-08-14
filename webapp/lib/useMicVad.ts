import { useCallback, useEffect, useRef, useState } from "react";
import type { MicVAD } from "@ricky0123/vad-web";
import { levelToSlots, makeLevelStore } from "@/lib/levelStore";
import { micErrorMessage } from "@/lib/secureContext";
import { MicProcessing, MIC_PROCESSING_DEFAULTS } from "@/lib/micProcessing";
import { useMirroredState } from "@/lib/useMirroredState";

/** Silence the VAD waits through before it calls an utterance finished. The
 * meter's dots are typed out over this window, so the wait reads as a
 * countdown — their delays in globals.css follow this number. */
const REDEMPTION_MS = 800;

/** Ceiling on one utterance, measured from the VAD hearing speech. Past it the
 * segment is cut and sent as it stands: the model takes about as long as the
 * clip does, so an unbroken monologue would otherwise leave you waiting. */
const MAX_SPEECH_MS = 30_000;

/** vad-web's hysteresis, restated so the meter watches the same edges it does:
 * speech starts above the first, ends below the second. */
const POSITIVE_SPEECH_THRESHOLD = 0.3;
const NEGATIVE_SPEECH_THRESHOLD = 0.25;

interface Options {
  /** A finished utterance, ready for the model. */
  onUtterance: (audio: Float32Array) => void;
  /** Ran inside the Microphone click, before anything is opened: whatever else
   * has to let go of the mouth, plus the checks that make starting pointless if
   * they fail. Throwing aborts the start and shows the message. */
  onBeforeStart: () => Promise<void>;
  /** Something else owns the synth — a request in flight, a playback. The mic
   * stays quiet rather than talking over it. */
  isBusy: () => boolean;
  /** A pointer is down on the scrub bar, which owns the synth until it lifts. */
  isScrubbing: () => boolean;
  /** Take the mouth, or hand it back. The mic claims nothing else. */
  setOwner: (owner: "mic" | "none") => void;
  setError: (message: string | null) => void;
}

/** The microphone half of the page: the VAD, what it hears, and who has the
 * mouth when it stops. Owns nothing of the playback side — it asks for the mic
 * back through `restoreMic` and is paused by everything that makes sound. */
export function useMicVad({
  onUtterance,
  onBeforeStart,
  isBusy,
  isScrubbing,
  setOwner,
  setError,
}: Options) {
  const vadRef = useRef<MicVAD | null>(null);
  /** User-intended mic state — the Microphone/Turn off toggle. The status alone
   * can't stand in for it (a dataset clip is "speaking" too). */
  const [micOn, micOnRef, setMicOn] = useMirroredState(false);
  /** Is the VAD hearing speech *this frame*? Drives the meter's colour, and
   * nothing else — unlike `recording` it flips tens of times a second, so it
   * must never gate a control. */
  const [heard, heardRef, setHeard] = useMirroredState(false);
  /** When the in-flight utterance started, or 0 if there isn't one. Watched
   * per frame against MAX_SPEECH_MS, and the meter types its dots while it
   * stands — an utterance is in flight but nothing is heard this instant. */
  const [speechStart, speechStartRef, setSpeechStart] = useMirroredState(0);
  // The ref is read by getStream/resumeStream, which vad-web calls on every
  // start().
  const [micProcessing, micProcessingRef, setMicProcessing] =
    useMirroredState<MicProcessing>(MIC_PROCESSING_DEFAULTS);
  const [levelStore] = useState(makeLevelStore);

  // Everything below reads the options through this ref rather than closing
  // over them, so a fresh closure from the page can't re-bind the callbacks —
  // startMic in particular hangs off the VAD it built.
  const options = {
    onUtterance,
    onBeforeStart,
    isBusy,
    isScrubbing,
    setOwner,
    setError,
  };
  const optionsRef = useRef(options);
  // After every render, not during one: the VAD's callbacks all fire from
  // timers and audio frames, long after the commit.
  useEffect(() => {
    optionsRef.current = options;
  });

  // A page that goes away shouldn't leave the mic open behind it.
  useEffect(
    () => () => {
      vadRef.current?.destroy();
      vadRef.current = null;
    },
    [],
  );

  /** Edge-triggered, so a frame that changes nothing never reaches React. */
  const showHeard = useCallback(
    (value: boolean) => {
      if (heardRef.current === value) return;
      setHeard(value);
    },
    [heardRef, setHeard],
  );

  /** Resume the VAD only if the user hasn't muted the mic. Everything that
   * takes the mouth hands it back this way. */
  const restoreMic = useCallback(async () => {
    const { isScrubbing, setOwner } = optionsRef.current;
    if (isScrubbing()) return; // the scrub owns the synth until pointer-up
    // A pause elsewhere (playback, a mic-processing toggle) discards whatever
    // was in flight, so the clock can't carry over into the next utterance.
    setSpeechStart(0);
    if (micOnRef.current) {
      await vadRef.current?.start();
      setOwner("mic");
    } else {
      setOwner("none");
    }
  }, [micOnRef, setSpeechStart]);

  /** Cut an over-long utterance short and send what's been said. Pausing with
   * submitUserSpeechOnPause is what ends the segment; onUtterance takes it from
   * there and hands the mic back when it's done, so the restart here is only for
   * the case where the VAD had nothing to give us after all. */
  const cutUtterance = useCallback(async () => {
    const vad = vadRef.current;
    if (!vad) return;
    setSpeechStart(0);
    showHeard(false);
    levelStore.set(0);
    vad.setOptions({ submitUserSpeechOnPause: true });
    await vad.pause();
    vad.setOptions({ submitUserSpeechOnPause: false });
    if (!optionsRef.current.isBusy()) await restoreMic();
  }, [restoreMic, showHeard, levelStore, setSpeechStart]);

  const startMic = useCallback(async () => {
    const { setOwner, setError } = optionsRef.current;
    setError(null);
    try {
      await optionsRef.current.onBeforeStart();

      if (!vadRef.current) {
        const { MicVAD } = await import("@ricky0123/vad-web");
        // vad-web's default getStream/resumeStream hardcode all three
        // processing flags on; ours re-read the ref, and since pause() stops
        // the tracks and start() re-acquires, a toggle lands on the next
        // listening cycle without rebuilding the VAD.
        const getStream = () =>
          navigator.mediaDevices.getUserMedia({
            audio: { channelCount: 1, ...micProcessingRef.current },
          });
        vadRef.current = await MicVAD.new({
          model: "v5",
          baseAssetPath: "/vad/",
          onnxWASMBasePath: "/vad/",
          getStream,
          resumeStream: getStream,
          redemptionMs: REDEMPTION_MS,
          preSpeechPadMs: 150, // default 800ms puts noticeable silence before speech
          positiveSpeechThreshold: POSITIVE_SPEECH_THRESHOLD,
          negativeSpeechThreshold: NEGATIVE_SPEECH_THRESHOLD,
          onSpeechStart: () => setSpeechStart(performance.now()),
          onVADMisfire: () => {
            setSpeechStart(0);
            levelStore.set(0);
          },
          onSpeechEnd: (audio) => {
            setSpeechStart(0);
            showHeard(false);
            levelStore.set(0);
            optionsRef.current.onUtterance(audio);
          },
          onFrameProcessed: ({ isSpeech }, frame) => {
            if (optionsRef.current.isBusy() || !micOnRef.current) return;
            if (isSpeech >= POSITIVE_SPEECH_THRESHOLD) showHeard(true);
            else if (isSpeech < NEGATIVE_SPEECH_THRESHOLD) showHeard(false);
            levelStore.set(levelToSlots(frame));
            if (
              speechStartRef.current &&
              performance.now() - speechStartRef.current >= MAX_SPEECH_MS
            ) {
              void cutUtterance();
            }
          },
        });
      }
      setMicOn(true);
      await vadRef.current.start();
      setOwner("mic");
    } catch (e) {
      setError(micErrorMessage(e));
      setMicOn(false);
      setOwner("none");
    }
  }, [
    showHeard,
    levelStore,
    cutUtterance,
    micOnRef,
    setMicOn,
    micProcessingRef,
    setSpeechStart,
    speechStartRef,
  ]);

  /** Turn the mic off. `submit` sends a half-spoken utterance rather than
   * binning it — true from the button, since people press it meaning "I'm
   * done"; false when the tab is left, where an unasked-for answer is worse. */
  const stopMic = useCallback(
    async (submit = false) => {
      setMicOn(false);
      showHeard(false);
      levelStore.set(0);
      setSpeechStart(0);
      const vad = vadRef.current;
      // Scoped to this one pause: the others (playback starting, a
      // mic-processing toggle) must keep discarding, or the response would feed
      // itself back in as a new utterance.
      if (submit) vad?.setOptions({ submitUserSpeechOnPause: true });
      await vad?.pause(); // fires onSpeechEnd synchronously if it had one
      if (submit) vad?.setOptions({ submitUserSpeechOnPause: false });
      // onUtterance has already taken the mouth by now if something was sent.
      if (!optionsRef.current.isBusy()) optionsRef.current.setOwner("none");
    },
    [showHeard, levelStore, setMicOn, setSpeechStart],
  );

  /** Flip one mic-processing flag. If we're listening right now, cycle the
   * stream so it takes effect immediately rather than after the next
   * utterance. */
  const toggleMicProcessing = useCallback(
    async (key: keyof MicProcessing) => {
      const next = {
        ...micProcessingRef.current,
        [key]: !micProcessingRef.current[key],
      };
      setMicProcessing(next);

      const vad = vadRef.current;
      if (
        !vad ||
        optionsRef.current.isBusy() ||
        !micOnRef.current ||
        !vad.listening
      )
        return;
      try {
        await vad.pause();
        await vad.start();
      } catch (e) {
        optionsRef.current.setError(micErrorMessage(e));
      }
    },
    [micOnRef, micProcessingRef, setMicProcessing],
  );

  /** Stop listening without touching the user's intent, for whatever is about
   * to make sound. `restoreMic` is what undoes it. */
  const pauseMic = useCallback(() => vadRef.current?.pause(), []);

  return {
    micOn,
    micOnRef,
    heard,
    /** An utterance is in flight: speech started and has not ended yet. */
    recording: speechStart !== 0,
    levelStore,
    micProcessing,
    startMic,
    stopMic,
    pauseMic,
    restoreMic,
    toggleMicProcessing,
  };
}
