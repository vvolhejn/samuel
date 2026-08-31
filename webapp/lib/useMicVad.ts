import { useCallback, useEffect, useRef, useState } from "react";
import type { MicVAD } from "@ricky0123/vad-web";
import { levelToSlots, makeLevelStore } from "@/lib/levelStore";
import { micErrorMessage } from "@/lib/secureContext";
import { MicProcessing, MIC_PROCESSING_DEFAULTS } from "@/lib/micProcessing";
import { useMirroredState } from "@/lib/useMirroredState";

/** Ceiling on one recording. Past it we stop and send what was said: the model
 * takes about as long as the clip does, so an unbroken monologue would otherwise
 * leave you waiting. */
const MAX_RECORDING_MS = 30_000;

/** vad-web's hysteresis, which is all we use it for: a frame above the first is
 * speech, one below the second isn't, and in between we hold the last answer. */
const POSITIVE_SPEECH_THRESHOLD = 0.3;
const NEGATIVE_SPEECH_THRESHOLD = 0.25;

/** Kept either side of the speech we found. The model hears the attack of the
 * first word and the tail of the last, both of which sit under the thresholds. */
const PAD_MS = 150;

/** One 16 kHz frame off the VAD, and how much of it the model thought was
 * speech. */
interface Frame {
  audio: Float32Array;
  isSpeech: number;
}

/** The recording, trimmed to the speech in it. Null if there wasn't any —
 * silence is not worth a round trip. */
function trimToSpeech(frames: Frame[]): Float32Array | null {
  let first = -1;
  let last = -1;
  for (const [i, frame] of frames.entries()) {
    if (frame.isSpeech < POSITIVE_SPEECH_THRESHOLD) continue;
    if (first === -1) first = i;
    last = i;
  }
  if (first === -1) return null;

  // Widen over the frames the model is unsure about before padding: quiet
  // consonants at either end of a word land between the two thresholds.
  while (first > 0 && frames[first - 1].isSpeech >= NEGATIVE_SPEECH_THRESHOLD)
    first--;
  while (
    last + 1 < frames.length &&
    frames[last + 1].isSpeech >= NEGATIVE_SPEECH_THRESHOLD
  )
    last++;

  const msPerFrame = frames[0].audio.length / 16; // 16 samples per ms at 16 kHz
  const pad = Math.round(PAD_MS / msPerFrame);
  const kept = frames.slice(
    Math.max(0, first - pad),
    Math.min(frames.length, last + 1 + pad),
  );

  const audio = new Float32Array(
    kept.reduce((total, frame) => total + frame.audio.length, 0),
  );
  let offset = 0;
  for (const frame of kept) {
    audio.set(frame.audio, offset);
    offset += frame.audio.length;
  }
  return audio;
}

interface Options {
  /** A finished recording, trimmed and ready for the model. */
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

/** The microphone half of the page: press to record, press again to send. The
 * VAD is not what decides when you're done — it only says which frames were
 * speech, so the recording can be trimmed to them before it goes off. Owns
 * nothing of the playback side, and is paused by everything that makes sound. */
export function useMicVad({
  onUtterance,
  onBeforeStart,
  isBusy,
  isScrubbing,
  setOwner,
  setError,
}: Options) {
  const vadRef = useRef<MicVAD | null>(null);
  /** User-intended mic state — the Microphone/Stop toggle, and so also "a
   * recording is in progress". */
  const [micOn, micOnRef, setMicOn] = useMirroredState(false);
  /** Between the press and the first frame of audio. onBeforeStart wakes the
   * backend, which takes a container cold start when it has been asleep, and
   * the VAD then loads its model. Nothing looks different for either. */
  const [starting, startingRef, setStarting] = useMirroredState(false);
  /** Is the VAD hearing speech *this frame*? Drives the meter's colour, and
   * nothing else — it flips tens of times a second, so it must never gate a
   * control. */
  const [heard, heardRef, setHeard] = useMirroredState(false);
  /** The recording so far. A ref, not state: it grows ~31 times a second and
   * nothing renders from it. */
  const framesRef = useRef<Frame[]>([]);
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

  /** Turn the mic off. `submit` sends what was recorded rather than binning it —
   * true from the Stop button, since that is what it's for; false when the tab
   * is left, where an unasked-for answer is worse. */
  const stopMic = useCallback(
    async (submit = false) => {
      setMicOn(false);
      showHeard(false);
      levelStore.set(0);
      const frames = framesRef.current;
      framesRef.current = [];
      await vadRef.current?.pause();
      if (submit) {
        const audio = trimToSpeech(frames);
        if (audio) optionsRef.current.onUtterance(audio);
        else optionsRef.current.setError("Didn't hear any speech.");
      }
      // onUtterance has already taken the mouth by now if something was sent.
      if (!optionsRef.current.isBusy()) optionsRef.current.setOwner("none");
    },
    [showHeard, levelStore, setMicOn],
  );

  const startMic = useCallback(async () => {
    const { setOwner, setError } = optionsRef.current;
    // The button stays pressable through the wait for anyone using a keyboard,
    // so a second press must not build a second VAD over the first.
    if (startingRef.current) return;
    setError(null);
    setStarting(true);
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
        // No onSpeechStart/onSpeechEnd: the VAD's own segmenting decides
        // nothing here, and the audio we send is the one we assemble below.
        vadRef.current = await MicVAD.new({
          model: "v5",
          baseAssetPath: "/vad/",
          onnxWASMBasePath: "/vad/",
          getStream,
          resumeStream: getStream,
          positiveSpeechThreshold: POSITIVE_SPEECH_THRESHOLD,
          negativeSpeechThreshold: NEGATIVE_SPEECH_THRESHOLD,
          onFrameProcessed: ({ isSpeech }, frame) => {
            if (optionsRef.current.isBusy() || !micOnRef.current) return;
            if (isSpeech >= POSITIVE_SPEECH_THRESHOLD) showHeard(true);
            else if (isSpeech < NEGATIVE_SPEECH_THRESHOLD) showHeard(false);
            levelStore.set(levelToSlots(frame));
            const frames = framesRef.current;
            frames.push({ audio: frame, isSpeech });
            const ms = frames.length * (frame.length / 16);
            if (ms >= MAX_RECORDING_MS) void stopMic(true);
          },
        });
      }
      framesRef.current = [];
      setMicOn(true);
      await vadRef.current.start();
      setOwner("mic");
    } catch (e) {
      setError(micErrorMessage(e));
      setMicOn(false);
      setOwner("none");
    } finally {
      setStarting(false);
    }
  }, [
    showHeard,
    levelStore,
    stopMic,
    micOnRef,
    setMicOn,
    micProcessingRef,
    startingRef,
    setStarting,
  ]);

  /** Resume the VAD only if the user hasn't turned the mic off. Everything that
   * takes the mouth hands it back this way. */
  const restoreMic = useCallback(async () => {
    const { isScrubbing, setOwner } = optionsRef.current;
    if (isScrubbing()) return; // the scrub owns the synth until pointer-up
    if (micOnRef.current) {
      // Whatever took the mouth (a playback, a mic-processing toggle) has been
      // making sound into the recording; start the take over rather than send a
      // recording with a hole in it.
      framesRef.current = [];
      await vadRef.current?.start();
      setOwner("mic");
    } else {
      setOwner("none");
    }
  }, [micOnRef]);

  /** Flip one mic-processing flag. If we're recording right now, cycle the
   * stream so it takes effect immediately. */
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
        framesRef.current = [];
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
    starting,
    heard,
    levelStore,
    micProcessing,
    startMic,
    stopMic,
    pauseMic,
    restoreMic,
    toggleMicProcessing,
  };
}
