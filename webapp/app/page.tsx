"use client";

import {
  useCallback,
  useEffect,
  useRef,
  useState,
  useSyncExternalStore,
} from "react";
import {
  fetchDatasetClips,
  fetchHealth,
  fetchPrecomputedIndex,
  synthesizeDatasetClip,
  synthesizeUtterance,
  wavBlob,
  DatasetClip,
  HealthResponse,
  PrecomputedIndex,
  SynthResponse,
} from "@/lib/audio";
import { insecureContextMessage } from "@/lib/secureContext";
import { downloadBlob, makeZip } from "@/lib/zip";
import { usePinkTrombone } from "@/lib/usePinkTrombone";
import { useOriginalAudio } from "@/lib/useOriginalAudio";
import { useMicVad } from "@/lib/useMicVad";
import { useMirroredState } from "@/lib/useMirroredState";
import type { Owner } from "@/lib/owner";
import { MUTED_LINK, STRIP, TextLink, sectionBox } from "@/components/ui";
import { LevelMeter } from "@/components/LevelMeter";
import { TractStage } from "@/components/TractStage";
import { DebugPanel } from "@/components/DebugPanel";
import type {
  PinkTromboneElement,
  VoiceboxEventDetail,
} from "@/types/pink-trombone";

/** The three stages of the page, in the order you use them. Exactly one is
 * highlighted at a time, which is how the eye gets handed along. */
type Section = "input" | "playback" | "tract";

/** Which of the two audio sources the transport plays: the model's imitation,
 * or the audio it was made from. */
type Source = "samuel" | "original";

/** Grace period before an unfocused window mutes its mic. A hidden tab doesn't
 * get it — that one mutes immediately. */
const BLUR_MUTE_MS = 60_000;

/** Nothing ever invalidates the secure-context snapshot. */
const subscribeNever = () => () => {};

/** One round trip: what the model heard, and what it said back. Held in memory
 * for the whole session, so cap it — a few seconds of 16 kHz float WAV is
 * ~250 kB per side, and nothing else evicts them. */
const MAX_HISTORY = 50;

/** What `imitate` needs back from a request: the answer, plus the audio that
 * was sent, for the transport and the session download. */
type SynthRequest = {
  response: SynthResponse;
  inputUrl: string;
  inputBlob: Blob;
};

interface Recording {
  kind: "mic" | "dataset";
  /** Audio the model was given: a WAV from the mic, an MP3 for a clip. */
  input: Blob;
  /** WAV of the model's output, rendered by the Python synth. Null for a
   * precomputed clip: those responses drop the reference audio. */
  output: Blob | null;
}

/** How long a precomputed clip pretends to think, in ms. The answer is on disk
 * and comes back in no time, which reads as a button that didn't work — and
 * makes the six clips feel unlike the mic, which really does take a moment. */
const FAKE_THINKING_MS = [500, 1000] as const;

const sleep = (ms: number) =>
  new Promise<void>((resolve) => setTimeout(resolve, ms));

function fakeThinking(): Promise<void> {
  const [lo, hi] = FAKE_THINKING_MS;
  return sleep(lo + Math.random() * (hi - lo));
}

export default function Home() {
  /** Who is driving the synth right now. The whole page hangs off this: what
   * a new action has to stop, what the mic may do, and what the UI dims. */
  const [owner, ownerRef, setOwner] = useMirroredState<Owner>("none");
  const [error, setError] = useState<string | null>(null);
  /** null until the first mic start asks the backend. The page never contacts
   * it on load: waking the container costs memory for as long as it stays
   * awake, and the clips are answered from the committed responses. */
  const [health, setHealth] = useState<HealthResponse | null>(null);
  /** The model's last answer. `lastResponse` is the copy the callbacks read;
   * refs must not be read in render, so the state is what the panel shows. */
  const [viewResponse, lastResponse, setResponse] =
    useMirroredState<SynthResponse | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  /** Which source the transport is pointed at. */
  const [source, sourceRef, setSource] = useMirroredState<Source>("samuel");
  /** Playback/scrub position within the current source, in [0, 1]. */
  const [scrubFrac, scrubFracRef, setScrubFrac] = useMirroredState(0);
  /** Pre-recorded clips shipped with the app, and the last one played (which
   * is the button left filled in). */
  const [clips, setClips] = useState<DatasetClip[]>([]);
  const [playedClip, setPlayedClip] = useState<string>("");
  /** The committed clip answers, if any were generated for this checkpoint. */
  const [precomputed, setPrecomputed] = useState<PrecomputedIndex | null>(null);
  /** Is the mouth yours to play? While it is, the voicebox pad under the tract
   * and the tract itself take drags; while it isn't they are read-only, and the
   * model has the only hands on them. */
  const [manual, manualRef, setManual] = useMirroredState(false);
  const [debugOpen, setDebugOpen] = useState(false);
  /** Is the intro past its first sentence showing? Only ever false below `md`,
   * where the fold is worth saving. */
  const [introOpen, setIntroOpen] = useState(false);
  /** Why this origin can't run the app at all, or null. Read through
   * useSyncExternalStore because it is a client-only fact: the exported HTML is
   * built without an origin. */
  const insecure = useSyncExternalStore(
    subscribeNever,
    insecureContextMessage,
    () => null,
  );

  const trombone = usePinkTrombone();
  /** The audio the model heard, and everything that plays or previews it. */
  const original = useOriginalAudio(trombone);
  /** Every utterance this session, for the debug panel's download. */
  const historyRef = useRef<Recording[]>([]);
  const [historyCount, setHistoryCount] = useState(0);
  /** Pointer is down on the voicebox pad, i.e. the synth is sounding by hand. */
  const voicingRef = useRef(false);
  /** Bumped whenever a playback starts or is cut short, so the completion of a
   * superseded one can't clear the new one's state out from under it. */
  const playIdRef = useRef(0);

  /** Is the mouth taken by something the mic must not talk over? */
  const isBusy = useCallback(
    () => ownerRef.current !== "none" && ownerRef.current !== "mic",
    [ownerRef],
  );

  // The mic and everything downstream of it. Its callbacks are read through a
  // ref inside the hook, so the ones named here may be defined further down —
  // which is what lets the mic hand an utterance to a playback that in turn
  // hands the mic back.
  const {
    micOn,
    micOnRef,
    starting: micStarting,
    heard,
    levelStore,
    micProcessing,
    startMic,
    stopMic,
    pauseMic,
    restoreMic,
    toggleMicProcessing,
  } = useMicVad({
    onUtterance: (audio) => void onUtterance(audio),
    onBeforeStart: async () => {
      const current = await fetchHealth();
      if (!current) {
        throw new Error(
          "Model backend unreachable — run: uv run --extra server python -m samuel.server --no-frontend",
        );
      }
      setHealth(current);
      await trombone.resume(); // we're in a user gesture
    },
    isBusy,
    isScrubbing: () => ownerRef.current === "scrub",
    setOwner,
    setError,
  });

  // Bring up the synth + tract visualization immediately; the AudioContext
  // stays suspended until the first user gesture (Start).
  useEffect(() => {
    // On an insecure origin nothing here works, and starting the synth anyway
    // is what makes the visualization glitch — see insecureContextMessage.
    const element = insecureContextMessage()
      ? null
      : document.querySelector<PinkTromboneElement>("pink-trombone");
    if (element) {
      trombone.init(element).catch((e) => {
        setError(e instanceof Error ? e.message : String(e));
      });
    }
  }, [trombone]);

  // The pre-recorded clips, one button each, and the answers committed for them.
  useEffect(() => {
    void fetchDatasetClips().then(setClips);
    void fetchPrecomputedIndex().then(setPrecomputed);
  }, []);

  /** Stop sounding the voicebox, if manual control was. */
  const endVoice = useCallback(() => {
    if (!voicingRef.current) return;
    voicingRef.current = false;
    trombone.endVoice();
  }, [trombone]);

  /** Take the mouth back off the user. Called by everything else that wants to
   * drive it — a playback, a scrub, the mic — rather than disabling those while
   * manual control is on: a dead button you have to find the release for is
   * worse than a mode that steps aside when you reach past it. */
  const exitManual = useCallback(() => {
    if (!manualRef.current) return;
    setManual(false);
    endVoice();
  }, [endVoice, manualRef, setManual]);

  /** Silence whatever is playing and orphan its completion handler, leaving the
   * scrub position where it stands. Called when something takes the mouth off a
   * playback, which is what makes a second clip interrupt the first instead of
   * being locked out. */
  const stopPlayback = useCallback(() => {
    playIdRef.current++;
    trombone.stop();
    original.pause();
    original.stopGrain();
    setIsPlaying(false);
  }, [trombone, original]);

  /** Take the mouth for `next`, and stop whoever had it. Keyed on the previous
   * owner, so an entry point cannot forget one of the ways to let go.
   *
   * A scrub is the one exception: dragging the bar during a playback seeks that
   * playback rather than stopping it, so the playback stays underneath and is
   * stopped when the scrub itself ends. Waiting on the model is not stopped at
   * all — the callers refuse to take the mouth off it. */
  const claim = useCallback(
    (next: Owner) => {
      const prev = ownerRef.current;
      if (prev === "manual") exitManual();
      if (prev === "mic") void pauseMic();
      if (prev === "scrub") {
        trombone.endScrub();
        original.stopGrain();
      }
      if (prev === "scrub" || (prev === "playback" && next !== "scrub"))
        stopPlayback();
      setOwner(next);
    },
    [
      exitManual,
      pauseMic,
      stopPlayback,
      trombone,
      original,
      ownerRef,
      setOwner,
    ],
  );

  /** End of a playback that wasn't superseded: hand the mic back. */
  const finishPlayback = useCallback(
    async (id: number) => {
      if (playIdRef.current !== id) return;
      setIsPlaying(false);
      await restoreMic();
    },
    [restoreMic],
  );

  const playSamuel = useCallback(
    async (response: SynthResponse, startFrac = 0) => {
      claim("playback"); // stops the mic too, so the synth can't retrigger it
      const id = playIdRef.current;
      setIsPlaying(true);
      try {
        await trombone.speak(response, {
          startFrac,
          onProgress: (frac) => {
            if (playIdRef.current === id) setScrubFrac(frac);
          },
        });
      } finally {
        await finishPlayback(id);
      }
    },
    [trombone, claim, finishPlayback, setScrubFrac],
  );

  /** Play the audio the model heard — your trimmed recording, or the dataset
   * clip — at its recorded level; the RMS normalisation happens server-side.
   * Drives the same scrub bar as the imitation, off the element's clock. */
  const playOriginal = useCallback(
    async (startFrac = 0) => {
      if (!original.loaded) return;
      claim("playback");
      const id = playIdRef.current;
      setIsPlaying(true);
      try {
        await original.play(startFrac);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
        await finishPlayback(id);
        return;
      }
      // Poll rather than lean on timeupdate, which fires a few times a second
      // and leaves the bar visibly stepping.
      const tick = () => {
        if (playIdRef.current !== id) return;
        setScrubFrac(original.currentFrac());
        if (original.ended()) {
          setScrubFrac(1);
          void finishPlayback(id);
        } else if (original.paused()) {
          void finishPlayback(id);
        } else {
          requestAnimationFrame(tick);
        }
      };
      requestAnimationFrame(tick);
    },
    [original, claim, finishPlayback, setScrubFrac],
  );

  const remember = useCallback((recording: Recording) => {
    const history = historyRef.current;
    history.push(recording);
    if (history.length > MAX_HISTORY)
      history.splice(0, history.length - MAX_HISTORY);
    setHistoryCount(history.length);
  }, []);

  /** One zip of every input/output pair this session. */
  const downloadHistory = useCallback(async () => {
    const entries = await Promise.all(
      historyRef.current.flatMap((recording, i) => {
        const stem = `${String(i + 1).padStart(2, "0")}-${recording.kind}`;
        const inputExt = recording.input.type === "audio/mpeg" ? "mp3" : "wav";
        const files: Array<[string, Blob]> = [
          [`${stem}-input.${inputExt}`, recording.input],
        ];
        // Absent for a precomputed clip — see Recording.output.
        if (recording.output)
          files.push([`${stem}-output.wav`, recording.output]);
        return files.map(([name, blob]) =>
          blob.arrayBuffer().then((b) => ({ name, bytes: new Uint8Array(b) })),
        );
      }),
    );
    downloadBlob(makeZip(entries), "samuel-session.zip");
  }, []);

  /** One round trip through the model: send it some audio, keep what comes
   * back, and play the imitation. `send` is whatever gets the answer, and the
   * wait for it is the one thing nothing else may interrupt. */
  const imitate = useCallback(
    async (kind: Recording["kind"], send: () => Promise<SynthRequest>) => {
      claim("model");
      try {
        const { response, inputUrl, inputBlob } = await send();
        setResponse(response);
        original.set(inputUrl);
        remember({
          kind,
          input: inputBlob,
          output: response.synth_audio_b64
            ? wavBlob(response.synth_audio_b64)
            : null,
        });
        // A fresh imitation is what you want to hear, whatever the toggle was
        // left on.
        setSource("samuel");
        await playSamuel(response);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
        await restoreMic();
      }
    },
    [playSamuel, restoreMic, original, remember, setResponse, setSource, claim],
  );

  const onUtterance = useCallback(
    async (audio: Float32Array) => {
      if (isBusy()) return;
      // Your own voice is now what the transport holds, so no clip button stays
      // filled in and no clip is credited under the bar.
      setPlayedClip("");
      await imitate("mic", () => synthesizeUtterance(audio));
    },
    [imitate, isBusy],
  );

  // Leaving shouldn't leave a live mic — or a held note — behind, and coming
  // back is always an explicit gesture: nothing auto-resumes. A hidden tab is
  // gone for good, so it mutes at once; merely losing focus gets BLUR_MUTE_MS of
  // grace, since another window on top is as likely to be devtools as it is to
  // be leaving.
  useEffect(() => {
    let timer: ReturnType<typeof setTimeout> | null = null;
    const cancel = () => {
      if (timer) clearTimeout(timer);
      timer = null;
    };
    const leave = () => {
      cancel();
      if (micOnRef.current) void stopMic();
      // Manual control sustains the synth indefinitely, which a tab you've
      // walked away from must not go on doing.
      if (ownerRef.current === "manual") claim("none");
    };
    const onVisibility = () => {
      if (document.hidden) leave();
    };
    const onBlur = () => {
      cancel();
      timer = setTimeout(leave, BLUR_MUTE_MS);
    };
    document.addEventListener("visibilitychange", onVisibility);
    window.addEventListener("blur", onBlur);
    window.addEventListener("focus", cancel);
    return () => {
      cancel();
      document.removeEventListener("visibilitychange", onVisibility);
      window.removeEventListener("blur", onBlur);
      window.removeEventListener("focus", cancel);
    };
  }, [stopMic, claim, micOnRef, ownerRef]);

  /** Mimic one of the pre-recorded clips. Interrupts anything playing — only a
   * request already in flight to the model holds it off. */
  const playClip = useCallback(
    async (name: string) => {
      const clip = clips.find((c) => c.name === name);
      if (!clip || ownerRef.current === "model") return;
      setError(null);
      setPlayedClip(name);
      await imitate("dataset", async () => {
        await trombone.resume(); // we're in a user gesture
        console.log(`${clip.name}: ${clip.source} @${clip.offset_s}s`);
        const answer = await synthesizeDatasetClip(
          clip,
          health?.model_fingerprint,
        );
        // A committed answer is there instantly; hold it for a beat so the
        // clips behave like the mic rather than firing on mousedown.
        if (answer.precomputed) await fakeThinking();
        return answer;
      });
    },
    [clips, health, trombone, imitate, ownerRef],
  );

  /** Start whichever source is selected, from `frac`. */
  const playFrom = useCallback(
    async (which: Source, frac: number) => {
      setError(null);
      if (which === "original") {
        await playOriginal(frac);
      } else if (lastResponse.current) {
        await playSamuel(lastResponse.current, frac);
      }
    },
    [playOriginal, playSamuel, lastResponse],
  );

  /** The Microphone/Stop button. Starting takes the whole mouth, like everything
   * else that makes a sound; stopping sends what you said. */
  const toggleMic = useCallback(() => {
    if (micOnRef.current) {
      void stopMic(true);
      return;
    }
    claim("none");
    void startMic();
  }, [claim, startMic, stopMic, micOnRef]);

  /** Play from the scrub position (or the start, if at the end); pause if
   * already playing. */
  const togglePlay = useCallback(async () => {
    if (isPlaying) {
      claim("none");
      await restoreMic();
      return;
    }
    if (ownerRef.current === "model") return;
    const from = scrubFracRef.current >= 0.995 ? 0 : scrubFracRef.current;
    await playFrom(sourceRef.current, from);
  }, [
    isPlaying,
    claim,
    restoreMic,
    playFrom,
    scrubFracRef,
    sourceRef,
    ownerRef,
  ]);

  /** Play the other take of the same utterance. The switch and the playback are
   * one action: this is offered as a link, and a link that says "play" has to
   * play. The two are the same utterance, so the position carries over. */
  const playOther = useCallback(async () => {
    const next: Source = sourceRef.current === "samuel" ? "original" : "samuel";
    if (next === "original" && !original.loaded) return;
    setSource(next);
    const from = scrubFracRef.current >= 0.995 ? 0 : scrubFracRef.current;
    await playFrom(next, from);
  }, [playFrom, original.loaded, sourceRef, setSource, scrubFracRef]);

  const onScrub = useCallback(
    (frac: number) => {
      setScrubFrac(frac);
      // Takes the tract off whatever had it, but rides on a running playback:
      // the bar seeks it rather than stopping it.
      if (ownerRef.current !== "scrub") claim("scrub");
      if (sourceRef.current === "original") {
        original.seek(frac);
        // Mid-playback the element is already the sound, and dragging just
        // seeks it. Stopped, the grains are the sound.
        if (original.paused()) original.playGrain(frac);
        return;
      }
      const response = lastResponse.current;
      if (!response) return;
      trombone.scrub(response, frac);
    },
    [
      trombone,
      setScrubFrac,
      original,
      claim,
      sourceRef,
      lastResponse,
      ownerRef,
    ],
  );

  /** Letting go of the bar plays on from where you dropped it, whether or not
   * it was playing when you grabbed it — the scrub is a seek, not a stop. */
  const onScrubEnd = useCallback(async () => {
    // Anything that took the mouth mid-drag (a clip, say) already released the
    // scrub through `claim`, and owns what happens next.
    if (ownerRef.current !== "scrub") return;
    const frac = scrubFracRef.current;
    const hasSource =
      sourceRef.current === "original"
        ? original.loaded
        : !!lastResponse.current;
    // playFrom claims the mouth, which is what ends the scrub in this branch.
    if (hasSource && frac < 0.995) {
      await playFrom(sourceRef.current, frac);
      return;
    }
    claim("none");
    await restoreMic();
  }, [
    original,
    restoreMic,
    playFrom,
    claim,
    scrubFracRef,
    sourceRef,
    lastResponse,
    ownerRef,
  ]);

  // A drag on the voicebox pad, reported by the element (GlottisUI in the fork
  // dispatches it, having worked out the pitch and voicing from the pointer).
  // preventDefault() is what stops it setting the AudioParams itself: a plain
  // `.value` write doesn't cancel our scheduled curves, so the two would fight
  // for the length of the drag. Scheduling stays ours.
  useEffect(() => {
    const element =
      document.querySelector<PinkTromboneElement>("pink-trombone");
    if (!element) return;

    const onVoicebox = (event: Event) => {
      event.preventDefault();
      const { frequency, tenseness } = (
        event as CustomEvent<VoiceboxEventDetail>
      ).detail;
      // Releasing the pad is not a reason to stop: manual control holds the
      // note until it is switched off, so "end" carries no values and there is
      // nothing to do with it.
      if (frequency === undefined || tenseness === undefined) return;
      trombone.voice(frequency, tenseness);
    };

    element.addEventListener("voicebox", onVoicebox);
    return () => element.removeEventListener("voicebox", onVoicebox);
  }, [trombone]);

  /** Manual control is a switch, not a key: it holds the note for as long as
   * it's on, and dragging the voicebox or the tract shapes what you're already
   * hearing. */
  const toggleManual = useCallback(() => {
    if (manualRef.current) {
      claim("none");
      void restoreMic(); // nothing to hand back to, so: nobody
      return;
    }
    // Silences the imitation but leaves the tract in the pose it reached: that
    // pose is the interesting starting point, and startVoice picks up the note
    // the playback left off on, so switching this on continues from wherever
    // the model got to rather than from some default.
    claim("manual");
    setManual(true);
    void trombone.resume(); // we're in a user gesture
    voicingRef.current = true;
    trombone.startVoice();
  }, [claim, restoreMic, trombone, manualRef, setManual]);

  // Playback is *not* a reason to disable anything: every control below
  // interrupts it cleanly. Waiting on the model is.
  const notBusy = insecure === null && owner !== "model";
  const hasSource =
    source === "original" ? original.loaded : viewResponse !== null;
  // Is there anything at all to play, on either source? Not `hasSource`:
  // flipping the switch to a side that happens to be empty shouldn't drain the
  // colour out of the switch you'd flip back with.
  const transportReady = original.loaded || viewResponse !== null;
  // One gate for the whole transport, so every control inside it dims and
  // desaturates on the same condition — a control that stays enabled here would
  // desaturate without dimming and read a shade darker than its neighbours.
  const transportLive = transportReady && !micOn;
  // A live mic and the transport are mutually exclusive: scrubbing sustains the
  // synth into your own microphone. Turning the mic off hands the page over.
  const canPlay = hasSource && notBusy && !micOn;
  const canScrub = hasSource && owner !== "model" && !micOn;
  // Who read the audio the transport holds, or null while it holds your voice.
  const clipCredit = clips.find((c) => c.name === playedClip)?.librivox ?? null;
  // The drawing is grey until something has a claim on it: the mic is on, a
  // pre-recorded clip has come back from the model, or you've taken it by hand.
  const tractActive = micOn || viewResponse !== null || manual;
  // At most one box is lit, and it's wherever the interesting thing is: pick an
  // input, watch the tract while it thinks, then the transport takes over and
  // keeps the light for as long as there is something to play — unless the mic
  // is still on, in which case it never left the input box. "tract" lights
  // nothing: the tract has no box, it just moves.
  const section: Section =
    owner === "model" || manual
      ? "tract"
      : micOn
        ? "input"
        : tractActive
          ? "playback"
          : "input";

  return (
    <main className="flex flex-1 flex-wrap items-start gap-6 py-4 md:gap-8 md:p-8">
      {/* Left: everything you operate. Right: the thing you look at. Below `md`
          there is no "right", so both stack and the prose tightens up to leave
          the drawing above the fold. The side padding lives here rather than on
          `main`, so the sections inside can bleed back out of it — see STRIP.

          Uncapped while stacked, or a wide phone would stop the column at 28rem
          and the strips would bleed to *that* edge rather than the screen's;
          the prose carries its own cap instead. From `md` it is a fixed 24rem
          and gives up its share of the row, which is what lets the drawing sit
          beside it at every width past the breakpoint rather than only once
          both are at full size. */}
      <div className="flex w-full min-w-0 flex-1 flex-col items-start gap-3 px-4 text-sm md:w-96 md:flex-none md:gap-4 md:px-0 md:text-base">
        <header>
          <h1 className="text-4xl font-bold text-highlight-600 md:text-5xl">
            Samuel
          </h1>
        </header>
        {/* TODO: write the real intro. */}
        {/* Below `md` the intro is one sentence and a "more"; everything after
            it is in the DOM the whole time and just hidden, so opening it is a
            class flip and nothing has to be measured. At `md` and up there is
            no fold to save and it's all shown, with both controls gone. */}
        <p className="max-w-md text-neutral-600">
          Samuel is a model that learns to control this silly mouth to mimic
          speech.{" "}
          <span className={introOpen ? "" : "hidden md:inline"}>
            Say something and it will parrot after you, or if you&apos;re shy,
            try one of the pre-made clips.
          </span>
          {!introOpen && (
            <button
              onClick={() => setIntroOpen(true)}
              aria-expanded={false}
              className={`md:hidden ${MUTED_LINK}`}
            >
              more
            </button>
          )}
        </p>
        <p
          className={`max-w-md text-neutral-600 ${introOpen ? "" : "hidden md:block"}`}
        >
          The mouth itself is{" "}
          <TextLink href="https://dood.al/pinktrombone/" muted>
            Pink Trombone
          </TextLink>
          , a project originally by Neil Thapen, described by him as
          &quot;bare-handed speech synthesis&quot;.
        </p>
        <p
          className={`max-w-md text-neutral-600 ${introOpen ? "" : "hidden md:block"}`}
        >
          Made by{" "}
          <TextLink href="https://vvolhejn.com">Václav Volhejn</TextLink>. Code{" "}
          <TextLink href="https://github.com/vvolhejn/samuel">
            on GitHub
          </TextLink>
          .
        </p>
        {introOpen && (
          <button
            onClick={() => setIntroOpen(false)}
            aria-expanded
            className={`text-neutral-600 md:hidden ${MUTED_LINK}`}
          >
            less
          </button>
        )}
        {/* Two columns: talk to it on the left, or pick a canned clip on the
            right. They split the row evenly and are allowed to shrink under
            their text, so what governs the width is the buttons (~130px each)
            and the labels wrap over them — which is what keeps these side by
            side down to a 320px screen instead of stacking. */}
        <div
          className={`flex items-center gap-4 md:gap-6 ${STRIP} ${sectionBox(section === "input")}`}
        >
          <div className="flex min-w-0 flex-1 flex-col items-center gap-1.5">
            {/* One button, two labels: it is the same thing you press to start
                and to be done, so it stays in one place and says which. */}
            <button
              onClick={toggleMic}
              disabled={insecure !== null || micStarting}
              title={insecure ? "Unavailable on an insecure origin" : undefined}
              /* Fixed width so the two labels don't resize it, and outlined
                 while recording: the filled pill is an invitation, and there is
                 nothing left to invite once you're being listened to. Both
                 declare a border, or the outlined one would stand 2px taller. */
              className={`w-32 rounded-full border py-1.5 text-sm ${
                micOn
                  ? "border-highlight-300 bg-white font-medium text-highlight-700 hover:bg-highlight-50"
                  : "border-transparent bg-highlight-600 font-semibold text-white hover:bg-highlight-700 disabled:opacity-40 disabled:hover:bg-highlight-600"
              }`}
            >
              {micStarting ? "Starting…" : micOn ? "Stop" : "Microphone"}
            </button>

            {/* Both faces share one grid cell, so the column is always as tall
                as the taller of them and toggling the mic can't move the page.
                The hidden one is `invisible`, which still takes up space but
                drops out of hit-testing and the tab order. */}
            <div className="grid w-full">
              <div
                className={`col-start-1 row-start-1 flex items-center justify-center ${micOn ? "" : "invisible"}`}
                aria-hidden={!micOn}
              >
                <LevelMeter store={levelStore} active={heard} />
              </div>

              {/* Reassurance, not an action: recordings go to the server only
                  to be mimicked back, and nothing is written to disk there.
                  Only the "self-host" escape hatch is clickable. */}
              <span
                className={`col-start-1 row-start-1 text-center text-xs text-neutral-500 ${micOn ? "invisible" : ""}`}
                aria-hidden={micOn}
              >
                Your audio is not stored.
                <br />
                <TextLink href="https://github.com/vvolhejn/samuel" muted>
                  Self-host
                </TextLink>{" "}
                if you don&apos;t trust me
              </span>
            </div>
          </div>

          <div className="flex min-w-0 flex-1 flex-col items-center gap-1.5">
            <span className="text-center text-sm text-neutral-500">
              or use pre-recorded audio
            </span>

            {/* One button per committed clip, numbered rather than named: which
                recording is which only matters once you've heard them. */}
            <div className="grid grid-cols-3 gap-1.5">
              {clips.map((clip, i) => (
                <button
                  key={clip.name}
                  onClick={() => void playClip(clip.name)}
                  disabled={!notBusy || micOn}
                  title={
                    insecure ? "Unavailable on an insecure origin" : undefined
                  }
                  className={
                    clip.name === playedClip
                      ? "rounded-md bg-highlight-600 px-3 py-1 text-sm font-semibold text-white disabled:opacity-40"
                      : /* White like the Play pill, not transparent: these keep
                           their own ground when the box behind them lights
                           up. */
                        "rounded-md border border-highlight-300 bg-white px-3 py-1 text-sm font-medium text-highlight-700 hover:bg-highlight-50 disabled:opacity-40 disabled:hover:bg-white"
                  }
                >
                  {i + 1}
                </button>
              ))}
            </div>
          </div>
        </div>
        {/* Same grey-until-there's-something rule as the tract, done in one
            place: with nothing to play, every accent inside the transport —
            the Play pill, the source switch, the scrubber — desaturates
            together rather than each control needing its own dead colour. A
            live mic greys it too, since the mic holds the page until it's off. */}
        <div
          className={`relative flex flex-col gap-2 ${STRIP} ${sectionBox(section === "playback")} ${
            transportLive ? "" : "grayscale"
          }`}
        >
          {/* Reaching for a dead transport says you're done talking, so treat
              it as the Stop button. Has to be an overlay rather than a click
              handler on the box: the controls underneath are disabled, and a
              disabled control swallows the click instead of bubbling. */}
          {micOn && (
            <button
              onClick={() => void stopMic(true)}
              aria-label="Stop recording to use the playback controls"
              className="absolute inset-0 z-10 md:rounded-xl"
            />
          )}
          {/* Two rows: Play and the bar it drives, then the source switch under
              them. Rows rather than a left column plus a bar, so the bar starts
              right after Play instead of clearing the switch below it. */}
          <div className="flex items-center gap-3">
            <button
              onClick={() => void togglePlay()}
              disabled={!canPlay && !isPlaying}
              /* White, not accent-filled: the solid pill is the microphone's, and
               two of them made the transport shout for a click it doesn't need.
               Explicitly white rather than transparent so it keeps its own
               ground when the box around it lights up. */
              className="w-16 rounded-full border border-highlight-300 bg-white py-1.5 text-sm font-medium text-highlight-700 hover:bg-highlight-50 disabled:opacity-40 disabled:hover:bg-white"
            >
              {isPlaying ? "Pause" : "Play"}
            </button>

            <input
              type="range"
              min={0}
              max={1000}
              value={Math.round(scrubFrac * 1000)}
              disabled={!canScrub}
              aria-label="Scrub through the current audio"
              onPointerDown={() => onScrub(scrubFrac)}
              onChange={(e) => onScrub(Number(e.currentTarget.value) / 1000)}
              onPointerUp={() => void onScrubEnd()}
              onKeyUp={() => void onScrubEnd()}
              onBlur={() => void onScrubEnd()}
              className="min-w-0 flex-1 accent-highlight-600 disabled:opacity-40"
            />
          </div>

          {/* The other take of the same utterance, as an aside rather than a
              switch: the imitation is what you came for, and a lit-up control
              next to Play only competed with it. The clip's credit shares this
              row, pushed to the far end, so crediting costs no height — and it
              sits under the bar it belongs to rather than by the buttons, since
              what it names is the audio now loaded, not the button you pressed. */}
          <div className="flex items-start justify-between gap-3 text-xs text-neutral-500">
            <button
              onClick={() => void playOther()}
              disabled={!original.loaded || !transportLive}
              className={`${MUTED_LINK} shrink-0 disabled:opacity-50 disabled:hover:text-inherit`}
            >
              {source === "samuel" ? "Play original" : "Play imitated"}
            </button>

            {clipCredit && (
              <span className="min-w-0 text-right">
                <TextLink href={clipCredit.url} muted>
                  {clipCredit.title}
                </TextLink>{" "}
                by {clipCredit.author}, read by {clipCredit.reader} (
                <TextLink href="https://librivox.org/" muted>
                  LibriVox
                </TextLink>
                )
              </span>
            )}
          </div>
        </div>
        {insecure && (
          <p className="rounded-lg border border-amber-300 bg-amber-50 p-3 text-sm text-amber-900">
            {insecure}
          </p>
        )}
        {error && <p className="text-sm text-red-600">{error}</p>}
        {/* Dev-only: inlined at build time, so the deployed bundle has no
            debug UI at all. Never on a phone either — it's a development
            affordance, and the screen is needed for the drawing. */}
        {process.env.NODE_ENV === "development" && (
          <div className="hidden md:block">
            <DebugPanel
              open={debugOpen}
              onToggle={() => setDebugOpen((v) => !v)}
              health={health}
              precomputed={precomputed}
              response={viewResponse}
              frac={scrubFrac}
              micProcessing={micProcessing}
              onToggleMicProcessing={(key) => void toggleMicProcessing(key)}
              historyCount={historyCount}
              onDownloadHistory={() => void downloadHistory()}
            />
          </div>
        )}
      </div>

      {/* The element draws a fixed 600×600 — the tract's 600×500 with the
          voicebox strip under it, which is the original Pink Trombone's canvas
          — so TractStage sizes the host to match and scales it down when the
          window is narrower than that. Left out entirely on an insecure origin,
          where it would stay blank: the synth it draws is never started there.
          No box of its own either: a tract that's moving already announces
          itself, and the greyed-out state covers the rest.

          The cap belongs on the inner wrapper, not this one: flexbox breaks
          lines on a size that `max-width` has already clamped, so a 616px cap
          out here is narrow enough to sit beside the column on a ~900px window
          — which squeezes the controls into a strip nobody asked for. Full
          width claims the whole line while stacked, and from `md` this takes
          whatever the fixed column leaves and the drawing scales into it. */}
      {insecure === null && (
        <div className="w-full p-2 md:min-w-0 md:flex-1">
          {/* The drawing never usefully exceeds its own bitmap, so cap it here.
              Left-aligned, not centred: on a wide screen the whole page sits to
              the left and the empty space goes at the end of it. */}
          <div className="w-full max-w-[600px]">
            <TractStage>
              {/* Read-only unless the mouth is yours. The element's drags write
                the same AudioParams our curves automate, and a `.value` write
                doesn't cancel a scheduled curve — so an idle poke used to fight
                the imitation and win for as long as your finger was down. */}
              <pink-trombone
                className="block h-[600px] w-[600px]"
                inactive={tractActive ? undefined : "true"}
                interactive={manual ? undefined : "false"}
              />

              {/* Over the tract rather than beside the buttons: the wait is the one
                moment nothing else on the page moves, and tucked next to the
                controls it was easy to miss. Bounded to the tract's own 500px so
                it doesn't cover the voicebox. */}
              {owner === "model" && (
                <div className="absolute inset-x-0 top-0 z-10 flex h-[500px] items-center justify-center rounded-xl bg-white/70">
                  {/* Solid strip behind the words: the tract is all thin dark lines,
                    and the wash alone doesn't hide enough of them to read over. */}
                  <span className="w-full bg-white py-2 text-center text-2xl text-neutral-600">
                    Thinking...
                  </span>
                </div>
              )}
            </TractStage>

            {/* A plain page control rather than one of the original's pills drawn
              into the canvas: it is chrome, not part of the picture, so it
              belongs to the page's own idiom (and stays focusable). Under the
              drawing, since what it hands you is the drawing. */}
            <div className="flex flex-col items-end gap-1 px-1 pt-2">
              <button
                onClick={toggleManual}
                disabled={micOn}
                title={
                  micOn
                    ? "Turn the microphone off first"
                    : "Play the mouth yourself: drag the voicebox to pitch and voice it, and drag the tract to move the tongue"
                }
                /* The border stays declared but transparent when lit, or the
                 button loses 2px of height and drags the line below it up (the
                 same reason sectionBox keeps its own). */
                className={
                  manual
                    ? "shrink-0 rounded-full border border-transparent bg-highlight-600 px-4 py-1.5 text-sm font-semibold text-white hover:bg-highlight-700"
                    : "shrink-0 rounded-full border border-neutral-300 bg-white px-4 py-1.5 text-sm font-medium text-neutral-600 hover:bg-neutral-50 disabled:opacity-40 disabled:hover:bg-white"
                }
              >
                Manual control
              </button>
              <span className="text-xs text-neutral-500">
                So that you see it&apos;s not easy.
              </span>
            </div>
          </div>
        </div>
      )}
    </main>
  );
}
