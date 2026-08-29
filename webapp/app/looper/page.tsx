"use client";

import { useCallback, useEffect, useState, useSyncExternalStore } from "react";
import Link from "next/link";
import { insecureContextMessage } from "@/lib/secureContext";
import { usePinkTrombone } from "@/lib/usePinkTrombone";
import { useLooper, RECORD_OFFSET_LIMIT_MS } from "@/lib/useLooper";
import { useMirroredState } from "@/lib/useMirroredState";
import { noteName } from "@/lib/midi";
import { MIC_PROCESSING_LABELS } from "@/lib/micProcessing";
import { CHANNELS, type Channel } from "@/lib/tractParams";
import type { OverrideMode } from "@/lib/loopScheduler";
import { SectionTitle, TextLink } from "@/components/ui";
import { LevelMeter } from "@/components/LevelMeter";
import { TractStage } from "@/components/TractStage";
import { PhaseBar } from "@/components/PhaseBar";
import { ChannelRow } from "@/components/ChannelRow";
import type { PinkTromboneElement } from "@/types/pink-trombone";

/** Nothing ever invalidates the secure-context snapshot. */
const subscribeNever = () => () => {};

const BAR_CHOICES = [1, 2, 4, 8];
const METRE_CHOICES = [3, 4, 5, 6, 7];

/** The computer keyboard as a one-octave piano, so the looper can be played
 * without a MIDI keyboard plugged in. The usual tracker layout: the home row
 * is the white keys, the row above it the black ones. */
const KEY_NOTES: Record<string, number> = {
  a: 0,
  w: 1,
  s: 2,
  e: 3,
  d: 4,
  f: 5,
  t: 6,
  g: 7,
  y: 8,
  h: 9,
  u: 10,
  j: 11,
  k: 12,
  o: 13,
  l: 14,
  p: 15,
};

export default function Looper() {
  const trombone = usePinkTrombone();
  const looper = useLooper(trombone);
  // Mirrored: the keydown handler is bound once and has to read the current
  // octave without being rebuilt every time it changes.
  const [octave, octaveRef, setOctave] = useMirroredState(4);
  const [learning, setLearning] = useState<Channel | null>(null);

  const insecure = useSyncExternalStore(
    subscribeNever,
    insecureContextMessage,
    () => null,
  );

  const { init, scheduler, midi } = looper;

  // Bring up the synth, then the looper on top of it. Same shape as the main
  // page: the AudioContext exists from page load but stays suspended until the
  // Start gesture resumes it.
  useEffect(() => {
    if (insecureContextMessage()) return;
    const element = document.querySelector<PinkTromboneElement>("pink-trombone");
    if (!element) return;
    trombone
      .init(element)
      .then(() => init())
      .catch((e) => looper.setError(e instanceof Error ? e.message : String(e)));
    // looper.setError is stable; re-running this would build a second synth.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [trombone, init]);

  // The computer keyboard, held-note for held-note like the real thing.
  useEffect(() => {
    if (!midi || !looper.running) return;
    const isTyping = (target: EventTarget | null) =>
      target instanceof HTMLElement &&
      (target.tagName === "INPUT" ||
        target.tagName === "SELECT" ||
        target.isContentEditable);

    const down = (event: KeyboardEvent) => {
      if (event.repeat || event.metaKey || event.ctrlKey) return;
      if (isTyping(event.target)) return;
      if (event.key === "z" || event.key === "x") {
        const shift = event.key === "z" ? -1 : 1;
        setOctave(Math.min(7, Math.max(1, octaveRef.current + shift)));
        return;
      }
      const offset = KEY_NOTES[event.key];
      if (offset === undefined) return;
      event.preventDefault();
      midi.pressNote((octaveRef.current + 1) * 12 + offset);
    };
    const up = (event: KeyboardEvent) => {
      const offset = KEY_NOTES[event.key];
      if (offset === undefined) return;
      midi.releaseNote((octaveRef.current + 1) * 12 + offset);
    };
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    return () => {
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
    };
  }, [midi, looper.running, octaveRef, setOctave]);

  const startLearn = useCallback(
    (channel: Channel) => {
      if (!midi) return;
      setLearning(channel);
      midi.learnNextCc((cc) => {
        looper.setOverride(channel, { cc });
        setLearning(null);
      });
    },
    [midi, looper],
  );

  const phase = useCallback(() => scheduler?.phase() ?? 0, [scheduler]);

  const clock = looper.clock;
  const beats = (clock?.beatsPerBar ?? 4) * (clock?.bars ?? 2);
  const heldNote = midi?.notes[midi.notes.length - 1];

  if (insecure !== null) {
    return (
      <main className="mx-auto max-w-2xl p-8">
        <h1 className="mb-4 text-3xl font-bold text-highlight-600">Looper</h1>
        <p className="text-neutral-600">{insecure}</p>
      </main>
    );
  }

  return (
    <main className="flex flex-1 flex-wrap items-start gap-6 p-4 md:gap-8 md:p-8">
      <div className="flex w-full min-w-0 flex-1 flex-col items-stretch gap-4 md:w-[30rem] md:flex-none">
        <header className="flex items-baseline gap-3">
          <h1 className="text-3xl font-bold text-highlight-600 md:text-4xl">
            Looper
          </h1>
          <Link
            href="/"
            className="text-sm text-neutral-500 underline decoration-dotted underline-offset-2 hover:text-highlight-600"
          >
            back to Samuel
          </Link>
        </header>

        <p className="text-sm text-neutral-600">
          Record a bar of speech; the model turns it into a parameter trajectory
          and the mouth loops it. Then play over the loop: MIDI takes the pitch
          off the recording while the articulation keeps running underneath.
          Needs the backend — see{" "}
          <TextLink href="https://github.com/vvolhejn/samuel" muted>
            the README
          </TextLink>
          .
        </p>

        {looper.error && (
          <p className="rounded-md border border-highlight-300 bg-highlight-50 px-3 py-2 text-sm text-highlight-700">
            {looper.error}
          </p>
        )}

        {/* Transport */}
        <section className="flex flex-col gap-3 rounded-xl border border-neutral-200 p-3">
          <SectionTitle>transport</SectionTitle>
          <div className="flex flex-wrap items-center gap-2">
            <button
              onClick={() => (looper.running ? looper.stop() : void looper.start())}
              className={`w-24 rounded-full border py-1.5 text-sm ${
                looper.running
                  ? "border-highlight-300 bg-white font-medium text-highlight-700 hover:bg-highlight-50"
                  : "border-transparent bg-highlight-600 font-semibold text-white hover:bg-highlight-700"
              }`}
            >
              {looper.running ? "Stop" : "Start"}
            </button>

            <button
              onClick={() =>
                looper.take === "idle" ? looper.record() : looper.cancelRecord()
              }
              disabled={!looper.running || looper.take === "thinking"}
              className={`w-32 rounded-full border py-1.5 text-sm disabled:opacity-40 ${
                looper.take === "recording"
                  ? "border-transparent bg-highlight-600 font-semibold text-white"
                  : looper.take === "armed"
                    ? "border-highlight-300 bg-highlight-50 font-medium text-highlight-700"
                    : "border-neutral-200 bg-white text-neutral-700 hover:bg-highlight-50"
              }`}
            >
              {looper.take === "armed"
                ? "armed…"
                : looper.take === "recording"
                  ? "recording"
                  : looper.take === "thinking"
                    ? "thinking…"
                    : "Record a bar"}
            </button>

            <button
              onClick={looper.clear}
              disabled={!looper.hasLoop}
              className="rounded-full border border-neutral-200 bg-white px-3 py-1.5 text-sm text-neutral-700 hover:bg-highlight-50 disabled:opacity-40"
            >
              Clear
            </button>
            <button
              onClick={looper.resetPhase}
              disabled={!looper.running || clock?.source === "external"}
              title="Move the downbeat to now"
              className="rounded-full border border-neutral-200 bg-white px-3 py-1.5 text-sm text-neutral-700 hover:bg-highlight-50 disabled:opacity-40"
            >
              Resync
            </button>
          </div>

          <PhaseBar
            phase={phase}
            beats={beats}
            armed={looper.take === "armed"}
            recording={looper.take === "recording"}
            pulse={looper.metronomeAudible}
          />

          <div className="flex flex-wrap items-center gap-3 text-xs text-neutral-600">
            <span className="text-neutral-500">click</span>
            <div className="flex overflow-hidden rounded-md border border-neutral-200">
              {(["off", "recording", "always"] as const).map((mode) => (
                <button
                  key={mode}
                  onClick={() => looper.setMetronomeMode(mode)}
                  className={`px-2 py-0.5 ${
                    looper.metronomeMode === mode
                      ? "bg-highlight-600 font-medium text-white"
                      : "bg-white text-neutral-600 hover:bg-highlight-50"
                  }`}
                >
                  {mode === "recording" ? "while recording" : mode}
                </button>
              ))}
            </div>
            <label className="flex items-center gap-1">
              <input
                type="checkbox"
                checked={looper.muteWhileRecording}
                onChange={(e) => looper.setMuteWhileRecording(e.target.checked)}
                className="accent-highlight-600"
              />
              mute the loop while recording
            </label>
          </div>

          <div className="flex items-center justify-between text-xs text-neutral-500">
            <span>
              {looper.hasLoop
                ? `loop: ${looper.takeSeconds?.toFixed(2) ?? "?"} s recorded`
                : "no loop yet"}
            </span>
            {looper.levelStore && (
              <LevelMeter
                store={looper.levelStore}
                active={looper.take === "recording"}
              />
            )}
          </div>
        </section>

        {/* Saved takes */}
        <section className="flex flex-col gap-2 rounded-xl border border-neutral-200 p-3">
          <SectionTitle>takes</SectionTitle>
          {looper.takes.length === 0 ? (
            <p className="text-xs text-neutral-500">
              Every take is saved here as the model&apos;s parameters, in this
              browser, so a refresh does not cost you the loop.
            </p>
          ) : (
            <ul className="flex flex-col gap-1">
              {looper.takes.map((take) => (
                <li
                  key={take.id}
                  className="flex items-center gap-2 text-xs text-neutral-600"
                >
                  <button
                    onClick={() => looper.loadTake(take.id)}
                    className="rounded-full border border-neutral-200 bg-white px-3 py-0.5 text-neutral-700 hover:bg-highlight-50"
                  >
                    Load
                  </button>
                  <span className="tabular-nums">
                    {new Date(take.id).toLocaleTimeString([], {
                      hour: "2-digit",
                      minute: "2-digit",
                    })}
                  </span>
                  <span className="text-neutral-500">
                    {take.bars} × {take.beatsPerBar}/4 ·{" "}
                    {take.bpm.toFixed(0)} bpm · {take.loopSeconds.toFixed(1)} s
                  </span>
                  <button
                    onClick={() => looper.removeTake(take.id)}
                    title="Delete this take"
                    className="ml-auto rounded-full px-2 py-0.5 text-neutral-400 hover:bg-highlight-50 hover:text-highlight-700"
                  >
                    ×
                  </button>
                </li>
              ))}
            </ul>
          )}
        </section>

        {/* Clock */}
        <section className="flex flex-col gap-2 rounded-xl border border-neutral-200 p-3">
          <SectionTitle>clock</SectionTitle>
          <div className="flex flex-wrap items-center gap-3 text-sm">
            <div className="flex overflow-hidden rounded-md border border-neutral-200">
              {(["internal", "external"] as const).map((source) => (
                <button
                  key={source}
                  onClick={() => looper.setClockSource(source)}
                  className={`px-2 py-0.5 text-xs ${
                    clock?.source === source
                      ? "bg-highlight-600 font-medium text-white"
                      : "bg-white text-neutral-600 hover:bg-highlight-50"
                  }`}
                >
                  {source === "internal" ? "internal" : "MIDI clock"}
                </button>
              ))}
            </div>

            {clock?.source === "internal" ? (
              <label className="flex items-center gap-2">
                <input
                  type="range"
                  min={40}
                  max={200}
                  step={1}
                  value={clock?.bpm ?? 120}
                  onChange={(e) => looper.setBpm(Number(e.target.value))}
                  className="w-32 accent-highlight-600"
                />
                <span className="w-16 tabular-nums text-neutral-600">
                  {clock?.bpm ?? 120} bpm
                </span>
              </label>
            ) : (
              <span className="text-xs text-neutral-500">
                {midi?.clockPresent
                  ? `${clock?.effectiveBpm().toFixed(1)} bpm · ${midi.clockRunning ? "running" : "stopped"}`
                  : "waiting for clock…"}
              </span>
            )}
          </div>

          <div className="flex flex-wrap items-center gap-4 text-xs text-neutral-600">
            <label className="flex items-center gap-1">
              bars
              <select
                value={clock?.bars ?? 2}
                onChange={(e) => looper.setShape({ bars: Number(e.target.value) })}
                className="rounded-md border border-neutral-200 bg-white px-1 py-0.5"
              >
                {BAR_CHOICES.map((n) => (
                  <option key={n} value={n}>
                    {n}
                  </option>
                ))}
              </select>
            </label>
            <label className="flex items-center gap-1">
              beats per bar
              <select
                value={clock?.beatsPerBar ?? 4}
                onChange={(e) =>
                  looper.setShape({ beatsPerBar: Number(e.target.value) })
                }
                className="rounded-md border border-neutral-200 bg-white px-1 py-0.5"
              >
                {METRE_CHOICES.map((n) => (
                  <option key={n} value={n}>
                    {n}
                  </option>
                ))}
              </select>
            </label>
            <span className="tabular-nums">
              {clock ? `${clock.loopSeconds().toFixed(2)} s per loop` : ""}
            </span>
          </div>
        </section>

        {/* MIDI */}
        <section className="flex flex-col gap-2 rounded-xl border border-neutral-200 p-3">
          <SectionTitle>midi</SectionTitle>
          {looper.midiSupport === null ? (
            <p className="text-xs text-neutral-500">
              Press Start to ask for MIDI access.
            </p>
          ) : looper.midiSupport !== "ok" ? (
            <p className="text-xs text-neutral-500">
              {midi?.error ?? "MIDI unavailable."} The computer keyboard still
              works.
            </p>
          ) : (
            <label className="flex items-center gap-2 text-xs text-neutral-600">
              input
              <select
                value={midi?.selectedId ?? ""}
                onChange={(e) => looper.selectMidiInput(e.target.value || null)}
                className="min-w-0 flex-1 rounded-md border border-neutral-200 bg-white px-1 py-0.5"
              >
                <option value="">all inputs</option>
                {midi?.inputs.map((input) => (
                  <option key={input.id} value={input.id}>
                    {input.name}
                  </option>
                ))}
              </select>
            </label>
          )}
          <p className="text-xs text-neutral-500">
            Keyboard: <code>a w s e d f…</code> plays, <code>z</code>/
            <code>x</code> shifts octave (now {octave}).{" "}
            {heldNote !== undefined ? (
              <span className="font-medium text-highlight-700">
                holding {noteName(heldNote)}
              </span>
            ) : (
              "nothing held"
            )}
          </p>
        </section>

        {/* Overrides */}
        <section className="flex flex-col gap-1 rounded-xl border border-neutral-200 p-3">
          <SectionTitle>hands on</SectionTitle>
          <p className="pb-1 text-xs text-neutral-500">
            <b>recorded</b> is the model&apos;s trajectory untouched.{" "}
            <b>transpose</b> and <b>scale</b> keep it and move it — the recorded
            intonation survives, which is most of what makes it sound like
            speech. <b>replace</b> throws it away for a value of your own.
          </p>
          {CHANNELS.map((channel) => (
            <ChannelRow
              key={channel}
              channel={channel}
              override={
                scheduler?.overrides[channel] ?? {
                  mode: "auto" as OverrideMode,
                  cc: null,
                  depth: 1,
                }
              }
              midi={midi}
              learning={learning === channel}
              onMode={(mode) => looper.setOverride(channel, { mode })}
              onLearn={() => startLearn(channel)}
              onClearCc={() => looper.setOverride(channel, { cc: null })}
            />
          ))}
          <label className="flex items-center gap-2 pt-2 text-xs text-neutral-600">
            transpose around
            <select
              value={scheduler?.rootNote ?? 60}
              onChange={(e) => looper.setRootNote(Number(e.target.value))}
              className="rounded-md border border-neutral-200 bg-white px-1 py-0.5"
            >
              {Array.from({ length: 25 }, (_, i) => i + 48).map((note) => (
                <option key={note} value={note}>
                  {noteName(note)}
                </option>
              ))}
            </select>
            <span className="text-neutral-500">
              — play this note and the pitch comes out as recorded
            </span>
          </label>
        </section>

        {/* Fiddly bits */}
        <section className="flex flex-col gap-2 rounded-xl border border-neutral-200 p-3">
          <SectionTitle>input</SectionTitle>
          <label className="flex items-center gap-2 text-xs text-neutral-600">
            record offset
            <input
              type="range"
              min={-RECORD_OFFSET_LIMIT_MS}
              max={RECORD_OFFSET_LIMIT_MS}
              step={5}
              value={looper.recordOffsetMs}
              onChange={(e) => looper.setRecordOffsetMs(Number(e.target.value))}
              className="w-32 accent-highlight-600"
            />
            <span className="w-14 tabular-nums">{looper.recordOffsetMs} ms</span>
          </label>
          <p className="text-xs text-neutral-500">
            Nothing reports microphone input latency, so a take lands slightly
            late against the grid. Scrub this until the loop sits on the beat.
            The take is recorded with {RECORD_OFFSET_LIMIT_MS} ms to spare
            either side, so it slides the loop you are hearing — one model
            frame at a time — instead of asking for another take.
          </p>
          <div className="flex flex-wrap gap-3 text-xs text-neutral-600">
            {MIC_PROCESSING_LABELS.map(({ key, label }) => (
              <label key={key} className="flex items-center gap-1">
                <input
                  type="checkbox"
                  checked={looper.micProcessing[key]}
                  onChange={() => looper.toggleMicProcessing(key)}
                  className="accent-highlight-600"
                />
                {label}
              </label>
            ))}
          </div>
          <p className="text-xs text-neutral-500">
            Keep echo cancellation on unless you are wearing headphones — the
            synth is playing into the room the take is recorded in.
          </p>
        </section>
      </div>

      <div className="w-full p-2 md:min-w-0 md:flex-1">
        <div className="w-full max-w-[600px]">
          <TractStage>
            {/* Read-only: the loop owns these parameters, and a drag would
                fight the scheduler for as long as your finger was down. */}
            <pink-trombone
              className="block h-[600px] w-[600px]"
              inactive={looper.running ? undefined : "true"}
              interactive="false"
            />
          </TractStage>
        </div>
      </div>
    </main>
  );
}
