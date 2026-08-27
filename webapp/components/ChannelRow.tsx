import { CHANNEL_COMBINE, CHANNEL_LABELS, type Channel } from "@/lib/tractParams";
import type { ChannelOverride, OverrideMode } from "@/lib/loopScheduler";
import type { MidiEngine } from "@/lib/midi";

/** What each mode means for this channel, said in the channel's own terms —
 * "transpose" and "scale" are the same `offset` mode underneath, but calling
 * it "offset" for pitch would not tell you it keeps the intonation. */
function modeLabel(channel: Channel, mode: OverrideMode): string {
  if (mode === "auto") return "recorded";
  if (mode === "replace") return "replace";
  switch (CHANNEL_COMBINE[channel]) {
    case "semitones":
      return "transpose";
    case "scale":
      return "scale";
    default:
      return "offset";
  }
}

const MODES: OverrideMode[] = ["auto", "offset", "replace"];

/** One channel's handover controls: where its value comes from, and what
 * drives it when that isn't the recording. */
export function ChannelRow({
  channel,
  override,
  midi,
  learning,
  onMode,
  onLearn,
  onClearCc,
}: {
  channel: Channel;
  override: ChannelOverride;
  midi: MidiEngine | null;
  /** This row is waiting for a controller to move. */
  learning: boolean;
  onMode: (mode: OverrideMode) => void;
  onLearn: () => void;
  onClearCc: () => void;
}) {
  const isPitch = channel === "frequency";
  const active =
    override.mode !== "auto" &&
    (isPitch ? (midi?.notes.length ?? 0) > 0 : override.cc !== null);

  return (
    <div data-channel={channel} className="flex items-center gap-2 py-1 text-sm">
      <span
        className={`w-44 shrink-0 truncate ${active ? "font-medium text-highlight-700" : "text-neutral-600"}`}
        title={CHANNEL_LABELS[channel]}
      >
        {CHANNEL_LABELS[channel]}
      </span>

      <div className="flex shrink-0 overflow-hidden rounded-md border border-neutral-200">
        {MODES.map((mode) => (
          <button
            key={mode}
            onClick={() => onMode(mode)}
            className={`px-2 py-0.5 text-xs ${
              override.mode === mode
                ? "bg-highlight-600 font-medium text-white"
                : "bg-white text-neutral-600 hover:bg-highlight-50"
            }`}
          >
            {modeLabel(channel, mode)}
          </button>
        ))}
      </div>

      {/* Pitch is played, not knob-twiddled; the rest need a controller
          assigning before their mode means anything. */}
      {isPitch ? (
        <span className="text-xs text-neutral-500">keyboard</span>
      ) : override.mode === "auto" ? null : (
        <button
          onClick={override.cc === null || learning ? onLearn : onClearCc}
          className={`rounded-md border px-2 py-0.5 text-xs ${
            learning
              ? "border-highlight-300 bg-highlight-50 text-highlight-700"
              : "border-neutral-200 bg-white text-neutral-600 hover:bg-highlight-50"
          }`}
          title={
            override.cc === null
              ? "Assign a controller: click, then move a knob"
              : "Click to unassign"
          }
        >
          {learning
            ? "move a knob…"
            : override.cc === null
              ? "learn CC"
              : `CC ${override.cc}`}
        </button>
      )}
    </div>
  );
}
