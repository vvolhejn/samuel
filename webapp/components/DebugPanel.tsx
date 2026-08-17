import {
  precomputedMatches,
  HealthResponse,
  PrecomputedIndex,
  SynthResponse,
} from "@/lib/audio";
import { MicProcessing, MIC_PROCESSING_LABELS } from "@/lib/micProcessing";
import { SectionTitle } from "@/components/ui";

const PANEL_PARAMS: Array<{ key: string; label: string; digits: number }> = [
  { key: "frequency", label: "frequency (Hz)", digits: 1 },
  { key: "voiceness", label: "voiceness", digits: 3 },
  { key: "intensity", label: "intensity", digits: 3 },
  { key: "tongueIndex", label: "tongue index", digits: 2 },
  { key: "tongueDiameter", label: "tongue diameter", digits: 2 },
  { key: "constrictionIndex", label: "constriction index", digits: 2 },
  { key: "constrictionDiameter", label: "constriction diameter", digits: 2 },
  { key: "lipDiameter", label: "lip diameter", digits: 2 },
];

/** Exact model-output values at the current playback/scrub position. */
function ModelOutput({
  response,
  frac,
}: {
  response: SynthResponse | null;
  frac: number;
}) {
  const nFrames = response?.n_frames ?? 0;
  const frame = response
    ? Math.min(nFrames - 1, Math.max(0, Math.round(frac * (nFrames - 1))))
    : 0;
  const row = (label: string, value: string) => (
    <div key={label} className="flex items-baseline justify-between gap-3">
      <dt className="text-neutral-500">{label}</dt>
      <dd className="font-mono tabular-nums text-neutral-800">{value}</dd>
    </div>
  );
  return (
    <section>
      <div className="mb-2 flex items-baseline justify-between">
        <SectionTitle>model output</SectionTitle>
        <span className="tabular-nums text-neutral-400">
          {response ? `frame ${frame + 1}/${nFrames}` : "no clip"}
        </span>
      </div>
      <dl className="space-y-1.5">
        {PANEL_PARAMS.map(({ key, label, digits }) =>
          // Older checkpoints predate some params (e.g. lipDiameter).
          row(
            label,
            response?.params[key]
              ? response.params[key][frame].toFixed(digits)
              : "–",
          ),
        )}
        {row(
          "voiced (pyin)",
          response ? (response.voiced[frame] ? "yes" : "no") : "–",
        )}
      </dl>
    </section>
  );
}

/** Everything that is useful while developing but noise while using the thing:
 * which checkpoint is loaded, the live parameter trajectories, and the mic
 * capture settings. Collapsed to a tab on the right by default. */
export function DebugPanel({
  open,
  onToggle,
  health,
  precomputed,
  response,
  frac,
  micProcessing,
  onToggleMicProcessing,
  historyCount,
  onDownloadHistory,
}: {
  open: boolean;
  onToggle: () => void;
  health: HealthResponse | null;
  precomputed: PrecomputedIndex | null;
  response: SynthResponse | null;
  frac: number;
  micProcessing: MicProcessing;
  onToggleMicProcessing: (key: keyof MicProcessing) => void;
  historyCount: number;
  onDownloadHistory: () => void;
}) {
  if (!open) {
    return (
      <button
        onClick={onToggle}
        className="rounded-full border border-neutral-300 px-4 py-1.5 text-sm font-medium text-neutral-600 hover:bg-neutral-50"
      >
        Debug
      </button>
    );
  }
  return (
    <aside className="w-full max-w-md space-y-4 rounded-lg border border-neutral-200 bg-neutral-50 p-3 text-xs">
      <div className="flex items-baseline justify-between">
        <SectionTitle>debug</SectionTitle>
        <button
          onClick={onToggle}
          className="text-neutral-400 hover:text-highlight-600"
        >
          hide
        </button>
      </div>

      <section>
        <div className="mb-1">
          <SectionTitle>checkpoint</SectionTitle>
        </div>
        {health ? (
          <p className="font-mono break-all text-neutral-500">
            {health.checkpoint.startsWith("https://") ? (
              <a
                href={health.checkpoint}
                target="_blank"
                rel="noreferrer"
                className="underline decoration-dotted hover:text-highlight-600"
              >
                {health.checkpoint}
              </a>
            ) : (
              health.checkpoint
            )}
          </p>
        ) : (
          <p className="text-neutral-400">backend unreachable</p>
        )}
      </section>

      <section>
        <div className="mb-1">
          <SectionTitle>precomputed clips</SectionTitle>
        </div>
        {!precomputed ? (
          <p className="text-neutral-500">
            none committed — every clip goes to the backend
          </p>
        ) : precomputedMatches(precomputed, health?.model_fingerprint) ? (
          <p className="font-mono text-neutral-500">
            {precomputed.clips.length} clip(s), {precomputed.model_fingerprint}
          </p>
        ) : (
          <p className="text-red-600">
            stale: made by {precomputed.model_fingerprint}, backend serves{" "}
            {health?.model_fingerprint}. Falling back to the backend — re-run{" "}
            <code>scripts/precompute_clip_responses.py</code>.
          </p>
        )}
      </section>

      <ModelOutput response={response} frac={frac} />

      <section title="Browser mic processing. Training audio has none of it; the server RMS-normalises either way, so these change the shape of the input, not its level.">
        <div className="mb-1">
          <SectionTitle>mic processing</SectionTitle>
        </div>
        <div className="space-y-1 text-neutral-600">
          {MIC_PROCESSING_LABELS.map(({ key, label }) => (
            <label key={key} className="flex items-center gap-1.5">
              <input
                type="checkbox"
                checked={micProcessing[key]}
                onChange={() => onToggleMicProcessing(key)}
                className="accent-highlight-600"
              />
              {label}
            </label>
          ))}
        </div>
      </section>

      <section>
        <div className="mb-1">
          <SectionTitle>session audio</SectionTitle>
        </div>
        <button
          onClick={onDownloadHistory}
          disabled={historyCount === 0}
          className="rounded-full border border-neutral-300 px-3 py-1 text-xs font-medium text-neutral-600 hover:bg-neutral-100 disabled:opacity-40 disabled:hover:bg-transparent"
        >
          {historyCount === 0
            ? "nothing recorded yet"
            : `download ${historyCount} utterance${historyCount === 1 ? "" : "s"}`}
        </button>
      </section>
    </aside>
  );
}
