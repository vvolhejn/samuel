import { useEffect, useRef } from "react";

/** The playhead, and the beat grid it runs over.
 *
 * Driven straight off the audio clock on every animation frame and written to
 * the DOM by hand: the loop turns tens of times a minute and the position
 * changes every frame, which is not something React should hear about. */
export function PhaseBar({
  phase,
  beats,
  armed,
  recording,
}: {
  /** Reads the current loop phase in [0, 1). Polled, not pushed. */
  phase: () => number;
  /** Ticks to draw across the bar — beats per bar times bars. */
  beats: number;
  armed: boolean;
  recording: boolean;
}) {
  const headRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let raf = 0;
    const tick = () => {
      const head = headRef.current;
      if (head) {
        head.style.transform = `translateX(${phase() * 100}%)`;
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [phase]);

  const fill = recording
    ? "bg-highlight-600"
    : armed
      ? "bg-highlight-300"
      : "bg-neutral-400";

  return (
    <div className="relative h-6 w-full overflow-hidden rounded-md border border-neutral-200 bg-white">
      {/* Beat ticks. The first is the downbeat and is drawn darker. */}
      {Array.from({ length: Math.max(1, beats) }, (_, i) => (
        <div
          key={i}
          className={`absolute top-0 h-full w-px ${i === 0 ? "bg-neutral-300" : "bg-neutral-200"}`}
          style={{ left: `${(i / Math.max(1, beats)) * 100}%` }}
        />
      ))}
      {/* The head is a full-width element slid across, so the transform is the
          only thing that changes per frame — no layout, no repaint of the
          ticks behind it. */}
      <div className="absolute inset-0">
        <div ref={headRef} className="h-full w-full will-change-transform">
          <div className={`h-full w-0.5 ${fill}`} />
        </div>
      </div>
    </div>
  );
}
