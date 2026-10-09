import { useEffect, useRef, useState } from "react";

/** The side of the square bitmap the element draws into. */
const TRACT_PX = 600;

/** Scales the drawing down to whatever width it's given, since the element
 * itself only ever draws its fixed bitmap. A transform rather than a smaller
 * canvas: both hit-tests measure getBoundingClientRect, so drags follow the
 * scale for free. The wrapper carries the scaled height, or the page would
 * reserve the full 600px under it. */
export function TractStage({ children }: { children: React.ReactNode }) {
  const [scale, setScale] = useState(1);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const element = ref.current;
    if (!element) return;
    const observer = new ResizeObserver(([entry]) => {
      setScale(Math.min(1, entry.contentRect.width / TRACT_PX));
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  return (
    <div
      ref={ref}
      className="w-full overflow-hidden"
      style={{ height: TRACT_PX * scale }}
    >
      <div
        className="relative origin-top-left"
        style={{
          width: TRACT_PX,
          height: TRACT_PX,
          transform: `scale(${scale})`,
        }}
      >
        {children}
      </div>
    </div>
  );
}
