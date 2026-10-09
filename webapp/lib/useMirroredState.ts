import { useCallback, useRef, useState } from "react";

/** State with a ref that holds the same value, for the values a callback has to
 * read without re-binding every time they change. The setter writes both, so
 * the two cannot drift apart. */
export function useMirroredState<T>(initial: T) {
  const [value, setValue] = useState(initial);
  const ref = useRef(initial);
  const set = useCallback((next: T) => {
    ref.current = next;
    setValue(next);
  }, []);
  return [value, ref, set] as const;
}
