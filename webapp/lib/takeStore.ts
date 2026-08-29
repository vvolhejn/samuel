/** Saved takes, in IndexedDB.
 *
 * A take costs a bar of your time and a model round trip, and until now a
 * refresh threw it away. What is stored is the model's answer rather than the
 * audio: it is small (eight channels of about 12 ms frames), it loads with no
 * backend and no wait, and it is exactly what the trajectory is built from.
 *
 * IndexedDB rather than localStorage because the payload is typed arrays, which
 * localStorage can only hold as JSON — several times the size, and lossy about
 * the float32 it started as.
 *
 * Everything here is allowed to fail. Private windows, a full disk and browsers
 * with site data turned off all refuse, and none of that should stop the looper
 * working, so every call resolves to a value the caller can carry on with. */

import type { SynthResponse } from "@/lib/audio";

const DB_NAME = "samuel-looper";
const DB_VERSION = 1;
const STORE = "takes";

/** Newest takes to keep. Older ones are dropped as new ones arrive, so the
 * list stays readable and the storage bounded without anyone tidying up. */
const KEEP = 20;

export interface StoredTake {
  /** Milliseconds since the epoch, which is also the sort order. */
  id: number;
  /** The loop this take was cut for, in seconds at the recorded tempo. */
  loopSeconds: number;
  /** Pad recorded either side of the loop, for the record offset to slide in. */
  padSeconds: number;
  bars: number;
  beatsPerBar: number;
  /** Tempo it was recorded at. A loaded take follows the current tempo, so
   * this is a label rather than something that is restored. */
  bpm: number;
  /** Checkpoint that predicted it (GET /api/health), for telling apart takes
   * that came from different models. */
  modelFingerprint: string | null;
  frameRate: number;
  /** Per channel, as the model predicted it: the loop plus both pads. */
  params: Record<string, Float32Array>;
}

/** What `loopTrajectory` needs, rebuilt from a stored take. */
export function takeResponse(take: StoredTake): SynthResponse {
  const nFrames = take.params.voiceness.length;
  return {
    frame_rate: take.frameRate,
    n_frames: nFrames,
    duration_s: nFrames / take.frameRate,
    params: Object.fromEntries(
      Object.entries(take.params).map(([channel, values]) => [
        channel,
        Array.from(values),
      ]),
    ),
    voiced: [],
  };
}

function open(): Promise<IDBDatabase | null> {
  return new Promise((resolve) => {
    let request: IDBOpenDBRequest;
    try {
      request = indexedDB.open(DB_NAME, DB_VERSION);
    } catch {
      resolve(null);
      return;
    }
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(STORE)) {
        db.createObjectStore(STORE, { keyPath: "id" });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => resolve(null);
    request.onblocked = () => resolve(null);
  });
}

function run<T>(
  mode: IDBTransactionMode,
  body: (store: IDBObjectStore) => IDBRequest,
  fallback: T,
): Promise<T> {
  return open().then(
    (db) =>
      new Promise<T>((resolve) => {
        if (!db) {
          resolve(fallback);
          return;
        }
        let request: IDBRequest;
        try {
          request = body(db.transaction(STORE, mode).objectStore(STORE));
        } catch {
          db.close();
          resolve(fallback);
          return;
        }
        request.onsuccess = () => {
          resolve((request.result as T | undefined) ?? fallback);
          db.close();
        };
        request.onerror = () => {
          resolve(fallback);
          db.close();
        };
      }),
  );
}

/** Every saved take, newest first. */
export async function listTakes(): Promise<StoredTake[]> {
  const takes = await run<StoredTake[]>("readonly", (store) => store.getAll(), []);
  return takes.sort((a, b) => b.id - a.id);
}

/** Save a take and prune the oldest beyond `KEEP`. Resolves to the saved list. */
export async function saveTake(take: StoredTake): Promise<StoredTake[]> {
  await run<null>("readwrite", (store) => store.put(take), null);
  const takes = await listTakes();
  for (const stale of takes.slice(KEEP)) await deleteTake(stale.id);
  return takes.slice(0, KEEP);
}

export async function deleteTake(id: number): Promise<void> {
  await run<null>("readwrite", (store) => store.delete(id), null);
}
