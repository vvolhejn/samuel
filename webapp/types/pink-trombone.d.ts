// Typings for the vendored Pink Trombone custom element
// (public/pink-trombone/pink-trombone.min.js, see Pink-Trombone/script/component.js).

interface PinkTromboneConstriction {
  index: AudioParam;
  diameter: AudioParam;
}

/** Detail of the element's "voicebox" event: one step of a drag on the glottis
 * pad. Cancelable — calling preventDefault() stops the element from setting the
 * parameters itself, which is what lets us schedule them instead (see
 * GlottisUI._report in the fork). `frequency`/`tenseness` are absent on "end". */
interface VoiceboxEventDetail {
  phase: "start" | "move" | "end";
  frequency?: number;
  tenseness?: number;
  loudness?: number;
}

interface PinkTromboneElement extends HTMLElement {
  setAudioContext(audioContext?: AudioContext): Promise<AudioContext>;
  audioContext: AudioContext;
  connect(destination: AudioNode): void;
  start(): void;
  stop(): void;
  enableUI(): void;
  startUI(): void;

  frequency: AudioParam;
  tenseness: AudioParam;
  intensity: AudioParam;
  loudness: AudioParam;
  tractLength: AudioParam;
  tongue: { index: AudioParam; diameter: AudioParam };
  vibrato: { frequency: AudioParam; gain: AudioParam; wobble: AudioParam };

  newConstriction(index: number, diameter: number): PinkTromboneConstriction;
  removeConstriction(constriction: PinkTromboneConstriction): void;
}

declare global {
  namespace React {
    namespace JSX {
      interface IntrinsicElements {
        "pink-trombone": React.DetailedHTMLProps<
          React.HTMLAttributes<HTMLElement>,
          HTMLElement
        > & {
          /** Present (and not "false") ⇒ the tract is drawn greyed out. */
          inactive?: string;
          /** "false" ⇒ drags on the tract and the voicebox are ignored.
           * Absent ⇒ they drive the synth, as in the original. */
          interactive?: string;
        };
      }
    }
  }
}

export type {
  PinkTromboneElement,
  PinkTromboneConstriction,
  VoiceboxEventDetail,
};
