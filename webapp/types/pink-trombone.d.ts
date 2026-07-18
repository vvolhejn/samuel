// Typings for the vendored Pink Trombone custom element
// (public/pink-trombone/pink-trombone.min.js, see Pink-Trombone/script/component.js).

interface PinkTromboneConstriction {
  index: AudioParam;
  diameter: AudioParam;
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
        >;
      }
    }
  }
}

export type { PinkTromboneElement, PinkTromboneConstriction };
