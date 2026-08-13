/*
    Modified for the samuel project (GPL-3.0, see Pink-Trombone/LICENSE):
      2026-08-14  replace the div-and-knob pad with the original Pink
                  Trombone's canvas "voicebox control": the two pale bars, the
                  semitone keyboard, the captions and arrows, and the
                  rounded-corner handle. Ported from the original's
                  Glottis.drawKeyboard / Glottis.handleTouches /
                  TractUI.drawPitchControl (dood.al/pinktrombone, MIT), keeping
                  its coordinates — see WIDTH/HEIGHT/RISE below.

                  Two departures, both noted at their constant: the keyboard is
                  three octaves rather than 20 semitones, and a drag reports
                  itself as a cancelable "voicebox" event before falling back to
                  writing the parameters itself.

    TODO
        throttle value setter
*/

import { COLOR_SCHEMES, FONT_FAMILY } from "./colors.js";

const clamp = (value, min, max) => (value < min ? min : value > max ? max : value);

// The original draws everything on one 600x600 canvas with the keyboard at
// `keyboardTop: 500`, `keyboardHeight: 100`. Here the tract has its own 600x500
// canvas and this is the 600x100 strip below it — so the original's coordinates
// are kept, measured from the top of the strip (y_here = y_original - 500), and
// two of its pieces stick out above that: the "voicebox control" caption (at
// y 490) and the top of the handle at full tenseness (y 495).
//
// Both canvases are therefore RISE px taller and offset up by RISE, overlapping
// the tract's empty bottom margin, and neither takes pointer events: the
// container div — which is exactly the strip — carries the listeners, so a drag
// meant for the tract can't be swallowed by the overlap.
const WIDTH = 600;
const HEIGHT = 100;
const RISE = 25;

// The two stacked bars: the original's drawBar(0.0, 0.4) and, at alpha 0.7,
// drawBar(0.52, 0.72) — fractions of the keyboard height.
const BAR = { top: 0.0, bottom: 0.4, radius: 8 };
const LOWER_BAR = { top: 0.52, bottom: 0.72, radius: 8, alpha: 0.7 };

// The keyboard. The original is `semitones: 20` from `baseNote: 87.3071` (F2),
// i.e. ~90-284 Hz across the full width. The samuel webapp also uses the handle
// as a readout of a pitch trajectory extracted by pyin over 70-500 Hz, which
// would spend whole utterances pinned against the right edge — so this is three
// octaves from C2, spanning 67-539 Hz, which contains that range. For the
// original's exact axis, set SEMITONES = 20 and BASE_NOTE = 87.3071.
const SEMITONES = 36;
const BASE_NOTE = 65.4064; // C2

// Which semitones get the heavier tick: scale degrees 0, 5 and 7 of each octave
// — the original's pattern and phase, which off a C base marks C, F and G.
const MARKS = [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0];
const MARK_OFFSET = 3;

// The handle: the original's 18x30 rectangle (w=9, h=15), whose corners are
// round only because it is stroked with a round lineJoin.
const HANDLE = { width: 9, height: 15, lineWidth: 4 };

// The original's vertical mapping, which does not use the whole strip: the
// pointer is offset 10px up, clamped into HEIGHT-26 and scaled by HEIGHT-28.
// Keeping the numbers keeps the handle inside the bars at both extremes.
const Y_OFFSET = 10;
const Y_CLAMP = HEIGHT - 26;
const Y_SCALE = HEIGHT - 28;

class GlottisUI {
  constructor() {
    this._container = document.createElement("div");
    this._container.style.margin = 0;
    this._container.style.padding = 0;
    // The canvases are absolutely positioned, so the container has to be what
    // they are positioned against — otherwise they escape their grid slot and
    // land on whatever ancestor happens to be positioned.
    this._container.style.position = "relative";

    this._canvases = {};
    this._contexts = {};

    ["voicebox", "voiceboxBackground"].forEach((id, index) => {
      const canvas = document.createElement("canvas");
      canvas.id = id;

      canvas.style.position = "absolute";
      canvas.style.top = `${-RISE}px`;
      canvas.style.left = "0px";
      canvas.width = WIDTH;
      canvas.height = HEIGHT + RISE;
      canvas.style.backgroundColor = "transparent";
      canvas.style.margin = 0;
      canvas.style.padding = 0;
      canvas.style.zIndex = 1 - index;
      // The listeners are on the container; see RISE.
      canvas.style.pointerEvents = "none";

      this._canvases[id] = canvas;
      this._contexts[id] = canvas.getContext("2d");

      this._container.appendChild(canvas);
    });

    this._inactive = false;
    this._colors = COLOR_SCHEMES.active;
    this._interactive = true;
    this._didDrawBackground = false;

    // Last values read back off the AudioParams — where the handle sits when
    // nobody is touching it, which is how it follows automation.
    this._frequency = 140;
    this._tenseness = 0.6;

    // The touch driving the voicebox, and where it put the handle. While a drag
    // is live the handle is drawn at the pointer rather than at the readback:
    // the original does the same, and it keeps the handle from lagging behind
    // any smoothing the page applies to the parameters.
    this._touchIdentifier = null;
    this._handle = null;

    this._alwaysVoice = true;

    // Mouse EventListeners
    this._container.addEventListener("mousedown", (event) => {
      this._startEvent(event, -1);
    });
    this._container.addEventListener("mousemove", (event) => {
      this._moveEvent(event, -1);
    });
    this._container.addEventListener("mouseup", (event) => {
      this._endEvent(event, -1);
    });
    // A drag that leaves the strip would otherwise stay held for good: unlike
    // the tract's handlers there is nothing to release it on the way back in.
    this._container.addEventListener("mouseleave", (event) => {
      this._endEvent(event, -1);
    });

    // Touch EventListeners
    this._container.addEventListener("touchstart", (event) => {
      event.preventDefault();
      Array.from(event.changedTouches).forEach((touch) => this._startEvent(touch, touch.identifier));
    });
    this._container.addEventListener("touchmove", (event) => {
      event.preventDefault();
      Array.from(event.changedTouches).forEach((touch) => this._moveEvent(touch, touch.identifier));
    });
    this._container.addEventListener("touchend", (event) => {
      event.preventDefault();
      Array.from(event.changedTouches).forEach((touch) => this._endEvent(touch, touch.identifier));
    });
    this._container.addEventListener("touchcancel", (event) => {
      event.preventDefault();
      Array.from(event.changedTouches).forEach((touch) => this._endEvent(touch, touch.identifier));
    });

    // The original's "always voice" toggle, relayed by PinkTromboneUI.
    this._container.addEventListener("message", (event) => {
      if (event.detail.type == "toggleButton" && event.detail.parameterName == "voice") {
        this._alwaysVoice = event.detail.newValue == "true";
      }
    });

    // RequestAnimationFrame after being attached to the DOM
    const mutationObserver = new MutationObserver((mutationsList, observer) => {
      if (document.contains(this._container)) {
        this._container.dispatchEvent(
          new CustomEvent("requestAnimationFrame", {
            bubbles: true,
          })
        );

        observer.disconnect();
      }
    });
    mutationObserver.observe(document.body, {
      subtree: true,
      childList: true,
    });

    // AnimationFrame: ask for the two parameters the handle is a picture of.
    // getParameter is answered synchronously, so they have landed by the time
    // the draw below reads them.
    this._container.addEventListener("animationFrame", (event) => {
      ["frequency", "tenseness"].forEach((parameterName) => {
        this._container.dispatchEvent(
          new CustomEvent("getParameter", {
            bubbles: true,
            detail: {
              parameterName: parameterName,
              render: true,
            },
          })
        );
      });

      this._draw();
    });

    this._container.addEventListener("didGetParameter", (event) => {
      if (event.detail.render !== true) return;

      // AudioParam.value reports the automation's current value, so this follows
      // scheduled curves without knowing anything about them.
      if (event.detail.parameterName == "frequency") this._frequency = event.detail.value;
      else if (event.detail.parameterName == "tenseness") this._tenseness = event.detail.value;
    });
  }

  get node() {
    return this._container;
  }

  // Greys out with the tract; see TractUI. The bars and keyboard are drawn once,
  // so the background has to be invalidated to be recoloured.
  get inactive() {
    return this._inactive;
  }
  set inactive(inactive) {
    inactive = Boolean(inactive);
    if (inactive === this._inactive) return;

    this._inactive = inactive;
    this._colors = inactive ? COLOR_SCHEMES.inactive : COLOR_SCHEMES.active;
    this._didDrawBackground = false;
  }

  // Whether a drag may drive the glottis. Off, the strip is a read-only picture
  // of where the parameters are; see TractUI.interactive.
  get interactive() {
    return this._interactive;
  }
  set interactive(interactive) {
    this._interactive = Boolean(interactive);
    if (!this._interactive && this._touchIdentifier !== null) this._release();
  }

  // ---- drawing -------------------------------------------------------------

  _draw() {
    if (!this._didDrawBackground) {
      this._drawBackground();
      this._didDrawBackground = true;
    }

    const context = this._contexts.voicebox;
    context.clearRect(0, 0, WIDTH, HEIGHT + RISE);
    this._drawHandle(context);
  }

  // The bars, the semitone keyboard, the captions and the pitch arrows — the
  // original's Glottis.drawKeyboard, which draws once and never again.
  _drawBackground() {
    const context = this._contexts.voiceboxBackground;
    const top = RISE; // y of the top of the strip on this canvas
    context.clearRect(0, 0, WIDTH, HEIGHT + RISE);
    context.lineCap = context.lineJoin = "round";

    context.strokeStyle = context.fillStyle = this._colors.tongueControl;
    context.globalAlpha = 1;
    this._drawBar(context, top, BAR);
    context.globalAlpha = LOWER_BAR.alpha;
    this._drawBar(context, top, LOWER_BAR);

    context.strokeStyle = context.fillStyle = this._colors.accent;
    const keyWidth = WIDTH / SEMITONES;
    for (let i = 0; i < SEMITONES; i++) {
      const x = (i + 1 / 2) * keyWidth;

      if (MARKS[(i + MARK_OFFSET) % 12] == 1) {
        context.lineWidth = 4;
        context.globalAlpha = 0.4;
      } else {
        context.lineWidth = 3;
        context.globalAlpha = 0.2;
      }
      context.beginPath();
      context.moveTo(x, top + 9);
      context.lineTo(x, top + HEIGHT * BAR.bottom - 9);
      context.stroke();

      context.lineWidth = 3;
      context.globalAlpha = 0.15;
      context.beginPath();
      context.moveTo(x, top + HEIGHT * LOWER_BAR.top + 6);
      context.lineTo(x, top + HEIGHT * LOWER_BAR.bottom - 6);
      context.stroke();
    }

    context.font = `17px ${FONT_FAMILY}`;
    context.textAlign = "center";
    context.globalAlpha = 0.7;
    context.fillText("voicebox control", 300, top - 10); // original: y 490
    context.fillText("pitch", 300, top + 92); // original: y 592

    context.globalAlpha = 0.3;
    context.save();
    context.translate(410, top + 87); // original: y 587
    this._drawArrow(context, 80, 2, 10);
    context.translate(-220, 0);
    context.rotate(Math.PI);
    this._drawArrow(context, 80, 2, 10);
    context.restore();

    context.globalAlpha = 1;
  }

  _drawBar(context, top, bar) {
    const { radius } = bar;
    context.lineWidth = radius * 2;
    context.beginPath();
    context.moveTo(radius, top + HEIGHT * bar.top + radius);
    context.lineTo(WIDTH - radius, top + HEIGHT * bar.top + radius);
    context.lineTo(WIDTH - radius, top + HEIGHT * bar.bottom - radius);
    context.lineTo(radius, top + HEIGHT * bar.bottom - radius);
    context.closePath();
    context.stroke();
    context.fill();
  }

  _drawArrow(context, length, headWidth, headLength) {
    context.lineWidth = 2;
    context.beginPath();
    context.moveTo(-length, 0);
    context.lineTo(0, 0);
    context.lineTo(0, -headWidth);
    context.lineTo(headLength, 0);
    context.lineTo(0, headWidth);
    context.lineTo(0, 0);
    context.closePath();
    context.stroke();
    context.fill();
  }

  // The original's TractUI.drawPitchControl: the handle sits wherever the
  // glottis was last put — by a finger, or here by whatever is automating the
  // parameters.
  _drawHandle(context) {
    const handle = this._handle ?? this._handleFromParameters();
    const x = handle.x;
    const y = handle.y + RISE; // strip coordinates -> this canvas
    const w = HANDLE.width;
    const h = HANDLE.height;

    context.lineCap = context.lineJoin = "round";
    context.lineWidth = HANDLE.lineWidth;
    context.strokeStyle = context.fillStyle = this._colors.accent;
    context.globalAlpha = 0.7;
    context.beginPath();
    context.moveTo(x - w, y - h);
    context.lineTo(x + w, y - h);
    context.lineTo(x + w, y + h);
    context.lineTo(x - w, y + h);
    context.closePath();
    context.stroke();
    context.globalAlpha = 0.15;
    context.fill();
    context.globalAlpha = 1;
  }

  // ---- the two axes --------------------------------------------------------

  _frequencyToX(frequency) {
    const semitone = 12 * Math.log2(Math.max(frequency, 1) / BASE_NOTE);
    return ((semitone - 0.5) * WIDTH) / SEMITONES;
  }

  _xToFrequency(x) {
    const semitone = (SEMITONES * x) / WIDTH + 0.5;
    return BASE_NOTE * Math.pow(2, semitone / 12);
  }

  // The original: tenseness = 1 - cos(t * π/2), for t the height up the strip,
  // so t = acos(1 - tenseness) / (π/2). Inverted exactly rather than
  // approximated, so the handle's height is an honest reading of the parameter.
  _tensenessToY(tenseness) {
    const t = Math.acos(1 - clamp(tenseness, 0, 1)) / (Math.PI * 0.5);
    return (1 - t) * Y_SCALE + Y_OFFSET;
  }

  _yToTenseness(y) {
    const localY = clamp(y - Y_OFFSET, 0, Y_CLAMP);
    const t = clamp(1 - localY / Y_SCALE, 0, 1);
    return 1 - Math.cos(t * Math.PI * 0.5);
  }

  _handleFromParameters() {
    return {
      x: clamp(this._frequencyToX(this._frequency), 0, WIDTH),
      y: this._tensenessToY(this._tenseness),
    };
  }

  // ---- pointer input -------------------------------------------------------

  _eventPosition(event) {
    // The container, not a canvas: it is exactly the strip (the canvases rise
    // above it, see RISE). getBoundingClientRect rather than the
    // offsetTop/offsetLeft the original and TractUI use: it survives the strip
    // being scaled or the page scrolled.
    const box = this._container.getBoundingClientRect();
    return {
      x: ((event.clientX - box.left) * WIDTH) / box.width,
      y: ((event.clientY - box.top) * HEIGHT) / box.height,
    };
  }

  _startEvent(event, touchIdentifier) {
    if (!this._interactive) return;
    if (this._touchIdentifier !== null) return; // one finger owns the voicebox

    this._touchIdentifier = touchIdentifier;
    this._setFromEvent(event, "start");
  }

  _moveEvent(event, touchIdentifier) {
    if (this._touchIdentifier !== touchIdentifier) return;
    this._setFromEvent(event, "move");
  }

  _endEvent(event, touchIdentifier) {
    if (this._touchIdentifier !== touchIdentifier) return;
    this._release();
  }

  _release() {
    this._touchIdentifier = null;
    // Hand the handle back to the readback. It stays where it was left — as the
    // original's does — until something moves the parameters again.
    this._handle = null;
    this._report("end", null, null);
  }

  _setFromEvent(event, phase) {
    const position = this._eventPosition(event);
    const x = clamp(position.x, 0, WIDTH);
    const frequency = this._xToFrequency(x);
    const tenseness = this._yToTenseness(position.y);

    this._handle = {
      x: x,
      y: clamp(position.y - Y_OFFSET, 0, Y_CLAMP) + Y_OFFSET,
    };

    this._report(phase, frequency, tenseness);
  }

  // Report the drag, and set the parameters ourselves only if nobody claimed it.
  // A page that schedules its own automation (as the samuel webapp does) calls
  // preventDefault() and takes over: a `.value` write from here would not cancel
  // its scheduled curves, and the two would fight for the length of the drag.
  _report(phase, frequency, tenseness) {
    const detail = { phase: phase };
    if (frequency !== null) {
      detail.frequency = frequency;
      detail.tenseness = tenseness;
      detail.loudness = Math.pow(tenseness, 0.25);
    }

    const claimed = !this._container.dispatchEvent(
      new CustomEvent("voicebox", {
        bubbles: true,
        cancelable: true,
        detail: detail,
      })
    );
    if (claimed) return;

    if (phase == "end") {
      if (!this._alwaysVoice) this._setParameter("intensity", 0);
      return;
    }
    if (phase == "start" && !this._alwaysVoice) this._setParameter("intensity", 1);

    this._setParameter("frequency", detail.frequency);
    this._setParameter("tenseness", detail.tenseness);
    this._setParameter("loudness", detail.loudness);
  }

  _setParameter(parameterName, newValue) {
    this._container.dispatchEvent(
      new CustomEvent("setParameter", {
        bubbles: true,
        detail: {
          parameterName: parameterName,
          newValue: newValue,
        },
      })
    );
  }
}

export default GlottisUI;
