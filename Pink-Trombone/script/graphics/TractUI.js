/*
    Modified for the samuel project (GPL-3.0, see Pink-Trombone/LICENSE):
      2026-08-09  restore the original Pink Trombone markings — drop the IPA
                  phoneme labels (the original leaves drawPositions disabled)
                  and draw the anatomical background layer instead; use the
                  webapp's Helvetica Neue stack rather than Arial
      2026-08-09  recolour the tract onto the webapp's accent ramp (was
                  pink/palePink/orchid/#C070C6), plus a grey "inactive"
                  scheme for when the page has no audio input selected
      2026-08-14  move the palette to colors.js (shared with the voicebox);
                  gate pointer input on the new `interactive` flag, so a page
                  that automates the tract can stop drags fighting its curves;
                  hit-test drags off getBoundingClientRect, which works when the
                  canvas isn't at the document origin (see _getEventPosition)

    TODO
        throttle value setter
*/

import { COLOR_SCHEMES, FONT_FAMILY } from "./colors.js";

class TractUI {
  constructor() {
    this._container = document.createElement("div");
    this._container.style.margin = 0;
    this._container.style.padding = 0;

    this._canvases = {};
    this._contexts = {};

    ["tract", "background"].forEach((id, index) => {
      const canvas = document.createElement("canvas");
      canvas.id = id;

      canvas.style.position = "absolute";
      canvas.height = 500;
      canvas.width = 600;
      canvas.style.backgroundColor = "transparent";
      canvas.style.margin = 0;
      canvas.style.padding = 0;
      canvas.style.zIndex = 1 - index;

      this._canvases[id] = canvas;
      this._contexts[id] = canvas.getContext("2d");

      this._container.appendChild(canvas);
    });

    this._canvas = this._canvases.tract;
    this._context = this._contexts.tract;

    this._tract = {
      origin: {
        x: 340,
        y: 460,
      },

      radius: 298,
      scale: 60,

      angle: {
        scale: 0.64,
        offset: -0.25,
      },
    };
    this._processor = null;
    this._parameters = {};

    this._inactive = false;
    this._colors = COLOR_SCHEMES.active;
    // Whether a drag on the tract may move the tongue / add constrictions.
    // Defaults to the original's behaviour: it always could.
    this._interactive = true;
    this._canvases.tract.style.touchAction = "none";

    this._touchConstrictionIndices = [];

    // AnimationFrame
    this._container.addEventListener("animationFrame", (event) => {
      this._container.dispatchEvent(
        new CustomEvent("getProcessor", {
          bubbles: true,
        })
      );

      this._container.dispatchEvent(
        new CustomEvent("getParameter", {
          bubbles: true,
          detail: {
            parameterName: "intensity",
          },
        })
      );
    });

    this._container.addEventListener("didGetProcessor", (event) => {
      this._processor = event.detail.processor;
      this._resize();
      this._drawTract();
    });
    this._container.addEventListener("didGetParameter", (event) => {
      const parameterName = event.detail.parameterName;
      const value = event.detail.value;

      this._parameters[parameterName] = value;
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

    // Mouse EventListeners
    this._canvases.tract.addEventListener("mousedown", (event) => {
      this._startEvent(event);
    });
    this._canvases.tract.addEventListener("mousemove", (event) => {
      this._moveEvent(event);
    });
    this._canvases.tract.addEventListener("mouseup", (event) => {
      this._endEvent(event);
    });

    // Touch EventListeners. preventDefault is what stops a drag on the tract
    // from scrolling the page instead — so it is called only for touches that
    // are actually driving the tract. Read-only (or a touch that began
    // elsewhere), the gesture is left alone and the page scrolls, which on a
    // phone is the only way past a 600px-tall drawing.
    this._canvases.tract.addEventListener("touchstart", (event) => {
      if (!this._interactive) return;
      event.preventDefault();
      Array.from(event.changedTouches).forEach((touch) => this._startEvent(touch));
    });
    this._canvases.tract.addEventListener("touchmove", (event) => {
      const touches = Array.from(event.changedTouches).filter((touch) => this._isDragging(touch));
      if (touches.length === 0) return;
      event.preventDefault();
      touches.forEach((touch) => this._moveEvent(touch));
    });
    const onTouchEnd = (event) => {
      const touches = Array.from(event.changedTouches).filter((touch) => this._isDragging(touch));
      if (touches.length === 0) return;
      event.preventDefault();
      touches.forEach((touch) => this._endEvent(touch));
    };
    this._canvases.tract.addEventListener("touchend", onTouchEnd);
    this._canvases.tract.addEventListener("touchcancel", onTouchEnd);

    // Constriction EventLiteners
    this._canvases.tract.addEventListener("didNewConstriction", (event) => {
      this._touchConstrictionIndices[event.detail.touchIdentifier] = event.detail.constrictionIndex;
    });
    this._canvases.tract.addEventListener("didRemoveConstriction", (event) => {
      this._touchConstrictionIndices[event.detail.touchIdentifier] = undefined;
    });
  }

  get node() {
    return this._container;
  }

  // Greys the tract out (no audio input selected yet). The background canvas is
  // drawn once, so it has to be invalidated for its labels to be recoloured.
  get inactive() {
    return this._inactive;
  }
  set inactive(inactive) {
    inactive = Boolean(inactive);
    if (inactive === this._inactive) return;

    this._inactive = inactive;
    this._colors = inactive ? COLOR_SCHEMES.inactive : COLOR_SCHEMES.active;
    this._didDrawBackground = false;
    if (this._processor) this._drawTract();
  }

  // Whether pointer input drives the tract. Off, the drawing still animates —
  // it just becomes read-only, which is what a page automating the AudioParams
  // wants: a direct `.value` write from a drag does not cancel scheduled
  // curves, so the two fight for as long as the finger is down.
  get interactive() {
    return this._interactive;
  }
  set interactive(interactive) {
    this._interactive = Boolean(interactive);
    // Belt and braces with the preventDefault gating in the touch handlers: with
    // this unset, a browser is free to hand the gesture to the scroller before
    // the drag has been recognised.
    this._canvases.tract.style.touchAction = this._interactive ? "none" : "";
  }

  /** Is this touch one we're already dragging with? */
  _isDragging(touch) {
    return this._touchConstrictionIndices[touch.identifier] !== undefined;
  }

  get width() {
    return this._container.offsetWidth;
  }
  get height() {
    return this._container.offsetHeight;
  }

  _resize() {
    this._resizeCanvases();
  }

  _resizeCanvases() {
    for (let id in this._canvases) {
      //this._canvases[id].style.width = this._container.offsetWidth;
      this._canvases[id].style.height = this._container.offsetHeight;
    }
  }

  _drawTract() {
    if (this._isDrawing) return;

    this._isDrawing = true;

    if (!this._didDrawBackground) {
      this._drawBackground();
      this._didDrawBackground = true;
    }

    this._context = this._contexts.tract;

    this._context.clearRect(0, 0, this._canvas.width, this._canvas.height);
    this._context.lineCap = this._context.lineJoin = "round";

    this._drawTongueControl();

    this._context.beginPath();
    this._context.lineWidth = 2;
    this._context.strokeStyle = this._context.fillStyle = this._colors.tract;
    this._moveTo(1, 0);

    for (let index = 1; index < this._processor.tract.length; index++)
      this._lineTo(index, this._processor.tract.diameter[index]);

    for (let index = this._processor.tract.length - 1; index >= 2; index--) this._lineTo(index, 0);

    this._context.closePath();
    this._context.stroke();
    this._context.fill();

    // NOSE
    const velum = this._processor.tract.nose.diameter[0];
    const velumAngle = velum * 4;

    this._context.beginPath();
    this._context.lineWidth = 2;
    this._context.strokeStyle = this._context.fillStyle = this._colors.tract;
    this._moveTo(this._processor.tract.nose.start, -this._processor.tract.nose.offset);

    for (let index = 1; index < this._processor.tract.nose.length; index++)
      this._lineTo(
        index + this._processor.tract.nose.start,
        -this._processor.tract.nose.offset - this._processor.tract.nose.diameter[index] * 0.9
      );

    for (let index = this._processor.tract.nose.length - 1; index >= 1; index--)
      this._lineTo(index + this._processor.tract.nose.start, -this._processor.tract.nose.offset);

    this._context.closePath();
    this._context.fill();

    this._context.beginPath();
    this._context.lineWidth = 2;
    this._context.strokeStyle = this._context.fillStyle = this._colors.tract;
    this._moveTo(this._processor.tract.nose.start - 2, 0);
    this._lineTo(this._processor.tract.nose.start, -this._processor.tract.nose.offset);
    this._lineTo(this._processor.tract.nose.start + velumAngle, -this._processor.tract.nose.offset);
    this._lineTo(this._processor.tract.nose.start + velumAngle - 2, 0);
    this._context.closePath();
    this._context.stroke();
    this._context.fill();

    this._context.fillStyle = this._colors.innerLabel;
    this._context.font = `20px ${FONT_FAMILY}`;
    this._context.textAlign = "center";
    this._context.globalAlpha = 1;

    this._drawText(this._processor.tract.length * 0.1, 0.425, "throat", false, false);
    this._drawText(this._processor.tract.length * 0.71, -1.8, "nasal", false, false);
    this._drawText(this._processor.tract.length * 0.71, -1.3, "cavity", false, false);

    this._context.font = `22px ${FONT_FAMILY}`;
    this._drawText(this._processor.tract.length * 0.6, 0.9, "oral", false, false);
    this._drawText(this._processor.tract.length * 0.7, 0.9, "cavity", false, false);

    this._drawAmplitudes();

    this._context.beginPath();
    this._context.lineWidth = 5;
    this._context.strokeStyle = this._colors.wall;
    this._context.lineJoin = this._context.lineCap = "round";
    this._moveTo(1, this._processor.tract.diameter[0]);
    for (let index = 2; index < this._processor.tract.length; index++)
      this._lineTo(index, this._processor.tract.diameter[index]);

    this._moveTo(1, 0);
    for (let index = 2; index <= this._processor.tract.nose.start - 2; index++) this._lineTo(index, 0);

    this._moveTo(this._processor.tract.nose.start + velumAngle - 2, 0);
    for (
      let index = this._processor.tract.nose.start + Math.ceil(velumAngle) - 2;
      index < this._processor.tract.length;
      index++
    )
      this._lineTo(index, 0);

    this._context.stroke();

    this._context.beginPath();
    this._context.lineWidth = 5;
    this._context.strokeStyle = this._colors.wall;
    this._context.lineJoin = "round";

    this._moveTo(this._processor.tract.nose.start, -this._processor.tract.nose.offset);
    for (let index = 1; index < this._processor.tract.nose.length; index++)
      this._lineTo(
        index + this._processor.tract.nose.start,
        -this._processor.tract.nose.offset - this._processor.tract.nose.diameter[index] * 0.9
      );

    this._moveTo(this._processor.tract.nose.start + velumAngle, -this._processor.tract.nose.offset);
    for (let index = Math.ceil(velumAngle); index < this._processor.tract.nose.length; index++)
      this._lineTo(index + this._processor.tract.nose.start, -this._processor.tract.nose.offset);

    this._context.stroke();

    this._context.globalAlpha = velum * 5;
    this._context.beginPath();
    this._moveTo(this._processor.tract.nose.start - 2, 0);
    this._lineTo(this._processor.tract.nose.start, -this._processor.tract.nose.offset);
    this._lineTo(this._processor.tract.nose.start + velumAngle, -this._processor.tract.nose.offset);
    this._lineTo(this._processor.tract.nose.start + velumAngle - 2, 0);
    this._context.stroke();

    this._context.fillStyle = this._colors.accent;
    this._context.font = `20px ${FONT_FAMILY}`;
    this._context.textAlign = "center";
    this._context.globalAlpha = 0.7;
    this._drawText(
      this._processor.tract.length * 0.95,
      0.8 + 0.8 * this._processor.tract.diameter[this._processor.tract.length - 1],
      " lip",
      false,
      false
    );

    this._context.globalAlpha = 1;
    this._context.fillStyle = "black";
    this._context.textAlign = "left";

    this._isDrawing = false;
  }

  // Static anatomical labels, drawn once onto the background canvas.
  // Mirrors drawBackground() in the original Pink Trombone.
  _drawBackground() {
    this._context = this._contexts.background;
    // Cleared because the scheme can change after the first draw.
    this._context.clearRect(0, 0, this._canvases.background.width, this._canvases.background.height);

    const length = this._processor.tract.length;

    this._context.fillStyle = this._colors.accent;
    this._context.font = `20px ${FONT_FAMILY}`;
    this._context.textAlign = "center";
    this._context.globalAlpha = 0.7;

    this._drawText(length * 0.44, -0.28, "soft", false, false);
    this._drawText(length * 0.51, -0.28, "palate", false, false);
    this._drawText(length * 0.77, -0.28, "hard", false, false);
    this._drawText(length * 0.84, -0.28, "palate", false, false);
    this._drawText(length * 0.95, -0.28, " lip", false, false);

    this._context.font = `17px ${FONT_FAMILY}`;
    this._drawText(length * 0.18, 3, "  tongue control", true, false);

    this._context.textAlign = "left";
    this._drawText(length * 1.03, -1.07, "nasals", false, false);
    this._drawText(length * 1.03, -0.28, "stops", false, false);
    this._drawText(length * 1.03, 0.51, "fricatives", false, false);

    this._context.strokeStyle = this._colors.accent;
    this._context.lineWidth = 2;
    this._context.beginPath();
    this._strokeTo(length * 1.03, 0, true);
    this._strokeTo(length * 1.07, 0, false);
    this._strokeTo(length * 1.03, -this._processor.tract.nose.offset, true);
    this._strokeTo(length * 1.07, -this._processor.tract.nose.offset, false);
    this._context.stroke();

    this._context.globalAlpha = 1;
    this._context = this._contexts.tract;
  }

  // Like _moveTo/_lineTo but without the wobble, for static geometry.
  _strokeTo(index, diameter, moveTo) {
    const angle = this._getAngle(index);
    const radius = this._getRadius(index, diameter);
    const x = this._getX(angle, radius);
    const y = this._getY(angle, radius);

    if (moveTo) this._context.moveTo(x, y);
    else this._context.lineTo(x, y);
  }

  _drawCircle(index, diameter, arcRadius) {
    const angle = this._getAngle(index);
    const radius = this._getRadius(index, diameter);

    this._context.beginPath();
    this._context.arc(this._getX(angle, radius), this._getY(angle, radius), arcRadius, 0, 2 * Math.PI);
    this._context.fill();
  }
  _drawTongueControl() {
    this._context.lineCap = this._context.lineJoin = "round";
    this._context.strokeStyle = this._context.fillStyle = this._colors.tongueControl;
    this._context.globalAlpha = 1.0;
    this._context.beginPath();
    this._context.lineWidth = 45;

    this._moveTo(this._processor.tract.tongue.range.index.minValue, this._processor.tract.tongue.diameter.minValue); // diameter/2?
    for (
      let index = this._processor.tract.tongue.range.index.minValue + 1;
      index <= this._processor.tract.tongue.range.maxValue;
      index++
    ) {
      this._lineTo(index, this._processor.tract.tongue.range.diameter.minValue);
    }
    this._lineTo(this._processor.tract.tongue.range.index.center, this._processor.tract.tongue.range.diameter.maxValue);
    this._context.closePath();
    this._context.stroke();
    this._context.fill();

    this._context.fillStyle = this._colors.accent;
    this._context.globalAlpha = 0.3;

    [0, -4.25, -8.5, 4.25, 8.5, -6.1, 6.1, 0, 0].forEach((indexOffset, _index) => {
      const diameter =
        _index < 5
          ? this._processor.tract.tongue.range.diameter.minValue
          : _index < 8
          ? this._processor.tract.tongue.range.diameter.center
          : this._processor.tract.tongue.range.diameter.maxValue;

      indexOffset *= this._processor.tract.length / 44;

      this._drawCircle(this._processor.tract.tongue.range.index.center + indexOffset, diameter, 3);
    });

    const tongueAngle = this._getAngle(this._processor.tract.tongue.index);
    const tongueRadius = this._getRadius(this._processor.tract.tongue.index, this._processor.tract.tongue.diameter);

    this._context.lineWidth = 4;
    this._context.strokeStyle = this._colors.accent;
    this._context.globalAlpha = 0.7;
    this._context.beginPath();
    this._context.arc(this._getX(tongueAngle, tongueRadius), this._getY(tongueAngle, tongueRadius), 18, 0, 2 * Math.PI);
    this._context.stroke();
    this._context.globalAlpha = 0.15;
    this._context.fill();
    this._context.globalAlpha = 1;
    this._context.fillStyle = this._colors.accent;
  }
  _drawAmplitudes() {
    this._context.strokeStyle = this._colors.accent;
    this._context.lineCap = "butt";
    this._context.globalAlpha = 0.3;

    for (let index = 2; index < this._processor.tract.length - 1; index++) {
      this._context.beginPath();
      this._context.lineWidth = Math.sqrt(this._processor.tract.amplitude.max[index]) * 3;

      this._moveTo(index, 0);
      this._lineTo(index, this._processor.tract.diameter[index]);

      this._context.stroke();
    }

    for (let index = 1; index < this._processor.tract.nose.length - 1; index++) {
      this._context.beginPath();
      this._context.lineWidth = Math.sqrt(this._processor.tract.nose.amplitude.max[index]) * 3;

      this._moveTo(this._processor.tract.nose.start + index, -this._processor.tract.nose.offset);
      this._lineTo(
        this._processor.tract.nose.start + index,
        -this._processor.tract.nose.offset - this._processor.tract.nose.diameter[index] * 0.9
      );

      this._context.stroke();
    }

    this._context.globalAlpha = 1;
  }
  _drawText(index, diameter, text, isStraight = true, normalize = true) {
    if (normalize) {
      index *= this._processor.tract.length / 44;
    }
    const angle = this._getAngle(index);
    const radius = this._getRadius(index, diameter);

    this._context.save();
    this._context.translate(this._getX(angle, radius), this._getY(angle, radius) + 2);

    if (!isStraight) this._context.rotate(angle - Math.PI / 2);

    this._context.fillText(text, 0, 0);
    this._context.restore();
  }
  _moveTo(index, diameter) {
    this.__to(index, diameter, true);
  }
  _lineTo(index, diameter) {
    this.__to(index, diameter, false);
  }
  __to(index, diameter, moveTo = true) {
    const wobble = this._getWobble(index);
    const angle = this._getAngle(index, diameter) + wobble;
    const radius = this._getRadius(index, diameter) + 100 * wobble;

    const x = this._getX(angle, radius);
    const y = this._getY(angle, radius);

    if (moveTo) this._context.moveTo(x, y);
    else this._context.lineTo(x, y);
  }

  _getX(angle, radius) {
    return this._tract.origin.x - radius * Math.cos(angle);
  }
  _getY(angle, radius) {
    return this._tract.origin.y - radius * Math.sin(angle);
  }

  _getAngle(index) {
    const angle =
      this._tract.angle.offset + (index * this._tract.angle.scale * Math.PI) / (this._processor.tract.lip.start - 1);
    return angle;
  }
  _getWobble(index) {
    var wobble =
      this._processor.tract.amplitude.max[this._processor.tract.length - 1] +
      this._processor.tract.nose.amplitude.max[this._processor.tract.nose.length - 1];
    wobble *= (0.03 * Math.sin(2 * index - 50 * (Date.now() / 1000)) * index) / this._processor.tract.length;
    return wobble;
  }
  _getRadius(index, diameter) {
    var radius = this._tract.radius - this._tract.scale * diameter;

    return radius;
  }

  _getIndex(x, y) {
    var angle = Math.atan2(y, x);
    while (angle > 0) angle -= 2 * Math.PI;

    const index =
      ((Math.PI + angle - this._tract.angle.offset) * (this._processor.tract.lip.start - 1)) /
      (this._tract.angle.scale * Math.PI);
    return index;
  }
  _getDiameter(x, y) {
    const diameter = (this._tract.radius - Math.sqrt(Math.pow(x, 2) + Math.pow(y, 2))) / this._tract.scale;
    return diameter;
  }

  _isNearTongue(index, diameter) {
    var isTongue = true;
    isTongue =
      isTongue &&
      this._processor.tract.tongue.range.index.minValue - 4 <= index &&
      index <= this._processor.tract.tongue.range.index.maxValue + 4;
    isTongue =
      isTongue &&
      this._processor.tract.tongue.range.diameter.minValue - 0.5 <= diameter &&
      diameter <= this._processor.tract.tongue.range.diameter.maxValue + 0.5;
    return isTongue;
  }

  // Pointer position in canvas pixels, relative to the tract's origin.
  //
  // Was `event.pageX - event.target.offsetLeft`, carried over from the original,
  // where the canvas is the whole page: pageX is measured from the document but
  // offsetLeft only from the offsetParent, so the two only cancel when the canvas
  // sits at the document origin. Anywhere else on a page — the samuel webapp puts
  // the tract in a column beside its controls — every drag landed hundreds of
  // pixels away and hit nothing. getBoundingClientRect is measured from the
  // viewport, like clientX, so it also subsumes the CSS-to-bitmap ratio (the
  // `scalar` this used to keep) and survives scrolling.
  _getEventPosition(event) {
    const canvas = this._canvases.tract;
    const box = canvas.getBoundingClientRect();
    const x = ((event.clientX - box.left) * canvas.width) / box.width - this._tract.origin.x;
    const y = ((event.clientY - box.top) * canvas.height) / box.height - this._tract.origin.y;

    return {
      index: this._getIndex(x, y),
      diameter: this._getDiameter(x, y),
    };
  }

  _setTongue(event, position) {
    Object.keys(position).forEach((parameterNameSuffix) => {
      event.target.dispatchEvent(
        new CustomEvent("setParameter", {
          bubbles: true,
          detail: {
            parameterName: "tongue." + parameterNameSuffix,
            newValue: position[parameterNameSuffix],
          },
        })
      );
    });
  }

  _startEvent(event) {
    if (!this._interactive) return; // read-only: no drag is ever begun, so
    // _moveEvent/_endEvent find nothing in _touchConstrictionIndices and fall
    // through on their own.
    const touchIdentifier = event instanceof Touch ? event.identifier : -1;
    if (this._touchConstrictionIndices[touchIdentifier] == undefined) {
      const position = this._getEventPosition(event);
      const isNearTongue = this._isNearTongue(position.index, position.diameter);
      if (isNearTongue) {
        this._touchConstrictionIndices[touchIdentifier] = -1;
        this._setTongue(event, position);
      } else {
        event.target.dispatchEvent(
          new CustomEvent("newConstriction", {
            bubbles: true,
            detail: {
              touchIdentifier: touchIdentifier,
              index: position.index,
              diameter: position.diameter,
            },
          })
        );
      }
    }
  }
  _moveEvent(event) {
    const touchIdentifier = event instanceof Touch ? event.identifier : -1;

    if (this._touchConstrictionIndices[touchIdentifier] !== undefined) {
      const position = this._getEventPosition(event);
      const constrictionIndex = this._touchConstrictionIndices[touchIdentifier];
      const isTongue = constrictionIndex == -1;

      if (isTongue) {
        this._setTongue(event, position);
      } else {
        event.target.dispatchEvent(
          new CustomEvent("setConstriction", {
            bubbles: true,
            detail: {
              constrictionIndex: constrictionIndex,
              index: position.index,
              diameter: position.diameter,
            },
          })
        );
      }
    }
  }
  _endEvent(event) {
    const touchIdentifier = event instanceof Touch ? event.identifier : -1;

    if (this._touchConstrictionIndices[touchIdentifier] !== undefined) {
      const constrictionIndex = this._touchConstrictionIndices[touchIdentifier];
      const isTongue = constrictionIndex == -1;

      if (isTongue) {
        // do nothing
      } else {
        event.target.dispatchEvent(
          new CustomEvent("removeConstriction", {
            bubbles: true,
            detail: {
              constrictionIndex: constrictionIndex,
              touchIdentifier: touchIdentifier,
            },
          })
        );
      }

      this._touchConstrictionIndices[touchIdentifier] = undefined;
    }
  }
}

export default TractUI;
