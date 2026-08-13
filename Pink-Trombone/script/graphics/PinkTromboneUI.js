/*
    Modified for the samuel project (GPL-3.0, see Pink-Trombone/LICENSE):
      2026-08-14  size the grid to the drawings it holds — the tract's 600x500
                  over the voicebox's 600x100, which together are the original
                  Pink Trombone's 600x600 canvas — and move the touch buttons
                  off the voicebox row onto the tract, where the original has
                  them. Forward `interactive` alongside `inactive`.

    TODO
        .setFreqeuncyRange(min, max)
*/

import TractUI from "./TractUI.js";
import GlottisUI from "./GlottisUI.js";
import ButtonsUI from "./ButtonsUI.js";

class PinkTromboneUI {
    constructor() {
        this._tractUI = new TractUI();
        this._glottisUI = new GlottisUI();
        this._buttonsUI = new ButtonsUI();

        this._container = document.createElement("div");
            this._container.style.height = "100%";
            this._container.style.width = "100%";

            // Both rows are the intrinsic size of the canvases inside them
            // (TractUI 600x500, GlottisUI 600x100), and there is no gap: the
            // two are one 600x600 picture, as in the original.
            this._container.style.display = "grid";
                this._container.style.gridTemplateRows = "500px 100px";
                this._container.style.gridTemplateColumns = "auto 100px";
                this._container.style.gridRowGap = "0px";

            this._container.appendChild(this._tractUI.node);
                this._tractUI.node.id = "tractUI";
                this._tractUI.node.style.gridColumn = "1 / span 2";
                this._tractUI.node.style.gridRow = "1";

            this._container.appendChild(this._glottisUI.node);
                this._glottisUI.node.id = "glottisUI";
                this._glottisUI.node.style.gridColumn = "1 / span 2"
                this._glottisUI.node.style.gridRow = "2";

            this._container.appendChild(this._buttonsUI.node);
                this._buttonsUI.node.id = "buttonsUI";
                this._buttonsUI.node.style.zIndex = 1;
                this._buttonsUI.node.style.gridColumn = "2";
                this._buttonsUI.node.style.gridRow = "1";
            
            this._container.addEventListener("message", event => {
                event.stopPropagation();
                Array.from(this._container.children).forEach(child => {
                    if(child !== event.target) {
                        child.dispatchEvent(new CustomEvent("message", {
                            detail : event.detail,
                        }));
                    }
                });
            });
    }

    get node() {
        return this._container;
    }

    // Greys the tract out; see TractUI.
    get inactive() {
        return this._tractUI.inactive;
    }
    set inactive(inactive) {
        this._tractUI.inactive = inactive;
        this._glottisUI.inactive = inactive;
    }

    // Whether pointer input drives the synth; see TractUI.
    get interactive() {
        return this._tractUI.interactive;
    }
    set interactive(interactive) {
        this._tractUI.interactive = interactive;
        this._glottisUI.interactive = interactive;
    }

    show() {
        this.node.style.display = "grid";
    }
    hide() {
        this.node.style.display = "none";
    }
}

export default PinkTromboneUI;