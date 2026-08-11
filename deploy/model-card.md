# samuel

A neural controller for [Pink Trombone](https://dood.al/pinktrombone/), a
physical model of the human vocal tract. Given speech, it predicts the vocal
tract parameter trajectories that make the synthesizer imitate it.

Code, training setup, and the webapp that runs this checkpoint:
**https://github.com/vvolhejn/samuel**

## Files

- `checkpoints/last.pt` — model weights (optimizer state stripped)
- `config.json` — the run config; `samuel.server` reads its `model` block to
  rebuild the architecture

## Usage

The checkpoint is not standalone — it needs the model definition and the
synthesizer from the repo above. See its README.
