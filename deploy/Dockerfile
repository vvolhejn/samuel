# The Python backend (samuel.server) as a Cloudflare Container, sitting behind
# the same Worker that serves the static frontend. See README.md beside this.
#
# The build context is the repo root (see image_build_context in
# webapp/wrangler.jsonc), so the COPY paths below are repo-relative.

FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU wheels only: PyPI's default torch bundles ~2.5 GB of CUDA libraries this
# container can never use. A separate --index-url (not --extra-index-url) so pip
# cannot resolve torch back to the PyPI build.
RUN pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu "torch==2.8.*"

# samuel.server's real import graph. The training deps (transformers, wandb,
# faster-whisper, plotly) are never touched by /api/*, so they stay out — this
# is deliberately not `uv sync`, which would install all of them.
RUN pip install --no-cache-dir \
      einops fastapi librosa omegaconf pydantic soundfile tqdm "uvicorn[standard]"

# Weights come from the Hub rather than the build context, so the image builds
# from a git checkout alone — runs/ is 64 GB and never leaves the cluster. The
# repo is public, so no credentials are needed here.
#
# Pin MODEL_REVISION to a commit: with a branch name, Docker caches this layer
# and a newer checkpoint on the Hub is silently ignored.
ARG MODEL_REPO=vvolhejn/samuel
ARG MODEL_REVISION=main
RUN pip install --no-cache-dir huggingface_hub \
 && python -c "import sys; from huggingface_hub import snapshot_download; \
snapshot_download(sys.argv[1], revision=sys.argv[2], local_dir='/app/model')" \
      "$MODEL_REPO" "$MODEL_REVISION"

# After the weights, since src/ changes more often — an edit here then re-pushes
# only this layer rather than the ~14 MB above it.
#
# Source on PYTHONPATH rather than pip-installed, so server.py's _REPO_ROOT
# (Path(__file__).parents[2]) resolves to /app.
COPY src/ /app/src/

# Match instance_type in webapp/wrangler.jsonc (standard-3 = 2 vCPU). A
# container sees the *host's* core count, not its CPU quota, so torch and numba
# would otherwise start one thread per host core and thrash: on a 128-core host
# capped at 2 CPUs, a 5 s utterance went from 0.9 s to over 2 minutes.
ENV OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    NUMBA_NUM_THREADS=2

# No SAMUEL_ALLOW_ORIGINS: the Worker serves the frontend from this same
# origin, so /api/* is never a cross-origin request.
ENV PYTHONPATH=/app/src \
    SAMUEL_SERVE_FRONTEND=0 \
    SAMUEL_CHECKPOINT=/app/model/checkpoints/last.pt \
    SAMUEL_HOST=0.0.0.0 \
    SAMUEL_PORT=8080

EXPOSE 8080
CMD ["python", "-m", "samuel.server"]
