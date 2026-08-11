# Deploying to Cloudflare, as one Worker

`webapp/wrangler.jsonc` deploys a single Worker that serves the Next.js static
export from `webapp/out/` and routes `/api/*` to the Python backend running as a
Cloudflare Container. Both halves share an origin, so the frontend's `/api/*`
calls stay relative and there is no CORS configuration anywhere.

Requires the Workers Paid plan — containers have no free tier.

## Status

Nothing has been deployed yet. Validated so far: `pnpm build`, the image build,
and the container serving `/api/health` locally (~10 s cold start). Still to do,
in order — delete this section once it is done:

1. Create the Hub model repo and upload the checkpoint (see below). Until then
   the image build fails at the download step with an auth error, because the
   Hub returns 401 rather than 404 for a repo that does not exist.
2. `wrangler deploy` from a machine with Docker.
3. Attach `samuel.vvolhejn.com` as a custom domain.
4. Optionally connect the git repo and check one build log for whether
   Cloudflare's builder has Docker (see "Deploying from git instead").

## Weights

The checkpoint lives in a public Hub repo, `vvolhejn/samuel`, so anyone cloning
this repo can run the app. The image downloads it at build time, which means a
git checkout is all you need to build — `runs/` is 64 GB and never leaves the
cluster — and the build needs no credentials.

Publishing a new checkpoint is two steps, so the write token stays off the
cluster. On the cluster:

```bash
deploy/stage-model.sh <run-dir>          # strips last.pt 42 MB -> 14 MB
```

Then, from a machine that has a Hub write token:

```bash
rsync -a cluster:path/to/samuel/deploy/model/ deploy/model/
cp deploy/model-card.md deploy/model/README.md    # only if you staged before this existed
uv run hf upload vvolhejn/samuel deploy/model .
```

Finally set `MODEL_REVISION` in `webapp/wrangler.jsonc` to the commit that
upload produced. It defaults to `main`, which works but is cache-unsafe: Docker
keys the download layer on the revision string, so a new checkpoint pushed to
the same branch is silently ignored until the pin changes.

`deploy/model/` is gitignored — it is only a staging area for the upload.

## Deploying

```bash
cd webapp
pnpm install
pnpm build          # prebuild vendors Pink Trombone + the VAD assets
pnpm exec wrangler deploy
```

`wrangler` builds the container image with a local Docker daemon and pushes it
to Cloudflare's registry, so this needs Docker on the machine you run it from.
The first deploy pushes the whole ~2 GB image; later ones re-push only changed
layers, and the Dockerfile is ordered so a `src/` edit costs one small layer.

### Deploying from git instead

Because the weights come from the Hub, a builder with a clean checkout has
everything it needs — so autodeploy is possible in principle. The open question
is whether Cloudflare's build machine has a Docker daemon; their docs do not
say. Connecting the repo and reading one build log settles it.

If it does not, build and push the image yourself, then reference the pushed tag
instead of the Dockerfile:

```jsonc
"image": "registry.cloudflare.com/<account-id>/samuel:<tag>"
```

Wrangler then has nothing to build, so CI only uploads the Worker and `out/`.
Frontend changes deploy on push; backend changes become a manual image push plus
a tag bump in this config. Tag by commit SHA rather than `:latest`, or Cloudflare
will not reliably notice a re-push.

## Cost

`instance_type` is `standard-3` (2 vCPU / 8 GiB), matching the thread counts in
the Dockerfile. Containers scale to zero; `sleepAfter` in
`webapp/worker/index.ts` sets how long an idle instance stays awake. Memory and
disk are billed for the whole awake window, CPU only while computing. The
allowance included with the plan covers roughly 3 hours awake at this size; past
that it is about $0.08/hour awake. `max_instances` is 1, since each extra awake
instance multiplies the memory bill.

Cold start measured locally is ~10 s: container boot, `import torch`, then the
checkpoint load.

## Custom domain

`wrangler deploy` publishes to `samuel.<subdomain>.workers.dev`. To serve it at
`samuel.vvolhejn.com`, add that hostname as a custom domain on the Worker
(Cloudflare must be the authoritative DNS for the zone) and it issues the
certificate itself.
