# Running OnFeather in Docker

One image carries all three tools — `of-free` (router), `of-solo` (second brain),
`of` (offload planner). They share a Python runtime and a data directory, so
splitting them into three images would triple the pull for isolation none of them
needs: only `of-free serve` listens on a socket.

The image is built by GitHub Actions and published to GitHub Container Registry,
so nothing is compiled on your machine.

## Quick start

```sh
git clone https://github.com/RickyLeRoi/OnFeather.git
cd OnFeather
cp .env.example .env      # provider keys are all optional; ONFEATHER_API_KEY is not
python3 -c 'import secrets; print(secrets.token_urlsafe(32))'   # into ONFEATHER_API_KEY
docker compose up -d
```

That key is the one step you cannot skip. Compose publishes the port on every
interface, and a router with no authentication on a network address is one
anybody on that network can spend your quota with, so it refuses to start
without one.

The router is now on `http://localhost:4141/v1`, OpenAI-compatible:

```sh
curl http://localhost:4141/health
curl http://localhost:4141/v1/models

curl http://localhost:4141/v1/chat/completions \
  -H "authorization: Bearer $ONFEATHER_API_KEY" \
  -H 'content-type: application/json' \
  -d '{"model": "auto", "messages": [{"role": "user", "content": "hello"}]}'
```

Any OpenAI client works by pointing its base URL at `http://localhost:4141/v1`
with `ONFEATHER_API_KEY` as its API key.

## Letting other machines reach the router

Compose publishes `4141` on every host interface, so on a LAN anyone who can
reach the port could spend your free-tier quota. The container therefore
**refuses to start** without a key: `docker compose logs free` says so and the
service stays down.

Set one in `.env` — `env_file` hands it to the container, and `of-free serve`
picks it up on its own:

```sh
python3 -c 'import secrets; print(secrets.token_urlsafe(32))'   # into ONFEATHER_API_KEY
docker compose up -d
docker compose logs free | grep auth        # → auth: API key required
```

Callers then send it the way OpenAI clients already do:

```sh
curl http://localhost:4141/v1/status -H "Authorization: Bearer $ONFEATHER_API_KEY"
```

`/health` stays open regardless, so the container healthcheck keeps passing.

If the router is only ever meant for this machine, narrow the published port to
loopback in `docker-compose.yml` as well. The key stays required — inside the
container `of-free` still binds every interface, and other containers on the
same network still reach it — but the LAN no longer sees the port at all:

```yaml
ports:
  - "127.0.0.1:${ONFEATHER_PORT:-4141}:4141"
```

## Home Assistant

The [Home Assistant integration](HASS.md) is not in this image and cannot be: it
is loaded by Home Assistant from its own `config/custom_components/` directory,
in whatever container or VM Home Assistant runs in. Nothing about it executes
here, and `.dockerignore` keeps `custom_components/` out of the build context.

What this image has to do for it is the section above — be reachable, and want a
key.

## Images and tags

| Tag | What it is |
| --- | --- |
| `latest` | current `main` |
| `v0.0.1`, `0.0` | release tags |
| `sha-abc1234` | one specific commit |

`docker-compose.yml` pulls `ghcr.io/rickyleroi/onfeather:latest` by default.
Pin something stable in `.env`:

```sh
ONFEATHER_TAG=v0.0.1
```

Published for `linux/amd64` and `linux/arm64`. If the package is private, run
`docker login ghcr.io -u <user>` with a token that has `read:packages` first.

To update: `docker compose pull && docker compose up -d`.

## The other two tools

Both run as one-shot commands against the same image and the same volume:

```sh
docker compose run --rm solo of-solo add "Rust is the team default" --type fact
docker compose run --rm solo of-solo list
docker compose run --rm solo of-solo search "rust"

docker compose run --rm tune of inspect /models/qwen3-30b-a3b-q4_k_m.gguf
docker compose run --rm tune of plan /models/qwen3-30b-a3b-q4_k_m.gguf --context 8192
```

Put your `.gguf` files in `./models` (or point `ONFEATHER_MODELS_DIR` elsewhere);
they are mounted read-only at `/models`.

## Where the data lives

Everything persists in the `onfeather-data` volume, mounted at `/data`, which is
the container's `$HOME`:

| Path | What |
| --- | --- |
| `/data/.onfeather/free.db` | quota ledger |
| `/data/.onfeather/solo/` | memories, as plain Markdown |

Read them from the host without a container:

```sh
docker run --rm -v onfeather_onfeather-data:/data -v "$PWD/backup:/backup" \
  alpine tar czf /backup/onfeather.tar.gz -C /data .onfeather
```

To use a host directory instead of a named volume, replace
`onfeather-data:/data` with `./data:/data` in `docker-compose.yml`. The container
runs as uid 1000; `chown 1000:1000 ./data` if your host user is not.

## Ollama on the host

The image ships a registry variant that points Ollama at
`host.docker.internal:11434` instead of `localhost`, and the compose file wires
`host.docker.internal` to the host gateway so it works on Linux too. That is what
`--registry=/opt/onfeather/share/providers-docker.yaml` in the `free` service
command is for — drop it if you don't use Ollama, or point it at a registry file
of your own.

If Ollama runs somewhere else entirely, copy that file, edit `base_url`, and
mount your copy over it.

## `of probe` is measuring the wrong machine

`onfeather-tune` plans an offload split from the hardware it finds. In a
container it finds the container:

- **macOS / Windows** — it sees the Docker VM. Wrong CPU, wrong RAM ceiling, no
  GPU. Planning from it produces a split for a machine you don't own.
- **Linux** — CPU and RAM are read through to the host and are accurate. The GPU
  needs the NVIDIA Container Toolkit and the `deploy.resources` block that is
  commented out in the `tune` service.

The reliable path is to probe on the host and hand the profile to the container:

```sh
of probe -o profile.json                    # on the host, native install
docker cp profile.json onfeather-free:/data/profile.json
docker compose run --rm tune of plan /models/model.gguf --profile /data/profile.json
```

Or just run `onfeather-tune` natively — it has no service to keep alive, and
`pip install onfeather-tune` is the whole setup.

## Building locally

`docker-compose.yml` has no `build:` key on purpose: with one, Compose builds
whenever the image is missing locally instead of pulling it. Building is opt-in
through an overlay file:

```sh
docker compose -f docker-compose.yml -f docker-compose.build.yml up -d --build
```

With the memory embeddings extra (adds lancedb and sentence-transformers,
roughly 2 GB):

```sh
SOLO_EXTRAS=embeddings \
  docker compose -f docker-compose.yml -f docker-compose.build.yml up -d --build

# or without Compose
docker build --build-arg SOLO_EXTRAS=embeddings -t onfeather:embeddings .
```

## Secrets

`.env` is gitignored and read by Compose at `up` time; keys reach the container
as environment variables and are never written to the ledger or logged. On a
shared host prefer Docker secrets or your orchestrator's mechanism over a file
on disk.
