# syntax=docker/dockerfile:1

# One image, three tools. They share a Python runtime and a data directory, and
# splitting them into three images would triple the pull for no isolation that
# matters: none of them is a network service except `of-free serve`.

FROM python:3.12-slim AS builder

# 20260803 ** RG Set to `embeddings` to pull in lancedb + sentence-transformers (adds ~2 GB).
ARG SOLO_EXTRAS=""

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore

RUN python -m venv /opt/onfeather

COPY onfeather-tune/ /src/onfeather-tune/
COPY onfeather-free/ /src/onfeather-free/
COPY onfeather-solo/ /src/onfeather-solo/

RUN /opt/onfeather/bin/pip install \
        /src/onfeather-tune \
        /src/onfeather-free \
        "/src/onfeather-solo${SOLO_EXTRAS:+[$SOLO_EXTRAS]}"

# 20260803 ** RG Ollama lives on the host, so `localhost` inside the container points at nothing.
RUN set -eu; \
    packaged="$(/opt/onfeather/bin/python -c 'import onfeather_free, pathlib; print(pathlib.Path(onfeather_free.__file__).parent / "providers.yaml")')"; \
    mkdir -p /opt/onfeather/share; \
    sed 's#http://localhost:11434/v1#http://host.docker.internal:11434/v1#' \
        "$packaged" > /opt/onfeather/share/providers-docker.yaml; \
    grep -q host.docker.internal /opt/onfeather/share/providers-docker.yaml


FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="OnFeather" \
      org.opencontainers.image.description="of-free (free-tier LLM router), of-solo (second brain), of (llama.cpp offload planner)" \
      org.opencontainers.image.source="https://github.com/RickyLeRoi/OnFeather" \
      org.opencontainers.image.licenses="Apache-2.0"

# 20260803 ** RG uid 1000 so a bind-mounted /data stays writable on a normal Linux host.
RUN useradd --create-home --home-dir /data --uid 1000 onfeather \
 && mkdir -p /models \
 && chown onfeather:onfeather /models

COPY --from=builder /opt/onfeather /opt/onfeather

ENV PATH="/opt/onfeather/bin:$PATH" \
    HOME=/data \
    PYTHONUNBUFFERED=1 \
    ONFEATHER_QUIET=1

USER onfeather
WORKDIR /data

# 20260803 ** RG Ledger and memories live under $HOME/.onfeather; GGUF files are mounted read-only.
VOLUME ["/data", "/models"]

EXPOSE 4141

# 20260803 ** RG No ENTRYPOINT: `docker run <image> of-solo list` has to reach the other two tools.
CMD ["of-free", "serve", "--host", "0.0.0.0", "--port", "4141"]
