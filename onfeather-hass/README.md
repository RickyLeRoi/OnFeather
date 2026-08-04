# onfeather-hass

Home Assistant integration for [OnFeather](https://github.com/RickyLeRoi/OnFeather).

Ollama gives Home Assistant a conversation entity. This does the same, plus the
part Ollama has no reason to have: free-tier quota. Assist talks to `of-free`,
which routes each request to whichever provider still has allowance, and the
sensors tell you how much of it is left before you find out the hard way.

```
Home Assistant ──HTTP──▶ of-free :4141 ──▶ Groq / Gemini / Cerebras / …
                                       └──▶ Ollama (local, unmetered)
```

## What it gives you

One config entry, two devices.

**OnFeather Free**

| Entity | State | Notable attributes |
|---|---|---|
| `conversation.onfeather` | last activity | — |
| `sensor.onfeather_free_current_model` | `groq/llama-3.3-70b` | provider, failovers, tokens in/out, latency, served at |
| `sensor.onfeather_free_next_model` | where the next request would go | strategy |
| `sensor.onfeather_free_provider_quota` | % headroom on the healthiest provider | per-provider headroom and rate limits |
| `binary_sensor.onfeather_free_providers_configured` | on when at least one has credentials | per-provider `configured` + the env var it wants |

**OnFeather Solo** — only if `of-solo` is installed alongside the router *and*
has been used at least once. Adding it later needs a reload of the entry.

| Entity | State | Notable attributes |
|---|---|---|
| `sensor.onfeather_solo_memories` | total | proposed / confirmed / rejected |
| `sensor.onfeather_solo_memories_to_review` | memories awaiting review | — |

`of tune` is deliberately absent. It is a one-shot planner with no running state
to report — see [the note below](#why-there-is-no-of-tune-device).

## Requirements

- Home Assistant **2026.7** or newer
- `of-free` reachable from Home Assistant

That second one is the part people get wrong. The router defaults to
`127.0.0.1`, which from another machine is nothing at all:

```bash
of-free serve --host 0.0.0.0 --port 4141
```

Off loopback, set a key as well — otherwise anyone on the network can spend your
quota:

```bash
export ONFEATHER_API_KEY=$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')
of-free serve --host 0.0.0.0
```

The router says which of the two it is doing on start-up.

## Install

Copy `custom_components/onfeather` into your Home Assistant `config/custom_components/`,
restart, then **Settings → Devices & services → Add integration → OnFeather**.

Or add this repository to HACS as a custom repository of type *Integration*.

## Configuring the agent

**Settings → Devices & services → OnFeather → Configure.**

- **Model** — `auto` lets the router choose per request, which is the point of
  the product. `private` forces a local model, so nothing leaves the house.
  Anything else pins one `provider/model` pair.
- **Control Home Assistant** — pick an LLM API here and the agent can act on
  your home. The router will then only route to models that support tool
  calling; it does the filtering itself, so you do not have to know which free
  tiers have it.
- **Turns to remember** — free-tier context windows are small. Older turns are
  dropped before sending, always on a user-message boundary so a tool result is
  never orphaned from the call it answers.

Then set it as the conversation agent of an Assist pipeline in
**Settings → Voice assistants**.

## Two behaviours worth knowing

**Answers arrive whole.** The router does not stream, so in Assist the reply
appears all at once rather than word by word. For text that is invisible. For a
voice pipeline it means speech starts after the model has finished, not during.

**A conversation keeps its model.** Each turn carries the Home Assistant
conversation id in `X-OnFeather-Session`, so the router pins the run to one
model. Without it, a provider running out of quota mid-conversation would hand
the transcript to a different model, which then argues with a stranger.

## Why there is no `of tune` device

`of tune` reads a GGUF file, computes an offload plan, prints it and exits. It
does not run models and holds no state, so there is nothing for a polling
integration to ask it. Reporting "the current model" would mean either
persisting the last plan it computed — a different thing from what is running —
or interrogating `llama-server` directly, which is a job for a `llama.cpp`
integration rather than this one.

## Development

Home Assistant 2026.7 requires **Python 3.14**, so the test harness will not
install on anything older:

```sh
python3.14 -m venv .venv
.venv/bin/pip install pytest-homeassistant-custom-component==0.13.348
.venv/bin/python -m pytest
```

That version pins `homeassistant==2026.7.4`. Bump both together — the harness
tracks HA release for release, and a mismatch fails at import rather than
politely.

## Licence

Apache-2.0, like the rest of OnFeather.
