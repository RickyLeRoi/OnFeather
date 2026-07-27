# OnFeather Free — `of-free`

**Aggregate free LLM tiers behind one quota-aware router.**

Groq, Google AI Studio, Cerebras, Mistral, GitHub Models and OpenRouter all run
real free tiers. Between them there is a useful amount of daily capacity — but
each has its own limits, its own reset clock, and its own idea of what a rate
limit header is called. Using them together by hand means tracking all of it in
your head and finding out you were wrong when a 429 lands.

`of-free` keeps the books, and routes each request to whichever provider can
still serve it.

## Status

Pre-alpha. The registry, quota ledger, router and OpenAI-compatible endpoint
work and are tested, including tool calling and schema-constrained answers.
Streaming is the significant thing still missing.

| Command | Status | What it does |
|---|---|---|
| `of-free status` | ✅ working | Remaining quota per provider, per window |
| `of-free route` | ✅ working | Where a request would go right now, and the fallbacks |
| `of-free chat` | ✅ working | Send a prompt, failing over across providers automatically |
| `of-free providers` | ✅ working | The registry, and what still needs a key |
| `of-free serve` | ✅ working | OpenAI-compatible endpoint, so any client can point at it |

## Isn't this LiteLLM?

LiteLLM overlaps, and it now has free-tier budget tracking of its own — you can
create a budget with `rpm_limit` and `tpm_limit` and attach keys to it. If you
already run LiteLLM as a proxy, that may be all you need.

What is different here:

- **The registry ships filled in.** LiteLLM's budgets are limits *you* configure.
  Here the published limits for every provider are checked in as version-controlled
  YAML with a `verified_at` date, so a fresh install already knows that Groq
  allows 30 requests a minute and Google's flagship allows 2.
- **Limits are hints, not configuration.** The ledger reconciles against the
  provider's own `x-ratelimit-remaining-*` headers, and a 429 overrides
  everything. A stale entry in the registry costs a little routing efficiency
  for one window, then reality corrects it.
- **Reset clocks are modelled properly.** Groq resets at UTC midnight, Google at
  Pacific midnight, per-minute limits roll continuously. Getting this wrong means
  believing a quota has refilled up to eight hours before it has.
- **It is a CLI, not a service.** No deployment, no config file to write.

## One account per provider

Every provider's terms permit programmatic use of their free tier. All of them
forbid creating multiple accounts to multiply your limits — that gets the
accounts and usually the IP banned.

This tool takes the first path only. Each provider maps to exactly one
credential, read from one environment variable, and there is no mechanism to
register a second. Aggregating *across* providers is the entire point;
aggregating across sock puppets is somebody else's project.

## Install

```bash
git clone https://github.com/RickyLeRoi/onfeather-free
cd onfeather-free
pip install -e .
```

Set whichever keys you have. Every one is optional, and the tool works with none
of them as long as Ollama is running:

```bash
cp .env.example .env    # then fill in what you have
```

`.env` is gitignored, and an exported shell variable always wins over it so you
can override one key for one command. Keys are never logged, never written to
the ledger, and never printed — `of-free status` reports whether a provider is
configured, not what with.

## Usage

```console
$ of-free status
PROVIDER              STATUS        HEADROOM    LIMITS
Groq                  ready         100%        req/min 30/30, req/day 14400/14400, tok/min 100000/100000
Google AI Studio      ready         73%         req/min 15/15, req/day 1104/1500, tok/min 1000000/1000000
Ollama (local)        ready         100%        —

* = confirmed by the provider's own rate-limit headers
```

```console
$ of-free route --strategy fast
→ Groq  llama-3.3-70b-versatile
  fastest available, 100% quota left
  https://api.groq.com/openai/v1

Fallbacks:
  Google AI Studio    gemini-2.0-flash        73% left
  Ollama (local)      qwen2.5:7b             100% left
```

```console
$ of-free chat "explain a monad in one sentence" -v
  → Groq / llama-3.3-70b-versatile  12+38 tok  0.41s

A monad is a way to chain computations that carry context…
```

When a provider rate-limits you mid-request, the failover is visible rather than
silent:

```console
$ of-free chat "..." -v
  ✗ groq: rate limited
  → Google AI Studio / gemini-2.0-flash  12+44 tok  1.02s
```

### As an endpoint

```console
$ of-free serve
onfeather-free listening on http://127.0.0.1:4141/v1
  strategy: balanced
  routable: 13 provider/model pairs

  export OPENAI_BASE_URL=http://127.0.0.1:4141/v1
  export OPENAI_API_KEY=unused
```

Point any OpenAI-compatible client at it and routing happens underneath:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:4141/v1", api_key="unused")
client.chat.completions.create(model="auto", messages=[...])
```

Three model names are special:

| `model` | Behaviour |
|---|---|
| `auto` | Route across everything available |
| `private` | Local only — never leaves the machine |
| `provider/model` | Ask for one explicitly |

Every response carries `X-OnFeather-Provider`, `X-OnFeather-Model` and
`X-OnFeather-Failovers`, so you can always see what actually served you.
`GET /v1/status` returns the full quota picture as JSON.

Built on the standard library — no framework, no `uvicorn`, nothing to deploy.
Streaming is not supported yet and is refused with an explicit 400 rather than a
response a client cannot parse.

### Agents, tools and schemas

An agentic client is a harder customer than a chat box, and the endpoint is
built for one. [`docs/agentic-clients.md`](docs/agentic-clients.md) is the full
contract; the short version:

- **`tools` and `tool_choice` pass through**, and only models that actually
  support function calling are routed to. `arguments` always comes back as a
  JSON *string* and every tool call has an `id`, whichever provider served it —
  two things free providers differ on, both of which silently end a tool loop.
- **`response_format: json_schema` is honoured or the attempt fails.** Providers
  that can be constrained natively are preferred and get the schema essentially
  as you wrote it; the rest are told in the prompt. Either way the answer is
  validated against the schema you sent, and a mismatch fails over to the next
  provider rather than being returned.
- **The conversation is never truncated, summarised or reordered**, and a model
  that cannot take it in one request is not offered the job — which on a free
  tier is usually decided by the tokens-per-minute allowance rather than the
  model's context window.
- **A run stays on one model.** Handing a half-written transcript to a different
  model mid-run is worse than waiting; see *Sessions* below.
- **Quota exhaustion answers 429**, with `Retry-After`, because a client that
  gets 400 gives up on a window that reopens in forty seconds.

### Sessions

Nothing in the OpenAI request says which run it belongs to, so a request that
carries `tools` is fingerprinted by its opening messages: the same system prompt
and first user message means the same run, and the run keeps the model it
started on for half an hour of activity. Send `X-OnFeather-Session: <id>` to say
so explicitly instead.

When the pinned provider becomes unusable mid-run, the same model on another
provider is tried before any different model, and the response says what
happened in `X-OnFeather-Repinned`.

### Strategies

| Strategy | Behaviour |
|---|---|
| `balanced` (default) | Route to whoever has the most quota left, preserving the scarcest |
| `fast` | Prefer low-latency providers, but yield when their quota runs low |
| `local` | Never leave the machine |

`--private` forces a request local regardless of strategy. Worth knowing: Google's
free tier may use your prompts for training, which is recorded in the registry
and is exactly the sort of request that flag exists for.

## How quota is tracked

Three sources, in increasing order of authority:

1. **Our own tally** — every request is recorded in SQLite and summed inside the
   limit's window. Always available, but blind to spending outside this tool.
2. **Response headers** — most providers report both what is left *and* what the
   allowance is, and both beat the registry. Free tiers differ between accounts,
   so a static file can only ever be a starting guess.
3. **A 429** — ground truth. Locks the provider out until the window turns over,
   whatever the other two believe.

Ordering matters in one place worth knowing about: the remaining-quota header is
returned *with* the response to the request that consumed it, so it already
accounts for that request. Recording our own tally and then subtracting it from
the header again undercounts on every single call.

## What checking against live providers found

Every provider in the registry was exercised with a real key on 2026-07-26. The
published documentation and the actual behaviour disagreed five times:

| Provider | Documented | Actually |
|---|---|---|
| Cerebras | free tier, no card | **402 Payment Required** on every model |
| OpenRouter | sends rate-limit headers | sends **none** |
| Mistral | `x-ratelimit-remaining-requests` | `x-ratelimit-remaining-**req-minute**` |
| Groq | 30 requests/minute | reports **1000** in its own headers |
| GitHub Models | 15/min, 150/day | reports **20000/min**, 2M tokens/min |

None of these broke anything, which is the point: the registry is a hint, the
headers correct it, and a 429 overrides both. But three of them would have made
the router take worse decisions for a whole window, so they are now recorded —
and `providers.yaml` carries a `verified_at` date so the next reader knows how
much to trust it.

The Mistral one is the reason [`headers.py`](src/onfeather_free/headers.py)
parses a family of spellings rather than a fixed list: matching only the OpenAI
form dropped that provider's headers silently, with no error and no clue.

## Local models are discovered, not declared

Which models a local runner holds is whatever you happened to pull, on this
machine, today — it cannot live in a checked-in file. So `providers.yaml`
declares that Ollama exists, and the actual catalogue is read from it at startup.

This doubles as the liveness check. A runner that is not running answers nothing,
its model list comes back empty, and it drops out of routing on its own rather
than being chosen as the last resort and then failing.

Both halves of that were real bugs: a hardcoded `qwen2.5:7b` 404'd on a machine
holding `qwen2.5-coder:7b-instruct`, and the one provider meant to guarantee you
are never stranded was the one guaranteed to fail.

## Known gaps

- **No streaming.** `stream: true` is refused with a 400. Passing the upstream
  SSE through is the next piece of work, and until then some clients need
  configuring.
- Token counting before a request is an estimate, so TPM limits are approximate
  until the response arrives.
- A permanent quota of zero looks like a temporary one. Google answers 429 with
  `limit: 0` when a project has no free allowance at all — the same status code
  as ordinary rate limiting, so it is retried once per cooldown forever instead
  of being sidelined like a 402. Distinguishing them means reading the error
  body, which is provider-specific.
- Model pinning is parsed but not enforced: `provider/model` is accepted and then
  routed normally. Session pinning is enforced; that is a different mechanism.
- **`tools` and `json_schema` in the registry are guesses like everything else
  there.** A model marked tool-capable that is not just fails the attempt and the
  next candidate serves it, but there is no header to reconcile against the way
  there is for quota, so a wrong flag costs a wasted request every time until
  someone corrects the YAML. Local models are the exception: Ollama is asked
  whether the model's prompt template renders tools, so that one is measured.

## Related

- [`onfeather-tune`](https://github.com/RickyLeRoi/onfeather-tune) — the other
  half: run heavy models on light local hardware.

## Licence

Apache-2.0
