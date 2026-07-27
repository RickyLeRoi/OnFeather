# Serving an agentic client

What `of-free serve` guarantees to a client that calls tools in a loop, or asks
for an answer constrained to a schema. Written against the two workloads that
drove it — AutoHeal's `generate` and `heal` — but nothing here is specific to
that client.

## The endpoint

```
POST http://127.0.0.1:4141/v1/chat/completions
Authorization: Bearer <anything>
Content-Type: application/json
```

The `Authorization` header is read and discarded. This is a personal proxy bound
to loopback; it holds *your* provider keys, and asking you to invent a second
credential to reach your own machine would protect nothing. `Bearer none` is
fine. If you bind it to anything other than `127.0.0.1`, put something in front
that does authenticate.

`GET /v1/models`, `GET /v1/status` and `GET /health` exist. Nothing else does.

`stream: true` is refused with a 400 naming the field, rather than a 200 the
client cannot parse.

## Workload A — a tool-calling loop

```json
{ "model": "auto", "tool_choice": "auto", "messages": [...], "tools": [...] }
```

**Routing.** `tools` in the request restricts routing to models flagged
`tools` in `providers.yaml`. A model that answers a tool request with prose
about the tool is not a slower option, it is a loop that never terminates. For
local models this is measured rather than declared: Ollama is asked for the
model's prompt template, and tool calling is claimed only if that template
renders tool definitions.

**Size.** The request is measured against what each model will actually accept
in one go, which is the smaller of two numbers:

- the model's own context window, and
- **the tokens-per-minute allowance**, because a single request cannot spend
  more than a window holds.

The second is usually the binding one on a free tier and is easy to miss: Groq's
Llama 3.3 70B holds 128k tokens and its free tier passes about 6k a minute, so a
15k-token turn is an immediate 429 rather than a slow answer. Routing therefore
treats it as a 6k model. That number is also the one the ledger reconciles from
`x-ratelimit-limit-tokens`, so the ceiling loosens by itself on an account with
a larger real allowance.

Where neither number is recorded, nothing is excluded — an unknown is not
evidence of a small model.

Nothing is ever truncated, summarised or reordered: an agentic client resends
the whole conversation each turn and expects it back intact, including
`assistant` messages with `content: null` plus `tool_calls`, and `tool` messages
with `tool_call_id`.

**Defaults.** Callers that send no `max_tokens` get 4096, configurable with
`--max-tokens`. This is the failure that looks like a model problem: a ceiling
low enough to truncate a tool call mid-JSON produces a malformed `arguments`
string and no error anywhere. `temperature`, `top_p`, `stop` and `seed` are
never invented — an unsent parameter stays unsent.

**What comes back.** The upstream assistant message, normalised rather than
rebuilt:

| Field | Guarantee |
|---|---|
| `choices[0].message` | always present |
| `message.content` | `null` when there are tool calls — never `""` |
| `tool_calls[].function.arguments` | always a JSON string, even from providers that return an object |
| `tool_calls[].function.name` | always a non-empty string, or the attempt fails over |
| `tool_calls[].id` | always present; synthesised deterministically when the provider omits one |
| `finish_reason` | `tool_calls` whenever there are tool calls |
| `usage`, `id`, `created`, `model` | always present; `model` is `provider/model` |

The `id` is what comes back as the next request's `tool_call_id`, so a provider
that omits it would end the conversation there. Synthesised ids hash the call's
name and arguments, so the same call yields the same id.

## Workload B — an answer constrained to a schema

```json
{ "response_format": { "type": "json_schema",
                       "json_schema": { "name": "response", "strict": true, "schema": {...} } } }
```

The schema is usually not repeated in the prompt — sending it as
`response_format` is the caller saying it does not have to be — so ignoring the
field does not degrade the answer, it returns the wrong object.

Three things happen, in order:

1. **Preference.** Models flagged `json_schema` outrank everything else for this
   request, ahead of quota. Emulating a missing feature costs more than spending
   scarcer quota.
2. **Translation, and mostly the absence of it.** `$schema` and `$id` are
   stripped for everyone — they identify a document, they are not constraints —
   and `$defs` deliberately is not, because a `$ref` still points at it. Beyond
   that the schema goes as written. Providers that route to an OpenAI-shaped
   constrained decoder take draft-07 as it stands, including `anyOf: [T, null]`
   and numeric bounds, and rewriting a schema that would have been accepted only
   throws information away.

   The exception is OpenAI's own strict mode, which documents a subset and
   answers 400 on the rest rather than ignoring it. A provider marked
   `schema_dialect: openai_strict` — GitHub Models, which is backed by it —
   loses `minimum`, `maximum`, `pattern`, `minLength` and the other unlisted
   keywords on the way out. They are still enforced on the way back.

   A provider with no native support at all gets `{"type": "json_object"}` and
   the full schema folded into its system prompt.
3. **Verification.** The answer is parsed — unwrapping a code fence or
   surrounding prose if the model added one — then keys the schema forbids are
   dropped, then it is validated against the schema *as you sent it*, including
   the constraints step 2 had to drop. A missing required key or a wrong type
   fails the attempt and the next provider gets a turn.

So `strict: true` holds even through a provider that has never heard of it. The
supported subset of draft-07 is `type` (including unions), `enum`, `required`,
`properties`, `additionalProperties`, `items`, `anyOf`/`oneOf`, `minimum`,
`maximum`, `minLength` and `maxLength`. Anything else is passed through to the
provider and not checked on the way back.

Each such call is independent, so routing is free to send consecutive calls to
different providers.

## Staying on one model

An agentic run that changes model halfway is worse than one that waits: the
transcript so far was written by a different model, in its own style of tool
call, and the new one has to keep faith with it.

There is no session id in an OpenAI request, so one is derived. A request
carrying `tools` is fingerprinted by its first two messages — the system prompt
and the opening user message, which every turn of a run resends unchanged. That
fingerprint pins provider and model for 30 minutes of activity, and comes back
as `X-OnFeather-Session`. Send that header yourself to be explicit; anything
non-empty works.

Requests without `tools` are not pinned. They are stateless by construction and
pinning one would hand it yesterday's decision for no reason.

When the pinned pair is unavailable, the order of preference is:

1. the same model on the same provider,
2. **the same model id on a different provider**,
3. the best remaining candidate.

Reaching 3 sets `X-OnFeather-Repinned: <old provider>/<old model>` on the
response. The run continues — refusing to answer helps nobody — but the header
says the transcript changed hands, which is worth logging.

## Failure, and what a client should do about it

Errors are OpenAI-shaped, always with all four keys:

```json
{"error": {"message": "...", "type": "...", "param": null, "code": "..."}}
```

| Status | When | `code` |
|---|---|---|
| `400` | Malformed request, or every provider rejected the body | `upstream_rejected` |
| `404` | A path that does not exist | — |
| `429` | Quota exhausted everywhere, or a provider rate-limited us | `quota_exhausted` |
| `503` | Providers unreachable, misconfigured, or unable to produce a usable answer | `upstream_unavailable` |

The distinction that matters is 429 against 400. Free-tier quota comes back —
often within a minute — so exhaustion is reported as 429 and a client that
retries gets served. A 400 is only returned when *every* candidate rejected the
request body itself, which retrying cannot fix.

`Retry-After` accompanies the 429 whenever a provider has told us when to come
back, directly or through its own `retry-after` header. Quota we ran out of by
our own tally has no such moment attached, and no header is invented for it.

An upstream 401 or 402 — our key, not your request — is reported as 503 rather
than passed through. A client seeing 401 concludes its own credential is wrong,
which is both false and unhelpful.

Nothing is retried internally beyond the failover chain: each candidate is tried
once, in order, and the first usable answer wins.

## Time

The gateway waits `--timeout` seconds for a provider, default **600**. A long
context on a free tier genuinely takes minutes. The server itself imposes no
limit on how long a request may take, so nothing is closed from this end.

If you put a reverse proxy in front, its own timeout becomes the real one —
nginx defaults `proxy_read_timeout` to 60 seconds, which will cut a working
request in half:

```nginx
proxy_read_timeout 660s;
proxy_send_timeout 660s;
```

There is no batching and no concurrency requirement; the endpoint handles
parallel requests but nothing about it assumes them.

## Checking it works

```bash
curl -s localhost:4141/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "auto",
  "tools": [{"type": "function", "function": {"name": "get_weather",
    "parameters": {"$schema": "http://json-schema.org/draft-07/schema#",
                   "type": "object", "properties": {"city": {"type": "string"}},
                   "required": ["city"], "additionalProperties": false}}}],
  "tool_choice": "auto",
  "messages": [{"role": "user", "content": "What is the weather in Rome?"}]
}' | jq '.choices[0].message.tool_calls[0]'
```

`arguments` should be a string, and `id` should be there.

```bash
curl -s localhost:4141/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "auto",
  "response_format": {"type": "json_schema", "json_schema": {"name": "response", "strict": true,
    "schema": {"$schema": "http://json-schema.org/draft-07/schema#", "type": "object",
      "properties": {"ref": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                     "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                     "reasoning": {"type": "string"}},
      "required": ["ref", "confidence", "reasoning"], "additionalProperties": false}}},
  "messages": [{"role": "user", "content": "Which element matches the failing selector?"}]
}' | jq -r '.choices[0].message.content' | jq 'keys'
```

That should print exactly `["confidence", "reasoning", "ref"]`, whichever
provider answered.
