"""Making the request, and learning from what comes back.

Every response teaches the ledger something: the `usage` block says what was
spent, the rate-limit headers say what the provider thinks is left, and a 429
says the quota is gone regardless of either. Failing over to the next candidate
is the point of the whole exercise, so a provider running out is an ordinary
event here rather than an error.

Failing over is also where an agentic caller gets hurt if this is done naively.
Its request carries tool definitions and, sometimes, a schema the answer must
obey; a provider that quietly ignores either returns HTTP 200 and unusable
content. So a wrong-shaped answer counts as a failed attempt here, exactly like
a 500, and the next candidate gets a turn.
"""

from __future__ import annotations

import contextlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import httpx

from . import compat
from .budget import Ledger
from .registry import Model, Provider, Registry
from .router import Candidate, Route, candidates, configured_providers

#: 20260725 RG An agentic turn on a free tier takes minutes.
DEFAULT_TIMEOUT = 600.0

#: 20260725 RG Too low truncates tool calls mid-JSON.
DEFAULT_MAX_TOKENS = 4096

#: 20260725 RG For a 429 that does not say when to come back.
DEFAULT_COOLDOWN_SECONDS = 60.0

#: 20260808 ** RG #Security No free window is longer than a day; past that it is a provider bug.
MAX_COOLDOWN_SECONDS = 86400.0

CONFIGURATION_STATUSES = frozenset({401, 402, 403})

CONFIGURATION_COOLDOWN_SECONDS = 900.0

RETRYABLE_STATUSES = frozenset({408, 409, 429})


class CompletionError(Exception):
    """Raised when every candidate failed."""

    def __init__(
        self, message: str, attempts: list[Attempt], *, quota_exhausted: bool = False
    ) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.quota_exhausted = quota_exhausted

    @property
    def status(self) -> int:
        """What to answer the caller with, which decides whether it retries.

        The three outcomes are genuinely different and collapsing them is how a
        client ends up retrying a malformed request twice, or giving up on a
        quota window that turns over in forty seconds.
        """
        if self.quota_exhausted or any(attempt.status == 429 for attempt in self.attempts):
            return 429

        codes = [attempt.status for attempt in self.attempts if attempt.status]
        if codes and all(
            400 <= code < 500
            and code not in RETRYABLE_STATUSES
            and code not in CONFIGURATION_STATUSES
            for code in codes
        ):
            return codes[0]

        return 503


@dataclass
class Attempt:
    provider: str
    model: str
    ok: bool
    status: int | None = None
    error: str | None = None
    """Our own account of what went wrong. Safe to hand to whoever called us."""
    detail: str | None = None
    """What the provider itself said, kept apart because it is not ours to pass
    on: these bodies routinely carry the organisation id, the project id and
    correlatable request ids, and the loopback default authenticates nobody.
    The server logs it and answers with `error`; the CLI, talking to the person
    running the process, prints both."""


@dataclass
class Completion:
    text: str
    provider: Provider
    model: str
    tokens_in: int
    tokens_out: int
    latency_s: float
    message: dict = field(default_factory=lambda: {"role": "assistant", "content": None})
    finish_reason: str = "stop"
    attempts: list[Attempt] = field(default_factory=list)

    @property
    def failed_over(self) -> bool:
        return len(self.attempts) > 1

    @property
    def tool_calls(self) -> list[dict]:
        return self.message.get("tool_calls") or []


class Client:
    """Sends chat completions through the router, keeping the ledger honest."""

    def __init__(
        self,
        registry: Registry,
        ledger: Ledger,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.registry = registry
        self.ledger = ledger
        self.timeout = timeout
        self.max_tokens = max_tokens
        # 20260808 ** RG #Security One pool per process: a client per request paid a TLS handshake.
        self._http = httpx.Client(
            timeout=timeout,
            transport=transport,
            # 20260808 ** RG #Security No env proxy or CA bundle: bearer tokens travel here.
            trust_env=False,
            limits=httpx.Limits(max_keepalive_connections=16, max_connections=32),
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> Client:
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def complete(
        self,
        messages: list[dict],
        *,
        capability: str = "chat",
        strategy: str = "balanced",
        private: bool = False,
        max_tokens: int | None = None,
        temperature: float | None = None,
        tools: list[dict] | None = None,
        tool_choice: object | None = None,
        response_format: dict | None = None,
        min_context: int = 0,
        require: tuple[str, str] | None = None,
        prefer: tuple[str, str] | None = None,
        now: datetime | None = None,
        environ: dict | None = None,
    ) -> Completion:
        moment = now or datetime.now(timezone.utc)
        requires = {"tools"} if tools else set()
        prefers = {"json_schema"} if compat.schema_of(response_format) else set()

        options = candidates(
            self.registry,
            self.ledger,
            capability=capability,
            requires=requires,
            prefers=prefers,
            min_context=min_context,
            strategy=strategy,
            private=private,
            now=moment,
            environ=environ,
        )
        if not options:
            raise CompletionError(
                "no provider available for this request",
                [],
                quota_exhausted=self._anything_locked_out(moment, environ),
            )

        options = _require_pair(options, require)
        options = _honour_pin(options, prefer)
        request = _Request(
            messages=messages,
            max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
            temperature=temperature,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
        )

        attempts: list[Attempt] = []
        for option in options:
            attempt, completion = self._try(option, request, environ)
            attempts.append(attempt)
            if completion is not None:
                completion.attempts = attempts
                return completion

        raise CompletionError(
            f"all {len(attempts)} candidates failed; last error: {attempts[-1].error}", attempts
        )

    def _anything_locked_out(self, moment: datetime, environ: dict | None = None) -> bool:
        """Whether an empty candidate list is a quota problem or a config one.

        Worth the extra query: a caller told 429 waits and retries, a caller told
        503 for a missing API key retries twice and then reports something true.
        """
        configured = configured_providers(self.registry, environ)
        # 20260808 ** RG #Security One read for the whole sweep, not four per limit per provider.
        view = self.ledger.snapshot(moment, providers=configured)
        return any(
            not self.ledger.status(provider, moment, snapshot=view).available
            for provider in self.registry.usable()
            if provider.name in configured
        )

    # -- one provider -----------------------------------------------------

    def _try(
        self, option: Candidate, request: _Request, environ: dict | None
    ) -> tuple[Attempt, Completion | None]:
        provider = option.provider
        route = Route(provider, option.model, option.status, reason="")
        payload = request.payload_for(provider, option.model)

        headers = {"Content-Type": "application/json"}
        key = self._api_key(provider, environ)
        if key:
            headers["Authorization"] = f"Bearer {key}"

        started = time.perf_counter()
        try:
            response = self._http.post(
                _join(provider.base_url, "chat/completions"), json=payload, headers=headers
            )
        except httpx.HTTPError as error:
            # 20260725 RG Unreachable is not a quota problem; do not charge the budget.
            return Attempt(provider.name, option.model.id, ok=False, error=str(error)), None

        latency = time.perf_counter() - started
        outcome = self._account(route, response, latency, request)

        # 20260725 RG Ordering is load-bearing: reconcile last.
        self.ledger.observe_headers(provider, dict(response.headers))
        return outcome

    def _account(
        self, route: Route, response: httpx.Response, latency: float, request: _Request
    ) -> tuple[Attempt, Completion | None]:
        provider, model = route.provider, route.model

        if response.status_code == 429:
            self._cool_down(provider, response)
            return (
                Attempt(provider.name, model.id, ok=False, status=429, error="rate limited"),
                None,
            )

        if response.status_code in CONFIGURATION_STATUSES:
            # 20260725 RG Config problem, not quota: Cerebras 402s without a free tier.
            self.ledger.lock_out(
                provider.name,
                until=time.time() + CONFIGURATION_COOLDOWN_SECONDS,
                reason=f"http {response.status_code}",
            )
            return (
                Attempt(
                    provider.name,
                    model.id,
                    ok=False,
                    status=response.status_code,
                    error=_configuration_hint(response.status_code, provider),
                ),
                None,
            )

        if response.status_code >= 400:
            self.ledger.record(provider.name, model=model.id, requests=1)
            return (
                Attempt(
                    provider.name,
                    model.id,
                    ok=False,
                    status=response.status_code,
                    # 20260808 ** RG #Security A classification travels; the body stays in `detail`.
                    error=f"rejected the request (http {response.status_code})",
                    detail=response.text[:200],
                ),
                None,
            )

        return self._success(route, response, latency, request)

    def _success(
        self, route: Route, response: httpx.Response, latency: float, request: _Request
    ) -> tuple[Attempt, Completion | None]:
        provider, model = route.provider, route.model

        def rejected(reason: str) -> tuple[Attempt, None]:
            self.ledger.record(provider.name, model=model.id, requests=1)
            return (
                Attempt(
                    provider.name, model.id, ok=False, status=response.status_code, error=reason
                ),
                None,
            )

        try:
            body = response.json()
            choice = body["choices"][0]
            raw_message = choice["message"]
        except (ValueError, KeyError, IndexError, TypeError) as error:
            return rejected(f"unparseable response: {error}")

        message = compat.normalise_message(raw_message, seed=str(body.get("id") or "") or model.id)
        problem = compat.message_problem(message)
        if problem:
            return rejected(problem)

        schema = compat.schema_of(request.response_format)
        if schema is not None:
            try:
                message = _conform(message, schema)
            except ValueError as error:
                # 20260725 RG HTTP 200 of the wrong shape is still worth failing over.
                return rejected(f"response did not match the schema: {error}")

        usage = body.get("usage") or {}
        tokens_in = int(usage.get("prompt_tokens") or 0)
        tokens_out = int(usage.get("completion_tokens") or 0)

        self.ledger.record(
            provider.name, model=model.id, requests=1, tokens=tokens_in + tokens_out
        )

        content = message.get("content")
        return (
            Attempt(provider.name, model.id, ok=True, status=response.status_code),
            Completion(
                text=content if isinstance(content, str) else "",
                provider=provider,
                model=model.id,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                latency_s=latency,
                message=message,
                finish_reason=compat.finish_reason(message, choice.get("finish_reason")),
            ),
        )

    def _cool_down(self, provider: Provider, response: httpx.Response) -> None:
        """Sideline a rate-limited provider for as long as it asks — within reason.

        `retry-after` is authoritative when present. Without it we guess a
        minute, which recovers a per-minute limit on its own and stops us
        hammering a daily one more than once a minute.

        Capped at a day, because the ledger persists lockouts: an upstream that
        sends milliseconds where seconds were meant would otherwise retire the
        provider for thirty years, with nothing short of `of-free reset` able to
        undo it.
        """
        seconds = DEFAULT_COOLDOWN_SECONDS
        raw = response.headers.get("retry-after")
        if raw:
            try:
                seconds = min(max(float(raw), 1.0), MAX_COOLDOWN_SECONDS)
            except ValueError:
                # 20260808 ** RG #Security Retry-After also admits an HTTP date; do not drop it.
                with contextlib.suppress(TypeError, ValueError):
                    when = parsedate_to_datetime(raw)
                    seconds = min(max(when.timestamp() - time.time(), 1.0), MAX_COOLDOWN_SECONDS)
        self.ledger.lock_out(provider.name, until=time.time() + seconds, reason="429")

    @staticmethod
    def _api_key(provider: Provider, environ: dict | None) -> str | None:
        import os

        source = environ if environ is not None else os.environ
        return source.get(provider.api_key_env) if provider.api_key_env else None


@dataclass
class _Request:
    """One caller's request, before it is bent into a given provider's shape."""

    messages: list[dict]
    max_tokens: int | None = None
    temperature: float | None = None
    tools: list[dict] | None = None
    tool_choice: object | None = None
    response_format: dict | None = None

    def payload_for(self, provider: Provider, model: Model) -> dict:
        dialect = provider.schema_dialect
        messages = self.messages
        payload: dict = {"model": model.id, "messages": messages}

        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if self.temperature is not None:
            payload["temperature"] = self.temperature

        tools = compat.adapt_tools(self.tools, dialect)
        if tools:
            payload["tools"] = tools
            if self.tool_choice is not None:
                payload["tool_choice"] = self.tool_choice

        if self.response_format is not None:
            messages = self._apply_response_format(payload, model, dialect, messages)
            payload["messages"] = messages

        return payload

    def _apply_response_format(
        self, payload: dict, model: Model, dialect: str, messages: list[dict]
    ) -> list[dict]:
        """Constrain the answer natively where possible, in the prompt otherwise.

        The schema is never repeated in the caller's own prompt -- sending it as
        `response_format` is the caller saying it does not have to be -- so a
        provider that cannot be constrained has to be told in words, or it
        answers with a perfectly valid object of some other shape.
        """
        schema = compat.schema_of(self.response_format)
        if schema is None:
            if self.response_format.get("type") == "json_object" and (
                {"json_schema", "json_object"} & model.capabilities
            ):
                payload["response_format"] = {"type": "json_object"}
            return messages

        if "json_schema" in model.capabilities:
            payload["response_format"] = compat.adapt_response_format(
                self.response_format, dialect
            )
            return messages

        if "json_object" in model.capabilities:
            payload["response_format"] = {"type": "json_object"}
        return compat.with_instruction(messages, compat.schema_instruction(schema))


def _conform(message: dict, schema: dict) -> dict:
    """Make the content satisfy the schema, or say why it cannot.

    Pruning before validating is deliberate. An extra key is the common failure
    and is recoverable; a missing required key or a wrong type is not, and is
    worth spending another candidate on.
    """
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("no content to validate")

    value = compat.prune(compat.extract_json(content), schema)
    errors = compat.validate(value, schema)
    if errors:
        raise ValueError("; ".join(errors[:3]))

    out = dict(message)
    out["content"] = json.dumps(value, ensure_ascii=False)
    return out


def _require_pair(options: list[Candidate], require: tuple[str, str] | None) -> list[Candidate]:
    """Keep only the provider/model the caller named, or fail saying so.

    `require` filters where `prefer` reorders. Naming `ollama/qwen2.5:7b` is the
    caller saying this must not leave the machine, so an impossible pin has to
    fail rather than quietly route somewhere else — a prompt that has left is
    not coming back.
    """
    if not require:
        return options

    provider_name, model_id = require
    kept = [
        option for option in options
        if option.provider.name == provider_name and option.model.id == model_id
    ]
    if not kept:
        raise CompletionError(
            f"{provider_name}/{model_id} was requested explicitly and is not available "
            "right now; send `auto` to let the router choose instead",
            [],
        )
    return kept


def _honour_pin(options: list[Candidate], prefer: tuple[str, str] | None) -> list[Candidate]:
    """Reorder so a pinned choice is tried first, then the same model elsewhere.

    An agentic run that changes model halfway is worse than one that waits: the
    conversation so far was written by a different model, in its own style of
    tool call, and the new one has to keep faith with it. So when the pinned pair
    is unavailable, another provider serving the *same model id* outranks a
    healthier provider serving a different one.
    """
    if not prefer:
        return options
    provider_name, model_id = prefer

    def rank(option: Candidate) -> int:
        if option.provider.name == provider_name and option.model.id == model_id:
            return 0
        if option.model.id == model_id:
            return 1
        return 2

    return sorted(options, key=rank)


def _configuration_hint(status: int, provider: Provider) -> str:
    """Say what to actually do, since none of these fix themselves."""
    env = provider.api_key_env or "the API key"
    if status == 401:
        return f"unauthorised — check {env}"
    if status == 402:
        return "payment required — this account has no free tier for this model"
    return f"forbidden — {env} may lack access to this model"


def _join(base: str, path: str) -> str:
    return f"{base.rstrip('/')}/{path.lstrip('/')}"
