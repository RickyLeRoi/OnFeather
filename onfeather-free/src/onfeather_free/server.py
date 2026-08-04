"""OpenAI-compatible endpoint.

Point any OpenAI client at this and it routes across free tiers without knowing
it. Built on the standard library rather than a web framework: a personal proxy
that needs `pip install fastapi uvicorn` to start is one more thing to maintain
than the job requires.

Serving an agentic client rather than a chat box changes two things. The request
body has to survive intact -- tools, tool choice, response format, and a
conversation that grows to tens of thousands of tokens over sixty turns, never
truncated or reordered. And the model has to stay put: the transcript so far was
written by one model, and swapping in another halfway leaves it arguing with a
stranger. Sessions below are how the second one is arranged without asking the
client for anything it does not already send.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from . import companions
from .budget import Ledger
from .client import Client, CompletionError
from .registry import Registry
from .router import STRATEGY_BALANCED, candidates, configured_providers

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 4141

# 20260804 ++ RG #HASS The container healthcheck sends no key.
PUBLIC_PATHS = frozenset({"/health", "/healthz"})

# 20260804 ++ RG #HASS Unset means no authentication, the loopback default.
API_KEY_ENV = "ONFEATHER_API_KEY"

# 20260725 RG Routed locally whatever the strategy.
PRIVATE_MODEL_ALIASES = frozenset({"private", "local"})

AUTO_MODEL_ALIASES = frozenset({"auto", "onfeather", "of-free", ""})

# 20260725 RG Pin a run explicitly instead of being fingerprinted.
SESSION_HEADER = "X-OnFeather-Session"

SESSION_TTL_SECONDS = 1800.0

CHARS_PER_TOKEN = 4


class Sessions:
    """Which model each ongoing run is pinned to.

    An agentic client sends no session id, so one is derived from the opening of
    the conversation: every turn of a run resends the same system prompt and
    first user message, and a different run has a different pair. It is a
    fingerprint rather than a guarantee — two identical runs share a pin, which
    costs nothing because they would want the same answer anyway.
    """

    def __init__(self, ttl: float = SESSION_TTL_SECONDS) -> None:
        self.ttl = ttl
        self._pins: dict[str, tuple[str, str, float]] = {}
        self._lock = threading.Lock()

    def get(self, key: str | None) -> tuple[str, str] | None:
        if not key:
            return None
        with self._lock:
            self._expire()
            entry = self._pins.get(key)
        return (entry[0], entry[1]) if entry else None

    def remember(self, key: str | None, provider: str, model: str) -> None:
        if not key:
            return
        with self._lock:
            self._pins[key] = (provider, model, time.time())
            self._expire()

    def _expire(self) -> None:
        cutoff = time.time() - self.ttl
        for key in [k for k, entry in self._pins.items() if entry[2] < cutoff]:
            del self._pins[key]


def session_key(payload: dict, headers: Any = None) -> str | None:
    """Identify the run this request belongs to, if it is part of one.

    Only tool-carrying requests get a key. A one-shot call is stateless by
    construction and pinning it would hand it yesterday's model for no reason.
    """
    explicit = headers.get(SESSION_HEADER) if headers is not None else None
    if explicit and explicit.strip():
        return explicit.strip()

    if not payload.get("tools"):
        return None

    messages = payload.get("messages") or []
    opening = json.dumps(messages[:2], sort_keys=True, ensure_ascii=False)[:8000]
    return hashlib.sha256(opening.encode()).hexdigest()[:16]


def estimated_tokens(payload: dict) -> int:
    """How big this request is, near enough to exclude a model that cannot hold it."""
    material = json.dumps(
        [payload.get("messages") or [], payload.get("tools") or []], ensure_ascii=False
    )
    return len(material) // CHARS_PER_TOKEN


@dataclass(frozen=True)
class Served:
    """The last request this server actually answered.

    `next` says where the router would go; this says where it went. They differ
    the moment a provider runs out mid-conversation, and the gap is the whole
    reason both are worth reporting.
    """

    provider: str
    model: str
    at: float
    failovers: int
    tokens_in: int
    tokens_out: int
    latency_s: float

    def as_dict(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model,
            "id": f"{self.provider}/{self.model}",
            "at": self.at,
            "failovers": self.failovers,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "latency_s": round(self.latency_s, 3),
        }


class Router:
    """Shared state for the handler, which cannot take constructor arguments."""

    def __init__(
        self,
        registry: Registry,
        ledger: Ledger,
        *,
        strategy: str = STRATEGY_BALANCED,
        verbose: bool = False,
        timeout: float | None = None,
        max_tokens: int | None = None,
        api_key: str | None = None,
    ) -> None:
        self.registry = registry
        self.ledger = ledger
        options = {}
        if timeout is not None:
            options["timeout"] = timeout
        if max_tokens is not None:
            options["max_tokens"] = max_tokens
        self.client = Client(registry, ledger, **options)
        self.strategy = strategy
        self.verbose = verbose
        self.api_key = api_key or None
        self.sessions = Sessions()
        # 20260804 ++ RG #HASS Rebound whole under the GIL; worker threads need no lock.
        self.last: Served | None = None

    def models(self) -> list[dict]:
        """Everything routable right now, plus the virtual `auto` model."""
        listed: list[dict] = [
            {"id": "auto", "object": "model", "owned_by": "onfeather",
             "description": "route automatically across available free tiers"}
        ]
        for provider in self.registry.usable():
            for model in provider.models:
                listed.append({
                    "id": f"{provider.name}/{model.id}",
                    "object": "model",
                    "owned_by": provider.name,
                })
        return listed

    def resolve(self, requested: str) -> tuple[str, bool, str | None]:
        """Map a requested model onto (capability, private, pinned provider).

        Three shapes are accepted: `auto` to let the router decide, `private` to
        force a local model, and `provider/model` to pin one explicitly.
        """
        name = (requested or "").strip()
        if name.lower() in PRIVATE_MODEL_ALIASES:
            return "chat", True, None
        if name.lower() in AUTO_MODEL_ALIASES:
            return "chat", False, None
        if "/" in name:
            provider_name = name.split("/", 1)[0]
            if provider_name in self.registry.providers:
                return "chat", False, provider_name
        # 20260725 RG Unknown name: route it rather than refuse.
        return "chat", False, None

    def retry_after(self) -> int | None:
        """Seconds until the first quota window turns over, if any is locked."""
        now = datetime.now(timezone.utc)
        waits = [
            self.ledger.status(provider, now).locked_until
            for provider in self.registry.usable()
        ]
        pending = [until - time.time() for until in waits if until is not None]
        return max(1, int(min(pending))) if pending else None


class Handler(BaseHTTPRequestHandler):
    router: Router

    protocol_version = "HTTP/1.1"
    server_version = "onfeather-free"

    # -- plumbing ---------------------------------------------------------

    def log_message(self, fmt: str, *args: Any) -> None:
        if self.router.verbose:
            super().log_message(fmt, *args)

    def _send(self, status: int, payload: dict, *, extra: dict | None = None) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for key, value in (extra or {}).items():
            self.send_header(key, str(value))
        self.end_headers()
        self.wfile.write(body)

    def _error(
        self,
        status: int,
        message: str,
        kind: str = "invalid_request_error",
        *,
        code: str | None = None,
        extra: dict | None = None,
    ) -> None:
        self._send(
            status,
            {"error": {"message": message, "type": kind, "param": None, "code": code}},
            extra=extra,
        )

    def _authorised(self, path: str) -> bool:
        """Whether this request may proceed.

        Unset key means no check at all, so a loopback install behaves exactly
        as it did before. When one is set it travels in `Authorization: Bearer`,
        which every OpenAI client already sends and every user already expects
        to fill in.
        """
        expected = self.router.api_key
        if not expected or path in PUBLIC_PATHS:
            return True

        offered = (self.headers.get("Authorization") or "").strip()
        if offered[:7].lower() == "bearer ":
            offered = offered[7:].strip()
        return bool(offered) and hmac.compare_digest(offered, expected)

    def _unauthorised(self) -> None:
        # 20260804 ++ RG #HASS Drain the body, or keep-alive reads it as the next request.
        self._read_json()
        self._error(
            401,
            "missing or invalid API key; send it as `Authorization: Bearer <key>`",
            code="invalid_api_key",
            extra={"WWW-Authenticate": 'Bearer realm="onfeather-free"'},
        )

    def _read_json(self) -> dict | None:
        try:
            length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            return None
        if length <= 0:
            return None
        try:
            return json.loads(self.rfile.read(length))
        except (ValueError, OSError):
            return None

    # -- routes -----------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0].rstrip("/")
        if not self._authorised(path):
            self._unauthorised()
            return

        if path in ("/health", "/healthz"):
            self._send(200, {"status": "ok"})
        elif path in ("/v1/models", "/models"):
            self._send(200, {"object": "list", "data": self.router.models()})
        elif path in ("/v1/status", "/status"):
            self._send(200, self._status_payload())
        else:
            self._error(404, f"unknown path {path}", "not_found")

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0].rstrip("/")
        if not self._authorised(path):
            self._unauthorised()
            return

        if path not in ("/v1/chat/completions", "/chat/completions"):
            self._error(404, f"unknown path {path}", "not_found")
            return

        payload = self._read_json()
        if payload is None:
            self._error(400, "request body must be JSON")
            return

        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            self._error(400, "`messages` must be a non-empty array")
            return

        if payload.get("stream"):
            # 20260725 RG Streaming needs upstream passthrough; not built.
            self._error(
                400,
                "streaming is not supported yet; retry with \"stream\": false",
                "not_implemented",
            )
            return

        capability, private, _pinned = self.router.resolve(payload.get("model", "auto"))
        key = session_key(payload, self.headers)
        pin = self.router.sessions.get(key)

        try:
            result = self.router.client.complete(
                messages,
                capability=capability,
                strategy=self.router.strategy,
                private=private,
                max_tokens=payload.get("max_tokens"),
                temperature=payload.get("temperature"),
                tools=payload.get("tools"),
                tool_choice=payload.get("tool_choice"),
                response_format=payload.get("response_format"),
                min_context=estimated_tokens(payload),
                prefer=pin,
            )
        except CompletionError as error:
            self._fail(error)
            return

        self.router.sessions.remember(key, result.provider.name, result.model)
        self.router.last = Served(
            provider=result.provider.name,
            model=result.model,
            at=time.time(),
            failovers=len(result.attempts) - 1,
            tokens_in=result.tokens_in,
            tokens_out=result.tokens_out,
            latency_s=result.latency_s,
        )
        headers = {
            "X-OnFeather-Provider": result.provider.name,
            "X-OnFeather-Model": result.model,
            "X-OnFeather-Failovers": len(result.attempts) - 1,
        }
        if key:
            headers["X-OnFeather-Session"] = key
        if pin and pin[1] != result.model:
            # 20260725 RG The pinned model became unusable mid-flight.
            headers["X-OnFeather-Repinned"] = f"{pin[0]}/{pin[1]}"

        self._send(200, _as_openai_response(result), extra=headers)

    def _fail(self, error: CompletionError) -> None:
        """Answer a total failure with the status that provokes the right retry."""
        detail = "; ".join(
            f"{attempt.provider}: {attempt.error}" for attempt in error.attempts
        )
        message = f"{error} ({detail})" if detail else str(error)
        status = error.status

        if status == 429:
            wait = self.router.retry_after()
            self._error(
                429, message, "rate_limit_error", code="quota_exhausted",
                extra={"Retry-After": wait} if wait else None,
            )
        elif status >= 500:
            self._error(status, message, "no_route", code="upstream_unavailable")
        else:
            self._error(status, message, "invalid_request_error", code="upstream_rejected")

    def _status_payload(self) -> dict:
        now = datetime.now(timezone.utc)
        configured = configured_providers(self.router.registry)

        providers = []
        for provider in self.router.registry.usable():
            state = self.router.ledger.status(provider, now)
            providers.append({
                "name": provider.name,
                "label": provider.label,
                # 20260804 ++ RG #HASS An unconfigured provider still reads as full headroom.
                "configured": provider.name in configured,
                "api_key_env": provider.api_key_env,
                "available": state.available,
                "headroom": round(state.headroom, 4),
                "local": provider.local,
                "limits": [
                    {
                        "unit": limit.limit.unit,
                        "window": limit.limit.window,
                        "remaining": limit.remaining,
                        "limit": limit.effective_limit,
                        "authoritative": limit.authoritative,
                    }
                    for limit in state.limits
                ],
            })

        routable = candidates(
            self.router.registry, self.router.ledger,
            strategy=self.router.strategy, now=now,
        )
        last = self.router.last
        payload = {
            "strategy": self.router.strategy,
            "providers": providers,
            "next": (
                {
                    "provider": routable[0].provider.name,
                    "model": routable[0].model.id,
                    "id": f"{routable[0].provider.name}/{routable[0].model.id}",
                }
                if routable else None
            ),
            "current": last.as_dict() if last else None,
        }

        # 20260804 ++ RG #HASS Absent, not zeroed, when of-solo is unused.
        solo = companions.solo_counts()
        if solo is not None:
            payload["solo"] = solo
        return payload


def _as_openai_response(result) -> dict:
    """Return the assistant message as it arrived, normalised but not rebuilt.

    `content` stays null when there are tool calls rather than becoming an empty
    string: a client that reads text and tool calls as alternatives will happily
    treat "" as an answer and stop the loop.
    """
    return {
        "id": f"ofree-{int(time.time() * 1000):x}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": f"{result.provider.name}/{result.model}",
        "choices": [{
            "index": 0,
            "message": result.message,
            "finish_reason": result.finish_reason,
        }],
        "usage": {
            "prompt_tokens": result.tokens_in,
            "completion_tokens": result.tokens_out,
            "total_tokens": result.tokens_in + result.tokens_out,
        },
    }


def build(router: Router, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> ThreadingHTTPServer:
    """Create the server without starting it, so tests can drive it."""
    handler = type("BoundHandler", (Handler,), {"router": router})
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server


def serve(
    registry: Registry,
    ledger: Ledger,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    strategy: str = STRATEGY_BALANCED,
    verbose: bool = False,
    timeout: float | None = None,
    max_tokens: int | None = None,
    api_key: str | None = None,
) -> None:
    """Run until interrupted."""
    router = Router(
        registry, ledger, strategy=strategy, verbose=verbose,
        timeout=timeout, max_tokens=max_tokens, api_key=api_key,
    )
    server = build(router, host, port)
    print(f"onfeather-free listening on http://{host}:{port}/v1")
    print(f"  strategy: {strategy}")
    print(f"  routable: {len(candidates(registry, ledger, strategy=strategy))} provider/model pairs")
    print(f"  tool calling: {len(candidates(registry, ledger, requires={'tools'}))} pairs")
    print(f"  auth: {'API key required' if api_key else 'open'}")
    if not api_key and host not in ("127.0.0.1", "localhost", "::1"):
        # 20260804 ++ RG #HASS Off loopback with no key: anyone can spend the quota.
        print(f"    warning: reachable from the network — set {API_KEY_ENV} to require a key")
    print("\n  export OPENAI_BASE_URL=http://%s:%d/v1" % (host, port))
    # 20260804 ++ RG #HASS Named, not printed: this reaches docker logs.
    print(f"  export OPENAI_API_KEY={'$' + API_KEY_ENV if api_key else 'unused'}\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        server.server_close()
