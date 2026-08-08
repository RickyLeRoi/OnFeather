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

import contextlib
import hashlib
import hmac
import json
import sys
import threading
import time
import traceback
from collections import OrderedDict
from collections.abc import Callable
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

# 20260808 ** RG #Security The only addresses we are willing to serve without a key.
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

# 20260808 ** RG #Security Anything else on a POST is a CORS simple request in disguise.
ALLOWED_CONTENT_TYPES = frozenset({"application/json", "text/json"})

# 20260725 RG Routed locally whatever the strategy.
PRIVATE_MODEL_ALIASES = frozenset({"private", "local"})

AUTO_MODEL_ALIASES = frozenset({"auto", "onfeather", "of-free", ""})

# 20260725 RG Pin a run explicitly instead of being fingerprinted.
SESSION_HEADER = "X-OnFeather-Session"

SESSION_TTL_SECONDS = 1800.0

# 20260808 ** RG #Security Hard caps: the key comes from the caller, so the map is hostile.
MAX_SESSIONS = 4096
MAX_SESSION_KEY_CHARS = 128

# 20260808 ** RG #Security A long agentic conversation fits in a few MB; past that it is an attack.
MAX_BODY_BYTES = 32 * 1024 * 1024

# 20260808 ** RG #Security How much is drained before a 401, after which we just hang up.
DRAIN_LIMIT_BYTES = 64 * 1024

# 20260808 ** RG #Security One thread and one descriptor per connection, so the count is ours.
MAX_CONNECTIONS = 64

# 20260808 ** RG #Security Slowloris budget: how long a socket may say nothing at all.
IDLE_TIMEOUT_SECONDS = 60.0

CHARS_PER_TOKEN = 4

# 20260808 ** RG #Security Nearer 1.5 in CJK: undercounting picks a model that cannot hold it.
CJK_CHARS_PER_TOKEN = 1.5

WIDE_CHAR_WEIGHT = CHARS_PER_TOKEN / CJK_CHARS_PER_TOKEN


class Sessions:
    """Which model each ongoing run is pinned to.

    An agentic client sends no session id, so one is derived from the opening of
    the conversation: every turn of a run resends the same system prompt and
    first user message, and a different run has a different pair. It is a
    fingerprint rather than a guarantee — two identical runs share a pin, which
    costs nothing because they would want the same answer anyway.

    Bounded by construction. The key can also arrive from the caller, so an
    unbounded map is an unbounded allocation an anonymous request can drive, and
    a full-scan expiry is a quadratic cost it can drive too.
    """

    def __init__(self, ttl: float = SESSION_TTL_SECONDS, capacity: int = MAX_SESSIONS) -> None:
        self.ttl = ttl
        self.capacity = capacity
        self._pins: OrderedDict[str, tuple[str, str, float]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str | None) -> tuple[str, str] | None:
        if not key:
            return None
        with self._lock:
            entry = self._pins.get(key)
            if entry is None:
                return None
            if entry[2] < time.time() - self.ttl:
                del self._pins[key]
                return None
            self._pins.move_to_end(key)
            return (entry[0], entry[1])

    def remember(self, key: str | None, provider: str, model: str) -> None:
        if not key:
            return
        with self._lock:
            self._pins[key] = (provider, model, time.time())
            self._pins.move_to_end(key)
            self._trim()

    def _trim(self) -> None:
        """Drop expired entries from the oldest end, then enforce the cap.

        The oldest end is the only place an expired entry can be, so this is
        O(1) amortised where a full scan was O(n) on every single request.
        """
        cutoff = time.time() - self.ttl
        while self._pins:
            oldest = next(iter(self._pins))
            if self._pins[oldest][2] >= cutoff:
                break
            del self._pins[oldest]
        while len(self._pins) > self.capacity:
            self._pins.popitem(last=False)


def session_key(payload: dict, headers: Any = None) -> str | None:
    """Identify the run this request belongs to, if it is part of one.

    Only tool-carrying requests get a key. A one-shot call is stateless by
    construction and pinning it would hand it yesterday's model for no reason.
    """
    explicit = headers.get(SESSION_HEADER) if headers is not None else None
    if isinstance(explicit, str) and explicit.strip():
        # 20260808 ** RG #Security Truncated: this lands in a map and in a response header.
        return explicit.strip()[:MAX_SESSION_KEY_CHARS]

    if not payload.get("tools"):
        return None

    messages = payload.get("messages") or []
    opening = json.dumps(messages[:2], sort_keys=True, ensure_ascii=False)[:8000]
    return hashlib.sha256(opening.encode()).hexdigest()[:16]


def _weighted_length(value: Any) -> float:
    """Characters in `value`, charged by how many tokens those characters cost.

    Walked rather than serialised: a dumps() of the whole payload built ~160 KB
    per turn of a long agentic run purely to call len() on it. ASCII, which is
    almost everything, is a flag check and a length.

    A CJK character is one character and two thirds of a token, so counting
    characters undercounts it fourfold — and this number is what rules a model
    out. The UTF-8 length pays for them without a per-character loop: three
    bytes for a CJK codepoint against one for ASCII, so half the excess is a
    good count of the wide ones.
    """
    if isinstance(value, str):
        if value.isascii():
            return len(value)
        wide = (len(value.encode("utf-8", "ignore")) - len(value)) / 2
        return (len(value) - wide) + wide * WIDE_CHAR_WEIGHT
    if isinstance(value, dict):
        return sum(_weighted_length(k) + _weighted_length(v) for k, v in value.items())
    if isinstance(value, list):
        return sum(_weighted_length(item) for item in value)
    return len(str(value)) if value is not None else 0


def estimated_tokens(payload: dict) -> int:
    """How big this request is, near enough to exclude a model that cannot hold it."""
    material = _weighted_length(payload.get("messages") or []) + _weighted_length(
        payload.get("tools") or []
    )
    return int(material // CHARS_PER_TOKEN)


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

    def close(self) -> None:
        """Release the upstream connection pool. The ledger is the caller's."""
        self.client.close()

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

    def resolve(self, requested: object) -> tuple[str, bool, tuple[str, str] | None]:
        """Map a requested model onto (capability, private, pinned pair).

        Three shapes are accepted: `auto` to let the router decide, `private` to
        force a local model, and `provider/model` to pin one explicitly.
        """
        # 20260808 ** RG #Security `model` arrives from JSON: it can be a dict, a number, null.
        name = requested.strip() if isinstance(requested, str) else ""
        if name.lower() in PRIVATE_MODEL_ALIASES:
            return "chat", True, None
        if name.lower() in AUTO_MODEL_ALIASES:
            return "chat", False, None
        if "/" in name:
            provider_name, model_id = name.split("/", 1)
            if provider_name in self.registry.providers and model_id:
                return "chat", False, (provider_name, model_id)
        # 20260725 RG Unknown name: route it rather than refuse.
        return "chat", False, None

    def retry_after(self) -> int | None:
        """Seconds until the first quota window turns over, if any is locked."""
        now = datetime.now(timezone.utc)
        # 20260808 ** RG #Security Only lockouts are needed here; one read gives them all.
        lockouts = self.ledger.snapshot(now, providers=self.registry.providers).lockouts
        pending = [
            lockouts[provider.name] - time.time()
            for provider in self.registry.usable()
            if provider.name in lockouts
        ]
        return max(1, int(min(pending))) if pending else None


def _hostname(header: str) -> str:
    """The host part of a `Host` header, without its port or IPv6 brackets."""
    host = header.strip()
    if host.startswith("["):
        return host[1:].partition("]")[0].lower()
    return host.partition(":")[0].lower()


class Handler(BaseHTTPRequestHandler):
    router: Router

    # 20260808 ** RG #Security Bound by build() to whatever address was published.
    allowed_hosts: frozenset[str] = LOOPBACK_HOSTS

    protocol_version = "HTTP/1.1"
    server_version = "onfeather-free"

    # 20260808 ** RG #Security A socket that says nothing must not hold a thread for ever.
    timeout = IDLE_TIMEOUT_SECONDS

    # 20260808 ** RG #Security Whether this request already has an answer on the wire.
    _replied = False

    # -- plumbing ---------------------------------------------------------

    def log_message(self, fmt: str, *args: Any) -> None:
        if self.router.verbose:
            super().log_message(fmt, *args)

    def _send(self, status: int, payload: dict, *, extra: dict | None = None) -> None:
        body = json.dumps(payload).encode()
        self._replied = True
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        # 20260808 ** RG #Security The path is echoed in a 404 body; nosniff keeps it JSON.
        self.send_header("X-Content-Type-Options", "nosniff")
        # 20260808 ** RG #Security Answers and quota figures are nobody's to cache.
        self.send_header("Cache-Control", "no-store")
        if self.close_connection:
            # 20260808 ** RG #Security Announce the hang-up: a silent close is a desync of its own.
            self.send_header("Connection", "close")
        for key, value in (extra or {}).items():
            self.send_header(key, str(value))
        self.end_headers()
        self.wfile.write(body)

    def send_error(self, code: int, message: str | None = None, explain: str | None = None) -> None:
        """Answer the base class's own refusals the way every other reply is answered.

        An unsupported method or an unparsable request line is handled above us,
        and the default page is HTML echoing part of the request — the one reply
        that would carry neither `nosniff` nor a shape a client can read.
        """
        if getattr(self, "command", None) == "HEAD":
            super().send_error(code, message, explain)
            return
        self.close_connection = True
        kind = "invalid_request_error" if code < 500 else "internal_error"
        with contextlib.suppress(OSError):
            self._error(code, message or explain or "bad request", kind)

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

        Compared as bytes, in the encodings the two sides actually arrived in:
        headers reach us decoded as latin-1, the key is held as text. As `str`,
        `compare_digest` raised TypeError on any non-ASCII byte, which turned an
        unauthenticated 401 into a dead handler thread.
        """
        expected = self.router.api_key
        if not expected or path in PUBLIC_PATHS:
            return True

        offered = (self.headers.get("Authorization") or "").strip()
        if offered[:7].lower() == "bearer ":
            offered = offered[7:].strip()
        if not offered:
            return False
        return hmac.compare_digest(
            offered.encode("latin-1", "ignore"), expected.encode("utf-8", "surrogateescape")
        )

    def _host_ok(self) -> bool:
        """Whether `Host` names an address this server actually published.

        This is the defence against DNS rebinding, and it is only needed while
        no key is set: JavaScript cannot forge Host, so a domain that re-resolves
        to loopback still arrives here under its own name and is turned away.
        Once a key is required, authentication is the defence, and a caller that
        legitimately reaches the router by its LAN address must keep working.
        """
        if self.router.api_key:
            return True
        return _hostname(self.headers.get("Host") or "") in self.allowed_hosts

    def _from_browser(self) -> bool:
        # 20260808 ** RG #Security No OpenAI client sends these headers; a browser always does.
        return bool(self.headers.get("Origin") or self.headers.get("Sec-Fetch-Site"))

    def _guard(self, path: str) -> bool:
        """Every check a request passes before the router sees it."""
        if self._from_browser():
            # 20260808 ** RG #Security Closing beats draining: nothing here is worth reading.
            self.close_connection = True
            self._error(403, "browser-originated requests are not accepted", "forbidden")
            return False
        if not self._host_ok():
            self.close_connection = True
            self._error(
                421,
                "unrecognised Host header; reach this server as "
                f"{' or '.join(sorted(self.allowed_hosts))}, or set {API_KEY_ENV} "
                f"to accept requests by any name",
                "misdirected_request",
            )
            return False
        if not self._authorised(path):
            self._unauthorised()
            return False
        return True

    def _unauthorised(self) -> None:
        # 20260804 ++ RG #HASS Drain the body, or keep-alive reads it as the next request.
        # 20260808 ** RG #Security A fixed ceiling, then hang up: never read GB from a stranger.
        length = self._content_length() or 0
        if 0 < length <= DRAIN_LIMIT_BYTES:
            with contextlib.suppress(OSError):
                self.rfile.read(length)
        elif length or self.headers.get("Content-Length") or self.headers.get("Transfer-Encoding"):
            self.close_connection = True

        self._error(
            401,
            "missing or invalid API key; send it as `Authorization: Bearer <key>`",
            code="invalid_api_key",
            extra={"WWW-Authenticate": 'Bearer realm="onfeather-free"'},
        )

    def _content_length(self) -> int | None:
        """Declared body size, or None when it is absent, malformed or absurd."""
        raw = self.headers.get("Content-Length")
        if raw is None:
            return None
        try:
            length = int(raw)
        except ValueError:
            return None
        return length if 0 <= length <= MAX_BODY_BYTES else None

    def _read_json(self) -> dict | None:
        """The request body, or None when there is nothing usable in it.

        Every path that cannot account for exactly how many bytes the body held
        closes the connection: bytes left in the buffer of a keep-alive
        connection are read as the next request line, which is a desync a
        reverse proxy in front of us would turn into request smuggling.
        """
        # 20260808 ** RG #Security chunked is unsupported; reading it as 0 bytes desyncs keep-alive.
        if self.headers.get("Transfer-Encoding"):
            self.close_connection = True
            return None

        length = self._content_length()
        if length is None:
            # 20260808 ** RG #Security Absent is fine; unreadable or oversized loses the stream.
            if self.headers.get("Content-Length") is not None:
                self.close_connection = True
            return None
        if length <= 0:
            return None

        try:
            body = self.rfile.read(length)
        except OSError:
            self.close_connection = True
            return None
        if len(body) != length:
            self.close_connection = True
            return None

        try:
            parsed = json.loads(body)
        except ValueError:
            return None
        # 20260808 ** RG #Security A body of `[]` or `"x"` is valid JSON and has no .get().
        return parsed if isinstance(parsed, dict) else None

    # -- routes -----------------------------------------------------------

    def _dispatch(self, route: Callable[[str], None]) -> None:
        """Run one route, turning anything unforeseen into a reply, not a reset.

        A caller that gets a connection reset reads it as "try again", so an
        input nobody anticipated becomes a retry loop against a server that
        keeps killing its own threads. It has to become a reply.
        """
        path = self.path.split("?", 1)[0].rstrip("/")
        self._replied = False
        try:
            if self._guard(path):
                route(path)
        except Exception:  # noqa: BLE001
            # 20260808 ** RG #Security The traceback goes to stderr; the caller gets JSON.
            traceback.print_exc()
            if self._replied:
                # 20260808 ** RG #Security One reply per request: a second one desyncs keep-alive.
                self.close_connection = True
                return
            with contextlib.suppress(OSError):
                self._error(500, "internal error", "internal_error")

    def do_GET(self) -> None:  # noqa: N802
        self._dispatch(self._get)

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch(self._post)

    def _get(self, path: str) -> None:
        if path in ("/health", "/healthz"):
            self._send(200, {"status": "ok"})
        elif path in ("/v1/models", "/models"):
            self._send(200, {"object": "list", "data": self.router.models()})
        elif path in ("/v1/status", "/status"):
            self._send(200, self._status_payload())
        else:
            self._error(404, f"unknown path {path}", "not_found")

    def _post(self, path: str) -> None:
        if path not in ("/v1/chat/completions", "/chat/completions"):
            self._error(404, f"unknown path {path}", "not_found")
            return

        media_type = (self.headers.get("Content-Type") or "").partition(";")[0].strip().lower()
        if media_type and media_type not in ALLOWED_CONTENT_TYPES:
            # 20260808 ** RG #Security text/plain is what makes a cross-origin POST preflight-free.
            self._read_json()
            self._error(415, "request body must be application/json", "unsupported_media_type")
            return

        declared = self.headers.get("Content-Length")
        if declared is not None and self._content_length() is None:
            # 20260808 ** RG #Security Refuse on the declared size, before reading a byte of it.
            self.close_connection = True
            if declared.strip().isdigit():
                self._error(
                    413, f"request body exceeds {MAX_BODY_BYTES} bytes", "payload_too_large"
                )
            else:
                self._error(400, "Content-Length is not a readable byte count")
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

        capability, private, pinned = self.router.resolve(payload.get("model", "auto"))
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
                # 20260808 ** RG #Security An explicit pin binds: ignoring it sends the prompt away.
                require=pinned,
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
        """Answer a total failure with the status that provokes the right retry.

        The caller is told which providers were tried and how each one refused,
        never what they said while refusing: those bodies carry organisation and
        project ids, and on the loopback default the caller is nobody in
        particular. The full text goes to stderr, where the operator is.
        """
        upstream = "; ".join(
            f"{attempt.provider}/{attempt.model}: {attempt.detail}"
            for attempt in error.attempts
            if attempt.detail
        )
        if upstream:
            # 20260808 ** RG #Security Unconditional: log_error is silenced without --verbose.
            print(f"upstream refused: {upstream}", file=sys.stderr)

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
        # 20260808 ** RG #Security Shared with candidates() below: Home Assistant polls this.
        view = self.router.ledger.snapshot(now, providers=self.router.registry.providers)

        providers = []
        for provider in self.router.registry.usable():
            state = self.router.ledger.status(provider, now, snapshot=view)
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
            strategy=self.router.strategy, now=now, snapshot=view,
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


class BoundedServer(ThreadingHTTPServer):
    """A threading server that will not spawn more threads than it can afford.

    One connection is one thread and one descriptor, and both are free to open
    before anything has been authenticated. The ceiling is refused in the accept
    loop, so a flood costs a socket close rather than a thread.
    """

    daemon_threads = True

    def __init__(self, *args: Any, max_connections: int = MAX_CONNECTIONS, **kwargs: Any) -> None:
        self._slots = threading.BoundedSemaphore(max_connections)
        super().__init__(*args, **kwargs)

    def process_request(self, request: Any, client_address: Any) -> None:
        if not self._slots.acquire(blocking=False):
            # 20260808 ** RG #Security close_request, not shutdown_request: nothing to release yet.
            self.close_request(request)
            return
        super().process_request(request, client_address)

    def shutdown_request(self, request: Any) -> None:
        try:
            super().shutdown_request(request)
        finally:
            self._slots.release()


def build(router: Router, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> ThreadingHTTPServer:
    """Create the server without starting it, so tests can drive it."""
    # 20260808 ** RG #Security The allowlist is what we published, plus loopback. Nothing else.
    allowed = frozenset({host.lower(), *LOOPBACK_HOSTS})
    handler = type("BoundHandler", (Handler,), {"router": router, "allowed_hosts": allowed})
    return BoundedServer((host, port), handler)


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
    if not api_key and host not in LOOPBACK_HOSTS:
        # 20260808 ** RG #Security Off loopback with no key: refuse to start, do not warn.
        raise SystemExit(
            f"refusing to bind {host} without authentication: anyone who reaches "
            f"the port spends your quota and reads your answers.\n"
            f"  export {API_KEY_ENV}=$(python3 -c "
            f"'import secrets; print(secrets.token_urlsafe(32))')\n"
            f"  or bind {DEFAULT_HOST} to stay on this machine."
        )

    router = Router(
        registry, ledger, strategy=strategy, verbose=verbose,
        timeout=timeout, max_tokens=max_tokens, api_key=api_key,
    )
    server = build(router, host, port)
    print(f"onfeather-free listening on http://{host}:{port}/v1")
    print(f"  strategy: {strategy}")
    print(f"  routable: {len(candidates(registry, ledger, strategy=strategy))} provider/model pairs")
    print(f"  tool calling: {len(candidates(registry, ledger, requires={'tools'}))} pairs")
    print(f"  auth: {'API key required' if api_key else 'open (loopback only)'}")
    print("\n  export OPENAI_BASE_URL=http://%s:%d/v1" % (host, port))
    # 20260804 ++ RG #HASS Named, not printed: this reaches docker logs.
    print(f"  export OPENAI_API_KEY={'$' + API_KEY_ENV if api_key else 'unused'}\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        server.server_close()
        router.close()
