"""OpenAI-compatible endpoint.

Point any OpenAI client at this and it routes across free tiers without knowing
it. Built on the standard library rather than a web framework: a personal proxy
that needs `pip install fastapi uvicorn` to start is one more thing to maintain
than the job requires.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .budget import Ledger
from .client import Client, CompletionError
from .registry import Registry
from .router import STRATEGY_BALANCED, candidates

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 4141

# 20260726 ** RG Requests naming these get routed locally, whatever the strategy.
PRIVATE_MODEL_ALIASES = frozenset({"private", "local"})

# 20260726 ** RG Virtual model names meaning "you choose".
AUTO_MODEL_ALIASES = frozenset({"auto", "onfeather", "of-free", ""})


class Router:
    """Shared state for the handler, which cannot take constructor arguments."""

    def __init__(
        self,
        registry: Registry,
        ledger: Ledger,
        *,
        strategy: str = STRATEGY_BALANCED,
        verbose: bool = False,
    ) -> None:
        self.registry = registry
        self.ledger = ledger
        self.client = Client(registry, ledger)
        self.strategy = strategy
        self.verbose = verbose

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
        # 20260726 ** RG Unknown name: route it anyway rather than refusing.
        return "chat", False, None


class Handler(BaseHTTPRequestHandler):
    router: Router  # 20260726 ** RG Injected by serve().

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
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, message: str, kind: str = "invalid_request_error") -> None:
        self._send(status, {"error": {"message": message, "type": kind}})

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
            # 20260726 ** RG Streaming needs upstream passthrough; not built yet.
            self._error(
                400,
                "streaming is not supported yet; retry with \"stream\": false",
                "not_implemented",
            )
            return

        capability, private, _pinned = self.router.resolve(payload.get("model", "auto"))
        try:
            result = self.router.client.complete(
                messages,
                capability=capability,
                strategy=self.router.strategy,
                private=private,
                max_tokens=payload.get("max_tokens"),
                temperature=payload.get("temperature"),
            )
        except CompletionError as error:
            detail = "; ".join(
                f"{attempt.provider}: {attempt.error}" for attempt in error.attempts
            )
            # 20260726 ** RG 503, not 500: the request was fine, the quota was not.
            self._error(503, f"{error} ({detail})" if detail else str(error), "no_route")
            return

        self._send(
            200,
            _as_openai_response(result),
            # 20260726 ** RG Expose the choice so callers can see what served them.
            extra={
                "X-OnFeather-Provider": result.provider.name,
                "X-OnFeather-Model": result.model,
                "X-OnFeather-Failovers": str(len(result.attempts) - 1),
            },
        )

    def _status_payload(self) -> dict:
        now = datetime.now(timezone.utc)
        providers = []
        for provider in self.router.registry.usable():
            state = self.router.ledger.status(provider, now)
            providers.append({
                "name": provider.name,
                "label": provider.label,
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
        return {
            "providers": providers,
            "next": (
                {"provider": routable[0].provider.name, "model": routable[0].model.id}
                if routable else None
            ),
        }


def _as_openai_response(result) -> dict:
    return {
        "id": f"ofree-{int(time.time() * 1000):x}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": f"{result.provider.name}/{result.model}",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": result.text},
            "finish_reason": "stop",
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
) -> None:
    """Run until interrupted."""
    router = Router(registry, ledger, strategy=strategy, verbose=verbose)
    server = build(router, host, port)
    print(f"onfeather-free listening on http://{host}:{port}/v1")
    print(f"  strategy: {strategy}")
    print(f"  routable: {len(candidates(registry, ledger, strategy=strategy))} provider/model pairs")
    print("\n  export OPENAI_BASE_URL=http://%s:%d/v1" % (host, port))
    print("  export OPENAI_API_KEY=unused\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        server.server_close()
