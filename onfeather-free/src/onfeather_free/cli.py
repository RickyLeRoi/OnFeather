"""Command line entry point for `of-free`."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import __version__, client as client_module, config, discovery, registry as registry_module
from . import router as router_module
from . import server as server_module
from .budget import Ledger
from .client import Client, CompletionError
from .registry import Registry
from .router import NoRouteAvailable

DEFAULT_LEDGER = Path.home() / ".onfeather" / "free.db"


def build_parser() -> argparse.ArgumentParser:
    """Assemble the CLI. Separate from main() so tests can build it without
    running anything: a typo here used to survive a fully green suite."""
    parser = argparse.ArgumentParser(
        prog="of-free",
        description="Aggregate free LLM tiers behind one quota-aware router.",
    )
    parser.add_argument("--version", action="version", version=f"onfeather-free {__version__}")
    parser.add_argument(
        "--ledger", type=Path, default=DEFAULT_LEDGER, help="quota database (default: %(default)s)"
    )
    parser.add_argument("--registry", type=Path, help="provider registry YAML (default: packaged)")
    parser.add_argument(
        "--env",
        type=Path,
        help=f"file with API keys (default: first of {', '.join(str(p) for p in config.SEARCH_PATHS)})",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    status = subcommands.add_parser("status", help="show remaining quota per provider")
    status.add_argument("--all", action="store_true", help="include unconfigured providers")
    status.set_defaults(handler=_cmd_status)

    route = subcommands.add_parser("route", help="show where a request would go")
    route.add_argument("--capability", default="chat", help="required capability (default: chat)")
    route.add_argument(
        "--strategy", default=router_module.STRATEGY_BALANCED, choices=router_module.STRATEGIES
    )
    route.add_argument("--private", action="store_true", help="restrict to local providers")
    route.add_argument("-n", "--limit", type=int, default=5, help="how many options to list")
    route.set_defaults(handler=_cmd_route)

    providers = subcommands.add_parser("providers", help="list the registry")
    providers.set_defaults(handler=_cmd_providers)

    reset = subcommands.add_parser("reset", help="clear lockouts and usage for a provider")
    reset.add_argument("provider", nargs="?", help="provider name (default: every provider)")
    reset.set_defaults(handler=_cmd_reset)

    chat = subcommands.add_parser("chat", help="send one prompt through the router")
    chat.add_argument("prompt", help="the message to send")
    chat.add_argument("--capability", default="chat")
    chat.add_argument(
        "--strategy", default=router_module.STRATEGY_BALANCED, choices=router_module.STRATEGIES
    )
    chat.add_argument("--private", action="store_true", help="restrict to local providers")
    chat.add_argument("--max-tokens", type=int, default=512)
    chat.add_argument("--temperature", type=float)
    chat.add_argument("-v", "--verbose", action="store_true", help="show routing and failovers")
    chat.set_defaults(handler=_cmd_chat)

    serve = subcommands.add_parser("serve", help="run an OpenAI-compatible endpoint")
    serve.add_argument("--host", default=server_module.DEFAULT_HOST)
    serve.add_argument("--port", type=int, default=server_module.DEFAULT_PORT)
    serve.add_argument(
        "--strategy", default=router_module.STRATEGY_BALANCED, choices=router_module.STRATEGIES
    )
    serve.add_argument(
        "--timeout", type=float, default=client_module.DEFAULT_TIMEOUT,
        help="seconds to wait on a provider (default: %(default)s)",
    )
    serve.add_argument(
        "--max-tokens", type=int, default=client_module.DEFAULT_MAX_TOKENS,
        help="output ceiling for callers that send none (default: %(default)s)",
    )
    serve.add_argument(
        "--api-key",
        help=f"require this key from callers (default: ${server_module.API_KEY_ENV}, "
        "or no authentication when that is unset too)",
    )
    serve.add_argument("-v", "--verbose", action="store_true", help="log every request")
    serve.set_defaults(handler=_cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # 20260808 ** RG #Security The allowlist is what the registry declares it uses, nothing more.
    known = registry_module.load(args.registry)
    config.load_env(
        args.env,
        allowed=frozenset(p.api_key_env for p in known if p.api_key_env),
    )
    return args.handler(args)


def _open(args: argparse.Namespace) -> tuple[Registry, Ledger]:
    registry = registry_module.load(args.registry)
    # 20260725 RG Local model lists are per-machine; ask the runner.
    discovery.discover_local(registry)
    return registry, Ledger(args.ledger)


def _cmd_chat(args: argparse.Namespace) -> int:
    registry, ledger = _open(args)
    client = Client(registry, ledger)

    try:
        result = client.complete(
            [{"role": "user", "content": args.prompt}],
            capability=args.capability,
            strategy=args.strategy,
            private=args.private,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
    except CompletionError as error:
        print(f"error: {error}", file=sys.stderr)
        for attempt in error.attempts:
            print(f"  {attempt.provider}/{attempt.model}: {attempt.error}", file=sys.stderr)
        ledger.close()
        return 1

    if args.verbose or result.failed_over:
        for attempt in result.attempts[:-1]:
            print(f"  ✗ {attempt.provider}: {attempt.error}", file=sys.stderr)
        print(
            f"  → {result.provider.label} / {result.model}  "
            f"{result.tokens_in}+{result.tokens_out} tok  {result.latency_s:.2f}s",
            file=sys.stderr,
        )
        print(file=sys.stderr)

    print(result.text)
    ledger.close()
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    registry, ledger = _open(args)
    configured = router_module.configured_providers(registry)
    now = datetime.now(timezone.utc)

    rows = []
    for provider in registry.usable():
        known = provider.name in configured
        if not known and not args.all:
            continue
        state = ledger.status(provider, now)
        rows.append((provider, state, known))

    if not rows:
        print("No providers configured. Set an API key, for example:\n", file=sys.stderr)
        for provider in registry.remote()[:3]:
            print(f"  export {provider.api_key_env}=...", file=sys.stderr)
        return 1

    print(f"{'PROVIDER':<22}{'STATUS':<14}{'HEADROOM':<12}LIMITS")
    for provider, state, known in rows:
        if not known:
            label, headroom = "no key", "-"
        elif state.locked_until:
            # 20260808 ** RG #Security "rate limited" alone hides whether it is 60s or thirty years.
            wait = state.locked_until - now.timestamp()
            label = f"locked {_duration(wait)}" if wait > 0 else "ready"
            headroom = "0%"
        elif state.available:
            label, headroom = "ready", f"{state.headroom:.0%}"
        else:
            label, headroom = "exhausted", "0%"

        detail = ", ".join(
            f"{limit.limit.unit[:3]}/{limit.limit.window[:3]} "
            f"{limit.remaining}/{limit.effective_limit}"
            f"{'*' if limit.authoritative or limit.limit_observed else ''}"
            for limit in state.limits
        )
        print(f"{provider.label:<22}{label:<14}{headroom:<12}{detail or '—'}")

    print("\n* = confirmed by the provider's own rate-limit headers")
    if any(state.locked_until for _provider, state, known in rows if known):
        print("  a lockout that will not clear: of-free reset <provider>")
    ledger.close()
    return 0


def _duration(seconds: float) -> str:
    """Compact wait, so a lockout says whether it is a minute or a lifetime."""
    if seconds < 90:
        return f"{int(seconds)}s"
    if seconds < 5400:
        return f"{int(seconds // 60)}m"
    if seconds < 172800:
        return f"{int(seconds // 3600)}h"
    return f"{int(seconds // 86400)}d"


def _cmd_reset(args: argparse.Namespace) -> int:
    """Undo a lockout. The way back from a provider that asked for thirty years."""
    registry = registry_module.load(args.registry)
    if args.provider and args.provider not in registry.providers:
        print(f"error: unknown provider {args.provider!r}", file=sys.stderr)
        return 1

    ledger = Ledger(args.ledger)
    ledger.clear(args.provider)
    ledger.close()
    print(f"cleared {args.provider or 'every provider'}")
    return 0


def _cmd_route(args: argparse.Namespace) -> int:
    registry, ledger = _open(args)

    try:
        chosen = router_module.choose(
            registry,
            ledger,
            capability=args.capability,
            strategy=args.strategy,
            private=args.private,
        )
    except NoRouteAvailable as error:
        print(f"No route: {error}", file=sys.stderr)
        ledger.close()
        return 1

    options = router_module.candidates(
        registry,
        ledger,
        capability=args.capability,
        strategy=args.strategy,
        private=args.private,
    )

    print(f"→ {chosen.provider.label}  {chosen.model.id}")
    print(f"  {chosen.reason}")
    print(f"  {chosen.base_url}")

    if len(options) > 1:
        print("\nFallbacks:")
        for option in options[1 : args.limit]:
            print(
                f"  {option.provider.label:<20}{option.model.id:<40}"
                f"{option.status.headroom:.0%} left"
            )

    ledger.close()
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    registry, ledger = _open(args)
    # 20260804 ++ RG #HASS Env first, so the key stays out of the process listing.
    api_key = args.api_key or os.environ.get(server_module.API_KEY_ENV)
    try:
        server_module.serve(
            registry, ledger,
            host=args.host, port=args.port, strategy=args.strategy, verbose=args.verbose,
            timeout=args.timeout, max_tokens=args.max_tokens, api_key=api_key,
        )
    except OSError as error:
        print(f"error: cannot bind {args.host}:{args.port} — {error}", file=sys.stderr)
        return 1
    finally:
        ledger.close()
    return 0


def _cmd_providers(args: argparse.Namespace) -> int:
    registry = registry_module.load(args.registry)
    configured = router_module.configured_providers(registry)

    for provider in registry:
        marks = []
        if not provider.openai_compatible:
            marks.append("not OpenAI-compatible")
        if provider.local:
            marks.append("local")
        if provider.name in configured:
            marks.append("configured")
        elif provider.api_key_env:
            marks.append(f"needs {provider.api_key_env}")

        suffix = f"  ({', '.join(marks)})" if marks else ""
        print(f"{provider.label}{suffix}")
        print(f"  {provider.base_url}")
        print(f"  capabilities: {', '.join(sorted(provider.capabilities)) or '—'}")
        print(f"  models: {', '.join(model.id for model in provider.models) or '—'}")
        if provider.verified_at:
            print(f"  limits verified: {provider.verified_at}")
        if provider.notes:
            print(f"  note: {provider.notes}")
        print()

    if not os.environ.get("ONFEATHER_QUIET"):
        print("Limits are hints; the ledger corrects them from response headers and 429s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
