"""Egress control for content-bearing requests.

`of-solo` ingests private material — chat exports, personal notes. The guarantee
it needs is not "we prefer to stay local" but "this process cannot send your
content off the machine", and a routing preference cannot provide that. A
misconfigured `OPENAI_BASE_URL`, a registry entry with the wrong capability, or
an ordinary typo would all leak silently and irreversibly.

So egress is enforced here, below the routing layer:

* the destination host must resolve **entirely** to loopback addresses;
* the URL is then rewritten to the literal address that was checked, so the
  name is never resolved a second time (a hostname that answers 127.0.0.1 to
  our check and a public address to the actual connection is the classic DNS
  rebinding move);
* anything else raises before a single byte is written.

What this does not defend against, stated plainly: a hostile process on the same
machine, a local server that forwards elsewhere (`OLLAMA_HOST` pointing at a
remote box is a real example), or the operator deliberately passing
`--allow-remote`. See SECURITY.md.
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse, urlunparse

import httpx

LOOPBACK_NETWORKS = (
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("::1/128"),
)

# 20260726 ** RG Only these schemes carry a body we care about.
ALLOWED_SCHEMES = frozenset({"http", "https"})


class EgressBlocked(Exception):
    """Raised when a request would leave the machine."""


def is_loopback_address(host: str) -> bool:
    """True if `host` is a literal loopback IP."""
    try:
        return any(ipaddress.ip_address(host) in net for net in LOOPBACK_NETWORKS)
    except ValueError:
        return False


def resolve_loopback(host: str) -> str:
    """Resolve `host` and return a literal loopback address, or raise.

    Every returned address must be loopback. A name that resolves to both
    127.0.0.1 and a public address is rejected rather than being allowed on the
    strength of the entry we happen to like.
    """
    if is_loopback_address(host):
        return host

    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as error:
        raise EgressBlocked(f"cannot resolve {host!r}: {error}") from error

    addresses = {info[4][0] for info in infos}
    if not addresses:
        raise EgressBlocked(f"{host!r} resolved to nothing")

    offending = sorted(a for a in addresses if not is_loopback_address(a))
    if offending:
        raise EgressBlocked(
            f"{host!r} resolves to non-loopback address(es): {', '.join(offending)}"
        )
    # 20260726 ** RG Prefer IPv4 for the widest local-server compatibility.
    return sorted(addresses, key=lambda a: ":" in a)[0]


def assert_local(url: str) -> str:
    """Validate a URL for local-only use, returning it pinned to a literal IP."""
    parts = urlparse(url)
    if parts.scheme not in ALLOWED_SCHEMES:
        raise EgressBlocked(f"refusing scheme {parts.scheme!r} in {url!r}")
    if not parts.hostname:
        raise EgressBlocked(f"no host in {url!r}")

    address = resolve_loopback(parts.hostname)
    host = f"[{address}]" if ":" in address else address
    netloc = f"{host}:{parts.port}" if parts.port else host
    return urlunparse(parts._replace(netloc=netloc))


class LoopbackOnlyTransport(httpx.BaseTransport):
    """Transport that refuses to talk to anything but this machine.

    Wrapping the transport rather than checking at call sites is deliberate:
    every future code path gets the check for free, including ones written by
    someone who has never read this file.
    """

    def __init__(self, inner: httpx.BaseTransport | None = None) -> None:
        self._inner = inner or httpx.HTTPTransport()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if not host:
            raise EgressBlocked("request has no host")

        address = resolve_loopback(host)
        if address != host:
            # 20260726 ** RG Pin to the checked address; never resolve twice.
            request.url = request.url.copy_with(host=address)
            request.headers["Host"] = host

        return self._inner.handle_request(request)

    def close(self) -> None:
        self._inner.close()


def local_client(*, timeout: float = 300.0, inner: httpx.BaseTransport | None = None) -> httpx.Client:
    """An httpx client that physically cannot reach the network."""
    return httpx.Client(timeout=timeout, transport=LoopbackOnlyTransport(inner))
