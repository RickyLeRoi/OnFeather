"""Egress control.

These are the tests that matter most in this repository. Everything else being
wrong costs a bad answer; this being wrong costs someone's private messages.
"""

from __future__ import annotations

import httpx
import pytest

from onfeather_solo import netguard
from onfeather_solo.netguard import EgressBlocked, LoopbackOnlyTransport, assert_local


# -- address classification ----------------------------------------------


@pytest.mark.parametrize("host", ["127.0.0.1", "127.0.0.53", "127.1.2.3", "::1"])
def test_loopback_addresses_are_recognised(host):
    assert netguard.is_loopback_address(host)


@pytest.mark.parametrize(
    "host",
    [
        "8.8.8.8",
        "192.168.1.10",          # 20260725 RG LAN is not this machine.
        "10.0.0.1",
        "169.254.169.254",       # 20260725 RG Cloud metadata endpoint.
        "0.0.0.0",
        "2001:4860:4860::8888",
        "fd00::1",
        "not-an-ip",
        "",
    ],
)
def test_everything_else_is_not_loopback(host):
    assert not netguard.is_loopback_address(host)


def test_private_lan_is_refused():
    """A NAS on the LAN is someone else's disk, not this machine."""
    with pytest.raises(EgressBlocked):
        assert_local("http://192.168.1.50:11434/v1")


# -- URL validation -------------------------------------------------------


def test_loopback_url_is_accepted_and_pinned():
    assert assert_local("http://127.0.0.1:11434/v1") == "http://127.0.0.1:11434/v1"


def test_localhost_is_resolved_to_a_literal_address():
    """Pinning closes the rebinding window: the name is never looked up again."""
    pinned = assert_local("http://localhost:11434/v1")
    assert netguard.is_loopback_address(httpx.URL(pinned).host)


def test_public_host_is_blocked():
    with pytest.raises(EgressBlocked, match="non-loopback"):
        assert_local("https://api.openai.com/v1")


def test_path_and_port_survive_pinning():
    pinned = assert_local("http://localhost:4141/v1/chat/completions")
    url = httpx.URL(pinned)
    assert url.port == 4141
    assert url.path == "/v1/chat/completions"


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://localhost/x", "gopher://localhost"])
def test_non_http_schemes_are_refused(url):
    with pytest.raises(EgressBlocked, match="scheme"):
        assert_local(url)


def test_url_without_a_host_is_refused():
    with pytest.raises(EgressBlocked):
        assert_local("http:///v1")


def test_unresolvable_host_is_refused_not_allowed():
    """Failing closed matters: a resolver error must never read as permission."""
    with pytest.raises(EgressBlocked):
        assert_local("http://this-host-does-not-exist.invalid/v1")


def test_mixed_resolution_is_refused(monkeypatch):
    """A name answering both 127.0.0.1 and a public address is rejected outright
    rather than accepted on the strength of the entry we prefer."""
    def fake(host, *args, **kwargs):
        return [
            (2, 1, 6, "", ("127.0.0.1", 0)),
            (2, 1, 6, "", ("203.0.113.7", 0)),
        ]

    monkeypatch.setattr(netguard.socket, "getaddrinfo", fake)
    with pytest.raises(EgressBlocked, match="203.0.113.7"):
        assert_local("http://sneaky.test/v1")


def test_empty_resolution_is_refused(monkeypatch):
    monkeypatch.setattr(netguard.socket, "getaddrinfo", lambda *a, **k: [])
    with pytest.raises(EgressBlocked, match="resolved to nothing"):
        assert_local("http://empty.test/v1")


# -- transport ------------------------------------------------------------


def recording_inner():
    seen = []

    class Inner(httpx.BaseTransport):
        def handle_request(self, request):
            seen.append(request)
            return httpx.Response(200, json={"ok": True})

        def close(self):
            pass

    return Inner(), seen


def test_transport_allows_loopback():
    inner, seen = recording_inner()
    with httpx.Client(transport=LoopbackOnlyTransport(inner)) as client:
        assert client.post("http://127.0.0.1:4141/v1/x", json={}).status_code == 200
    assert len(seen) == 1


def test_transport_blocks_public_hosts_before_sending():
    """The assertion that matters: the inner transport is never reached, so no
    byte of the body is written to a socket."""
    inner, seen = recording_inner()
    with httpx.Client(transport=LoopbackOnlyTransport(inner)) as client:
        with pytest.raises(EgressBlocked):
            client.post("https://api.groq.com/openai/v1/chat/completions",
                        json={"messages": [{"content": "private diary"}]})
    assert seen == [], "request reached the wire despite being blocked"


def test_transport_blocks_redirect_to_a_public_host():
    """A local server answering 302 to a remote URL must not become an exit."""
    class Redirector(httpx.BaseTransport):
        def handle_request(self, request):
            if request.url.host in ("127.0.0.1", "::1"):
                return httpx.Response(302, headers={"Location": "https://evil.test/v1"})
            return httpx.Response(200, json={"leaked": True})

        def close(self):
            pass

    with httpx.Client(transport=LoopbackOnlyTransport(Redirector()), follow_redirects=True) as c:
        with pytest.raises(EgressBlocked):
            c.post("http://127.0.0.1:4141/v1/x", json={})


def test_transport_pins_the_host_and_preserves_the_header():
    inner, seen = recording_inner()
    with httpx.Client(transport=LoopbackOnlyTransport(inner)) as client:
        client.get("http://localhost:11434/v1/models")

    assert netguard.is_loopback_address(seen[0].url.host)
    assert seen[0].headers["Host"] == "localhost"


def test_local_client_is_guarded_by_default():
    inner, seen = recording_inner()
    with netguard.local_client(inner=inner) as client:
        with pytest.raises(EgressBlocked):
            client.get("http://example.com/")
    assert seen == []
