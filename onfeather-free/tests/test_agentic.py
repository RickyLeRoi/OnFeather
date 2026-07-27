"""What an agentic client needs, end to end through a real socket.

These are the acceptance criteria of a tool-calling caller rather than a chat
box: the conversation arrives intact, tool calls survive the round trip in the
exact shape a validator expects, a schema is honoured or the attempt fails, and
the model does not change hands halfway through a run.
"""

from __future__ import annotations

import json
import threading
import time

import httpx
import pytest

from onfeather_free import client as client_module
from onfeather_free.budget import Ledger
from onfeather_free.client import Client
from onfeather_free.server import Router, build

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "browser_click",
            "description": "click an element",
            "parameters": {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {"ref": {"type": "string"}},
                "required": ["ref"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_snapshot",
            "parameters": {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
]

HEAL_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "ref": {"anyOf": [{"type": "string"}, {"type": "null"}], "description": "the locator"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reasoning": {"type": "string", "description": "why"},
    },
    "required": ["ref", "confidence", "reasoning"],
    "additionalProperties": False,
}

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "response", "strict": True, "schema": HEAL_SCHEMA},
}


# -- the harness ----------------------------------------------------------


class Gateway:
    def __init__(self, base: str, router: Router, seen: list[tuple[str, dict]]) -> None:
        self.base = base
        self.router = router
        self.seen = seen

    def post(self, payload: dict, **kwargs) -> httpx.Response:
        return httpx.post(f"{self.base}/v1/chat/completions", json=payload, timeout=30, **kwargs)

    @property
    def bodies(self) -> list[dict]:
        return [body for _url, body in self.seen]

    @property
    def hosts(self) -> list[str]:
        return [httpx.URL(url).host for url, _body in self.seen]


@pytest.fixture
def gateway(registry, environ, monkeypatch):
    """Start a gateway whose upstream is a function under the test's control."""
    running: list = []

    def start(responder) -> Gateway:
        for key, value in environ.items():
            monkeypatch.setenv(key, value)

        seen: list[tuple[str, dict]] = []

        def record(request: httpx.Request) -> httpx.Response:
            seen.append((str(request.url), json.loads(request.content)))
            return responder(request)

        ledger = Ledger(":memory:")
        router = Router(registry, ledger)
        router.client = Client(registry, ledger, transport=httpx.MockTransport(record))

        server = build(router, "127.0.0.1", 0)
        thread = threading.Thread(
            target=lambda: server.serve_forever(poll_interval=0.01), daemon=True
        )
        thread.start()
        running.append((server, ledger))
        return Gateway(f"http://127.0.0.1:{server.server_address[1]}", router, seen)

    yield start

    for server, ledger in running:
        server.shutdown()
        server.server_close()
        ledger.close()


def answers(content=None, tool_calls=None, *, finish_reason=None, status=200):
    """An upstream that always replies the same way."""
    message: dict = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls

    def responder(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={
            "id": "up-1",
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason or "stop"}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 5},
        })

    return responder


def long_conversation(turns: int = 29, filler: int = 5000) -> list[dict]:
    """A run's worth of accumulated history: over 32k tokens, tool calls included."""
    messages: list[dict] = [
        {"role": "system", "content": "You generate Playwright tests."},
        {"role": "user", "content": "Write a test for the checkout flow."},
    ]
    for index in range(turns):
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": f"call_{index}",
                "type": "function",
                "function": {"name": "browser_snapshot", "arguments": "{}"},
            }],
        })
        messages.append({
            "role": "tool",
            "tool_call_id": f"call_{index}",
            "content": f"snapshot {index} " + "x" * filler,
        })
    return messages


def tokens_in(messages: list[dict]) -> int:
    return len(json.dumps(messages)) // 4


# -- workload A: the agentic loop -----------------------------------------


def test_the_conversation_arrives_whole_and_in_order(gateway):
    """Truncating does not raise an error, it produces an agent that loops forever."""
    gate = gateway(answers("done"))
    messages = long_conversation()
    assert tokens_in(messages) > 32000

    response = gate.post({"model": "auto", "messages": messages, "tools": TOOLS,
                          "tool_choice": "auto"})

    assert response.status_code == 200
    assert gate.bodies[0]["messages"] == messages


def test_assistant_tool_calls_and_tool_results_are_propagated_verbatim(gateway):
    gate = gateway(answers("done"))
    messages = long_conversation(turns=2, filler=10)
    gate.post({"messages": messages, "tools": TOOLS})

    sent = gate.bodies[0]["messages"]
    assert sent[2]["content"] is None
    assert sent[2]["tool_calls"][0]["id"] == "call_0"
    assert sent[3] == {"role": "tool", "tool_call_id": "call_0", "content": sent[3]["content"]}


def test_a_conversation_too_big_for_a_model_does_not_go_to_it(gateway):
    """The local model tops out at 32k, and a run that overflows it fails at turn forty."""
    gate = gateway(answers("done"))
    gate.post({"messages": long_conversation(), "tools": TOOLS})
    assert "localhost" not in gate.hosts


def test_tools_and_tool_choice_reach_the_provider(gateway):
    gate = gateway(answers("done"))
    gate.post({"messages": [{"role": "user", "content": "hi"}],
               "tools": TOOLS, "tool_choice": "auto"})

    sent = gate.bodies[0]
    assert sent["tool_choice"] == "auto"
    assert [tool["function"]["name"] for tool in sent["tools"]] == [
        "browser_click", "browser_snapshot"
    ]
    assert sent["tools"][0]["function"]["parameters"]["required"] == ["ref"]


def test_a_caller_that_sends_no_ceiling_gets_one(gateway):
    """No max_tokens means the provider's default, and a low one truncates a tool call."""
    gate = gateway(answers("done"))
    gate.post({"messages": [{"role": "user", "content": "hi"}], "tools": TOOLS})

    assert gate.bodies[0]["max_tokens"] == client_module.DEFAULT_MAX_TOKENS


def test_an_explicit_ceiling_still_wins(gateway):
    gate = gateway(answers("done"))
    gate.post({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 128})
    assert gate.bodies[0]["max_tokens"] == 128


def test_sampling_parameters_are_not_invented(gateway):
    """The caller left these to the provider; making them up changes its behaviour."""
    gate = gateway(answers("done"))
    gate.post({"messages": [{"role": "user", "content": "hi"}]})

    sent = gate.bodies[0]
    assert "temperature" not in sent
    assert "top_p" not in sent and "stop" not in sent and "seed" not in sent


# -- tool calls coming back -----------------------------------------------


def test_tool_call_arguments_come_back_as_a_json_string(gateway):
    """Some providers hand back a parsed object; every client's validator wants text."""
    gate = gateway(answers(None, [{
        "id": "call_x", "type": "function",
        "function": {"name": "browser_click", "arguments": {"ref": "e7"}},
    }]))

    message = gate.post({"messages": [{"role": "user", "content": "hi"}],
                         "tools": TOOLS}).json()["choices"][0]["message"]
    arguments = message["tool_calls"][0]["function"]["arguments"]

    assert isinstance(arguments, str)
    assert json.loads(arguments) == {"ref": "e7"}


def test_content_stays_null_when_there_are_tool_calls(gateway):
    gate = gateway(answers(None, [{
        "id": "call_x", "function": {"name": "browser_click", "arguments": "{}"},
    }]))
    message = gate.post({"messages": [{"role": "user", "content": "hi"}],
                         "tools": TOOLS}).json()["choices"][0]["message"]

    assert message["content"] is None
    assert message["role"] == "assistant"


def test_a_tool_call_without_an_id_is_given_one(gateway):
    """That id comes back as `tool_call_id`; without it the conversation cannot continue."""
    gate = gateway(answers(None, [{"function": {"name": "browser_click", "arguments": "{}"}}]))
    call = gate.post({"messages": [{"role": "user", "content": "hi"}],
                      "tools": TOOLS}).json()["choices"][0]["message"]["tool_calls"][0]

    assert call["id"]
    assert call["type"] == "function"


def test_finish_reason_says_tool_calls(gateway):
    gate = gateway(answers(None, [{
        "id": "c", "function": {"name": "browser_click", "arguments": "{}"},
    }], finish_reason="stop"))

    body = gate.post({"messages": [{"role": "user", "content": "hi"}], "tools": TOOLS}).json()
    assert body["choices"][0]["finish_reason"] == "tool_calls"


def test_a_nameless_tool_call_fails_over_rather_than_being_returned(gateway):
    """A response the client's schema rejects is a failure, whatever its status code."""
    calls: list[int] = []

    def responder(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(200, json={"choices": [{"message": {
                "role": "assistant", "content": None,
                "tool_calls": [{"id": "c", "function": {"arguments": "{}"}}],
            }}]})
        return httpx.Response(200, json={"choices": [{"message": {
            "role": "assistant", "content": "recovered",
        }}]})

    gate = gateway(responder)
    response = gate.post({"messages": [{"role": "user", "content": "hi"}], "tools": TOOLS})

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "recovered"
    assert response.headers["x-onfeather-failovers"] == "1"


# -- workload B: the structured answer ------------------------------------


def test_a_native_schema_provider_is_preferred_and_gets_the_schema(gateway):
    gate = gateway(answers('{"ref": "e7", "confidence": 0.9, "reasoning": "matched"}'))
    response = gate.post({"messages": [{"role": "user", "content": "diagnose"}],
                          "response_format": RESPONSE_FORMAT})

    assert response.status_code == 200
    assert response.headers["x-onfeather-provider"] == "bigcontext"
    assert gate.bodies[0]["response_format"]["json_schema"]["name"] == "response"


def test_the_schema_is_translated_into_the_providers_dialect(gateway):
    """BigContext enforces OpenAI's strict subset, which 400s on numeric bounds."""
    gate = gateway(answers('{"ref": null, "confidence": 0.9, "reasoning": "y"}'))
    gate.post({"messages": [{"role": "user", "content": "d"}], "response_format": RESPONSE_FORMAT})

    schema = gate.bodies[0]["response_format"]["json_schema"]["schema"]
    assert "minimum" not in schema["properties"]["confidence"]
    assert schema["properties"]["ref"]["anyOf"] == [{"type": "string"}, {"type": "null"}]
    assert "$schema" not in schema


def test_a_bound_the_provider_never_saw_is_still_enforced(gateway):
    """Dropping `maximum` to get past strict mode must not mean accepting a 4."""
    gate = gateway(answers('{"ref": "e7", "confidence": 4, "reasoning": "y"}'))
    response = gate.post({"messages": [{"role": "user", "content": "d"}],
                          "response_format": RESPONSE_FORMAT})

    assert response.status_code == 503
    assert "maximum" in response.json()["error"]["message"]


def test_a_provider_without_native_schemas_is_told_in_the_prompt(gateway):
    """The caller does not repeat the schema in its prompt, so silence is not degradation."""
    gate = gateway(answers('{"ref": "e7", "confidence": 0.9, "reasoning": "y"}'))
    gate.router.registry.providers.pop("bigcontext")
    gate.router.registry.providers.pop("ollama")

    response = gate.post({"messages": [{"role": "system", "content": "be terse"},
                                       {"role": "user", "content": "diagnose"}],
                          "response_format": RESPONSE_FORMAT})

    assert response.status_code == 200
    sent = gate.bodies[0]
    assert sent["response_format"] == {"type": "json_object"}
    assert "confidence" in sent["messages"][0]["content"]
    assert sent["messages"][0]["content"].startswith("be terse")


def test_extra_keys_are_stripped_from_the_answer(gateway):
    """`strict: true` means the caller rejects anything it did not ask for."""
    gate = gateway(answers(
        '{"ref": "e7", "confidence": 0.9, "reasoning": "y", "notes": "chatty"}'
    ))
    body = gate.post({"messages": [{"role": "user", "content": "d"}],
                      "response_format": RESPONSE_FORMAT}).json()

    assert json.loads(body["choices"][0]["message"]["content"]) == {
        "ref": "e7", "confidence": 0.9, "reasoning": "y"
    }


def test_a_fenced_answer_is_unwrapped(gateway):
    gate = gateway(answers(
        '```json\n{"ref": "e7", "confidence": 0.9, "reasoning": "y"}\n```'
    ))
    body = gate.post({"messages": [{"role": "user", "content": "d"}],
                      "response_format": RESPONSE_FORMAT}).json()

    assert json.loads(body["choices"][0]["message"]["content"])["ref"] == "e7"


def test_an_answer_of_the_wrong_shape_fails_over(gateway):
    seen: list[str] = []

    def responder(request: httpx.Request) -> httpx.Response:
        seen.append(httpx.URL(str(request.url)).host)
        content = ('{"ref": "e7"}' if len(seen) == 1
                   else '{"ref": null, "confidence": 0.4, "reasoning": "y"}')
        return httpx.Response(200, json={"choices": [{"message": {
            "role": "assistant", "content": content}}]})

    gate = gateway(responder)
    response = gate.post({"messages": [{"role": "user", "content": "d"}],
                          "response_format": RESPONSE_FORMAT})

    assert response.status_code == 200
    assert response.headers["x-onfeather-failovers"] == "1"
    assert json.loads(response.json()["choices"][0]["message"]["content"])["confidence"] == 0.4


def test_when_nobody_can_satisfy_the_schema_the_caller_is_told_to_retry(gateway):
    gate = gateway(answers('{"ref": "e7"}'))
    response = gate.post({"messages": [{"role": "user", "content": "d"}],
                          "response_format": RESPONSE_FORMAT})

    assert response.status_code == 503
    assert "schema" in response.json()["error"]["message"]


# -- staying on one model -------------------------------------------------


def test_an_agentic_run_keeps_the_model_it_started_on(gateway):
    """Spending quota makes the first provider rank lower, and switching mid-run
    hands the transcript to a model that did not write it."""
    gate = gateway(answers("ok"))
    messages = [{"role": "system", "content": "gen"}, {"role": "user", "content": "go"}]

    first = gate.post({"messages": messages, "tools": TOOLS})
    second = gate.post({"messages": [*messages, {"role": "assistant", "content": "ok"}],
                        "tools": TOOLS})

    assert first.headers["x-onfeather-provider"] == second.headers["x-onfeather-provider"]
    assert first.headers["x-onfeather-model"] == second.headers["x-onfeather-model"]
    assert second.headers["x-onfeather-session"] == first.headers["x-onfeather-session"]


def test_a_one_shot_call_is_not_pinned_to_anything(gateway):
    """Every heal is independent, so it should get whoever has the most quota now."""
    gate = gateway(answers("ok"))
    first = gate.post({"messages": [{"role": "user", "content": "go"}]})
    second = gate.post({"messages": [{"role": "user", "content": "go"}]})

    assert "x-onfeather-session" not in first.headers
    assert second.headers["x-onfeather-provider"] != first.headers["x-onfeather-provider"]


def test_a_different_run_gets_its_own_pin(gateway):
    gate = gateway(answers("ok"))
    first = gate.post({"messages": [{"role": "user", "content": "run one"}], "tools": TOOLS})
    second = gate.post({"messages": [{"role": "user", "content": "run two"}], "tools": TOOLS})

    assert first.headers["x-onfeather-session"] != second.headers["x-onfeather-session"]


def test_an_explicit_session_header_overrides_the_fingerprint(gateway):
    gate = gateway(answers("ok"))
    headers = {"X-OnFeather-Session": "run-42"}
    first = gate.post({"messages": [{"role": "user", "content": "a"}]}, headers=headers)
    second = gate.post({"messages": [{"role": "user", "content": "b"}]}, headers=headers)

    assert first.headers["x-onfeather-session"] == "run-42"
    assert second.headers["x-onfeather-model"] == first.headers["x-onfeather-model"]


def test_losing_the_pinned_provider_is_reported_rather_than_hidden(gateway):
    gate = gateway(answers("ok"))
    messages = [{"role": "user", "content": "go"}]
    first = gate.post({"messages": messages, "tools": TOOLS})

    gate.router.ledger.lock_out(first.headers["x-onfeather-provider"], until=9e18)
    second = gate.post({"messages": messages, "tools": TOOLS})

    assert second.status_code == 200
    assert second.headers["x-onfeather-repinned"].endswith(first.headers["x-onfeather-model"])


# -- what the caller is told to do about failure --------------------------


def test_exhausted_quota_is_a_429_with_a_time_to_wait(gateway):
    """429 makes the client retry; 400 makes it give up on a window that reopens in a minute."""
    gate = gateway(answers("ok"))
    gate.router.registry.providers.pop("ollama")
    for name in ("fastcloud", "bigcontext"):
        gate.router.ledger.lock_out(name, until=time.time() + 45)

    response = gate.post({"messages": [{"role": "user", "content": "hi"}]})

    assert response.status_code == 429
    assert response.json()["error"]["type"] == "rate_limit_error"
    assert 0 < int(response.headers["retry-after"]) <= 45


def test_a_request_every_provider_rejects_is_not_worth_retrying(gateway):
    gate = gateway(lambda request: httpx.Response(400, text="unknown field: tool_choice"))
    gate.router.registry.providers.pop("ollama")

    response = gate.post({"messages": [{"role": "user", "content": "hi"}], "tools": TOOLS})

    assert response.status_code == 400
    assert "tool_choice" in response.json()["error"]["message"]


def test_an_upstream_outage_is_retryable(gateway):
    gate = gateway(lambda request: httpx.Response(500, text="down"))
    gate.router.registry.providers.pop("ollama")

    response = gate.post({"messages": [{"role": "user", "content": "hi"}]})
    assert response.status_code == 503


def test_a_bad_gateway_key_does_not_look_like_the_callers_fault(gateway):
    """Our credential is wrong, not their request: 4xx would stop a client that
    another provider could still have served."""
    gate = gateway(lambda request: httpx.Response(401, text="invalid api key"))
    gate.router.registry.providers.pop("ollama")

    response = gate.post({"messages": [{"role": "user", "content": "hi"}]})
    assert response.status_code == 503


def test_every_error_body_has_the_openai_shape(gateway):
    gate = gateway(lambda request: httpx.Response(500, text="down"))
    gate.router.registry.providers.pop("ollama")

    error = gate.post({"messages": [{"role": "user", "content": "hi"}]}).json()["error"]
    assert set(error) == {"message", "type", "param", "code"}


# -- time -----------------------------------------------------------------


def test_the_gateway_waits_ten_minutes_for_a_provider():
    """A long context on a free tier genuinely takes minutes."""
    assert client_module.DEFAULT_TIMEOUT == 600.0


def test_the_server_never_times_out_a_request_of_its_own_accord():
    """`timeout` on the handler would close the socket under a slow provider."""
    from onfeather_free.server import Handler

    assert getattr(Handler, "timeout", None) is None


def test_a_slow_provider_still_reaches_the_caller(gateway):
    def responder(_request: httpx.Request) -> httpx.Response:
        time.sleep(1.5)
        return httpx.Response(200, json={"choices": [{"message": {
            "role": "assistant", "content": "eventually"}}]})

    gate = gateway(responder)
    response = gate.post({"messages": [{"role": "user", "content": "hi"}]})

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "eventually"


def test_the_serve_command_can_override_both():
    from onfeather_free.cli import build_parser

    args = build_parser().parse_args(["serve", "--timeout", "90", "--max-tokens", "8192"])
    assert args.timeout == 90.0
    assert args.max_tokens == 8192


def test_the_serve_command_defaults_match_the_client():
    from onfeather_free.cli import build_parser

    args = build_parser().parse_args(["serve"])
    assert args.timeout == client_module.DEFAULT_TIMEOUT
    assert args.max_tokens == client_module.DEFAULT_MAX_TOKENS
