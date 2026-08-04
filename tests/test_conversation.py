"""Conversation tests, concentrated on the two lossy conversions.

Everything the agent does is a translation: chat log to OpenAI messages on the
way out, OpenAI message to chat log delta on the way back. Both have a detail
that breaks the tool-calling loop silently when it is wrong, so both are tested
for that detail rather than for their happy path.
"""

from __future__ import annotations

import pytest

from custom_components.onfeather.conversation import (
    _as_delta_stream,
    _assistant_message,
    _to_message,
    _to_tool_input,
    _trim,
)
from homeassistant.components import conversation
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm

AGENT = "conversation.onfeather"


# -- chat log to OpenAI ---------------------------------------------------


def test_user_and_system_content():
    assert _to_message(conversation.SystemContent(content="be brief")) == {
        "role": "system",
        "content": "be brief",
    }
    assert _to_message(conversation.UserContent(content="hello")) == {
        "role": "user",
        "content": "hello",
    }


def test_assistant_content_without_tools():
    content = conversation.AssistantContent(agent_id=AGENT, content="hi")
    assert _to_message(content) == {"role": "assistant", "content": "hi"}


def test_tool_calls_are_serialised_the_way_openai_wants_them():
    """`arguments` is a JSON string, not an object. Providers reject the object."""
    content = conversation.AssistantContent(
        agent_id=AGENT,
        content=None,
        tool_calls=[
            llm.ToolInput(id="call_1", tool_name="HassTurnOn", tool_args={"name": "kitchen"})
        ],
    )
    message = _to_message(content)

    assert message["content"] is None
    call = message["tool_calls"][0]
    assert call == {
        "id": "call_1",
        "type": "function",
        "function": {"name": "HassTurnOn", "arguments": '{"name":"kitchen"}'},
    }


def test_tool_results_quote_the_call_they_answer():
    content = conversation.ToolResultContent(
        agent_id=AGENT, tool_call_id="call_1", tool_name="HassTurnOn", tool_result={"ok": True}
    )
    assert _to_message(content) == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": '{"ok":true}',
    }


def test_attachments_are_refused_rather_than_dropped():
    """Silently discarding the image a user attached is worse than saying so."""
    content = conversation.UserContent(content="what is this?", attachments=[object()])
    with pytest.raises(HomeAssistantError, match="attachments"):
        _to_message(content)


# -- OpenAI to chat log ---------------------------------------------------


async def collect(message: dict) -> list[dict]:
    return [delta async for delta in _as_delta_stream(message)]


async def test_a_whole_answer_is_one_delta():
    """The chat log flushes what it holds at end of stream, so one delta is a stream."""
    deltas = await collect({"role": "assistant", "content": "hello"})
    assert deltas == [{"role": "assistant", "content": "hello"}]


async def test_tool_calls_become_tool_inputs():
    deltas = await collect(
        {
            "tool_calls": [
                {
                    "id": "call_9",
                    "type": "function",
                    "function": {"name": "HassTurnOn", "arguments": '{"name": "kitchen"}'},
                }
            ]
        }
    )
    call = deltas[0]["tool_calls"][0]
    assert call.tool_name == "HassTurnOn"
    assert call.tool_args == {"name": "kitchen"}


def test_the_upstream_call_id_is_kept():
    """It is what pairs the result back to the call; a fresh one orphans it."""
    call = _to_tool_input(
        {"id": "call_9", "function": {"name": "HassTurnOn", "arguments": "{}"}}
    )
    assert call.id == "call_9"


def test_a_missing_id_still_yields_a_usable_call():
    call = _to_tool_input({"function": {"name": "HassTurnOn", "arguments": "{}"}})
    assert call.id


def test_empty_arguments_are_an_empty_object_not_a_crash():
    assert _to_tool_input({"function": {"name": "x", "arguments": ""}}).tool_args == {}


def test_arguments_that_are_not_json_say_so():
    with pytest.raises(HomeAssistantError, match="not valid JSON"):
        _to_tool_input({"function": {"name": "x", "arguments": "{not json"}})


def test_an_empty_answer_names_the_model_that_gave_it():
    """Better than the chat log's own 'unable to get response' further down."""
    response = {
        "model": "groq/llama-3.3-70b",
        "choices": [{"message": {"role": "assistant", "content": ""}, "finish_reason": "length"}],
    }
    with pytest.raises(HomeAssistantError, match="groq/llama-3.3-70b.*length"):
        _assistant_message(response)


def test_a_response_with_no_choices():
    with pytest.raises(HomeAssistantError, match="no message"):
        _assistant_message({"choices": []})


def test_a_tool_call_with_no_text_is_a_valid_answer():
    message = {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1"}]}
    assert _assistant_message({"choices": [{"message": message}]}) is message


# -- trimming -------------------------------------------------------------


def conversation_of(turns: int) -> list[dict]:
    messages = [{"role": "system", "content": "prompt"}]
    for index in range(turns):
        messages.append({"role": "user", "content": f"q{index}"})
        messages.append({"role": "assistant", "content": f"a{index}"})
    return messages


def test_short_conversations_are_untouched():
    messages = conversation_of(3)
    assert _trim(messages, 20) == messages


def test_zero_keeps_everything():
    messages = conversation_of(50)
    assert _trim(messages, 0) == messages


def test_the_system_prompt_always_survives():
    trimmed = _trim(conversation_of(30), 2)
    assert trimmed[0] == {"role": "system", "content": "prompt"}


def test_only_the_last_turns_are_kept():
    trimmed = _trim(conversation_of(10), 2)
    assert [m["content"] for m in trimmed if m["role"] == "user"] == ["q8", "q9"]


def test_a_tool_result_is_never_left_without_its_call():
    """Most OpenAI-compatible providers reject an orphaned `tool` message outright."""
    messages = [
        {"role": "system", "content": "prompt"},
        {"role": "user", "content": "q0"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1"}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "{}"},
        {"role": "assistant", "content": "a0"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
    ]
    trimmed = _trim(messages, 1)

    roles = [message["role"] for message in trimmed]
    assert roles == ["system", "user", "assistant"]
    for index, message in enumerate(trimmed):
        if message["role"] == "tool":
            previous = trimmed[index - 1]
            assert previous.get("tool_calls"), "a tool result outlived its call"


def test_a_conversation_that_is_all_one_turn():
    messages = [{"role": "system", "content": "prompt"}, {"role": "user", "content": "q"}]
    assert _trim(messages, 1) == messages
