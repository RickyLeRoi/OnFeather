"""The conversation platform for the OnFeather integration.

Assist talks to the router, and the router picks whichever free tier still has
quota. Two consequences shape this file.

The router does not stream, so a whole answer arrives at once — which the chat
log accepts perfectly well, as a delta stream of exactly one delta. And because
the router may change its mind between requests, every turn of a run carries the
same session header, so the model that started a conversation is the one that
finishes it.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator, Callable
from typing import Any, Literal

from voluptuous_openapi import convert

from homeassistant.components import conversation
from homeassistant.const import CONF_LLM_HASS_API, CONF_MODEL, CONF_PROMPT, MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.json import json_dumps

from . import OnFeatherConfigEntry
from .api import OnFeatherError
from .const import (
    CONF_MAX_HISTORY,
    DEFAULT_MAX_HISTORY,
    DEFAULT_MODEL,
    DOMAIN,
    MAX_TOOL_ITERATIONS,
)
from .entity import free_device

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: OnFeatherConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the conversation entity."""
    async_add_entities([OnFeatherConversationEntity(entry)])


class OnFeatherConversationEntity(
    conversation.ConversationEntity, conversation.AbstractConversationAgent
):
    """A conversation agent backed by the OnFeather router."""

    _attr_has_entity_name = True
    _attr_name = None
    # 20260804 ++ RG #HASS The router answers in one piece; nothing to stream.
    _attr_supports_streaming = False

    def __init__(self, entry: OnFeatherConfigEntry) -> None:
        self.entry = entry
        self._attr_unique_id = f"{entry.entry_id}_conversation"
        self._attr_device_info = free_device(entry)
        if entry.options.get(CONF_LLM_HASS_API):
            self._attr_supported_features = conversation.ConversationEntityFeature.CONTROL

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        conversation.async_set_agent(self.hass, self.entry, self)

    async def async_will_remove_from_hass(self) -> None:
        conversation.async_unset_agent(self.hass, self.entry)
        await super().async_will_remove_from_hass()

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Whatever the model underneath happens to speak, which we cannot know."""
        return MATCH_ALL

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Answer one turn."""
        options = self.entry.options

        try:
            await chat_log.async_provide_llm_data(
                user_input.as_llm_context(DOMAIN),
                options.get(CONF_LLM_HASS_API),
                options.get(CONF_PROMPT),
                user_input.extra_system_prompt,
            )
        except conversation.ConverseError as err:
            return err.as_conversation_result()

        await self._async_converse(chat_log)
        return conversation.async_get_result_from_chat_log(user_input, chat_log)

    async def _async_converse(self, chat_log: conversation.ChatLog) -> None:
        """Run the request, and any tool calls it asks for, to a final answer."""
        client = self.entry.runtime_data.client
        options = self.entry.options
        max_history = int(options.get(CONF_MAX_HISTORY, DEFAULT_MAX_HISTORY))

        tools: list[dict[str, Any]] | None = None
        if chat_log.llm_api:
            tools = [
                _format_tool(tool, chat_log.llm_api.custom_serializer)
                for tool in chat_log.llm_api.tools
            ]

        for _iteration in range(MAX_TOOL_ITERATIONS):
            payload: dict[str, Any] = {
                "model": options.get(CONF_MODEL, DEFAULT_MODEL),
                "messages": _trim(
                    [_to_message(content) for content in chat_log.content], max_history
                ),
                "stream": False,
            }
            if tools:
                payload["tools"] = tools

            try:
                response = await client.chat(payload, session_id=chat_log.conversation_id)
            except OnFeatherError as err:
                raise HomeAssistantError(f"OnFeather could not answer: {err}") from err

            message = _assistant_message(response)
            async for _content in chat_log.async_add_delta_content_stream(
                self.entity_id, _as_delta_stream(message)
            ):
                pass

            if not chat_log.unresponded_tool_results:
                break


def _assistant_message(response: dict[str, Any]) -> dict[str, Any]:
    """The one message the router returned, or a clear account of why there is none."""
    choices = response.get("choices") or []
    message = choices[0].get("message") if choices else None
    if not isinstance(message, dict):
        raise HomeAssistantError("OnFeather returned no message")

    if not message.get("content") and not message.get("tool_calls"):
        # 20260804 ++ RG #HASS Naming the model beats the chat log's generic failure.
        served = response.get("model") or "the model"
        reason = (choices[0].get("finish_reason") or "no reason given") if choices else ""
        raise HomeAssistantError(f"{served} returned an empty answer ({reason})")
    return message


async def _as_delta_stream(
    message: dict[str, Any],
) -> AsyncGenerator[conversation.AssistantContentDeltaDict]:
    """A complete answer, shaped as the single delta the chat log expects.

    The chat log flushes whatever it is holding when the stream ends, so one
    delta carrying the whole message is a valid stream rather than a trick.
    """
    delta: conversation.AssistantContentDeltaDict = {"role": "assistant"}
    if content := message.get("content"):
        delta["content"] = content
    if thinking := message.get("reasoning_content"):
        delta["thinking_content"] = thinking
    if calls := message.get("tool_calls"):
        delta["tool_calls"] = [_to_tool_input(call) for call in calls]
    yield delta


def _to_tool_input(call: dict[str, Any]) -> llm.ToolInput:
    """Turn an OpenAI tool call into the chat log's own shape."""
    function = call.get("function") or {}
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError as err:
            raise HomeAssistantError(
                f"tool call arguments were not valid JSON: {arguments!r}"
            ) from err
    if not isinstance(arguments, dict):
        arguments = {}

    fields: dict[str, Any] = {
        "tool_name": function.get("name") or "",
        "tool_args": arguments,
    }
    # 20260804 ++ RG #HASS Keep the upstream id: it pairs the result back to the call.
    if call_id := call.get("id"):
        fields["id"] = call_id
    return llm.ToolInput(**fields)


def _to_message(content: conversation.Content) -> dict[str, Any]:
    """Turn one chat log entry into an OpenAI message."""
    if isinstance(content, conversation.ToolResultContent):
        return {
            "role": "tool",
            "tool_call_id": content.tool_call_id,
            "content": json_dumps(content.tool_result),
        }

    if isinstance(content, conversation.AssistantContent):
        message: dict[str, Any] = {"role": "assistant", "content": content.content}
        if content.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.tool_name,
                        "arguments": json_dumps(call.tool_args),
                    },
                }
                for call in content.tool_calls
            ]
        return message

    if isinstance(content, conversation.UserContent):
        if content.attachments:
            raise HomeAssistantError(
                "OnFeather cannot forward attachments: the router does not model "
                "which free tiers accept images"
            )
        return {"role": "user", "content": content.content}

    if isinstance(content, conversation.SystemContent):
        return {"role": "system", "content": content.content}

    raise TypeError(f"unexpected content type: {type(content)}")


def _trim(messages: list[dict[str, Any]], max_history: int) -> list[dict[str, Any]]:
    """Drop the oldest turns, never orphaning a tool result.

    Free-tier context windows are small, and the router refuses a request no
    model can hold rather than truncating it, so the trimming has to happen
    here. The cut always lands on a user message: a `tool` message whose
    assistant tool_calls parent has been dropped is rejected outright by most
    OpenAI-compatible providers.
    """
    if max_history < 1:
        return messages

    system = messages[:1] if messages and messages[0]["role"] == "system" else []
    rest = messages[len(system) :]

    starts = [index for index, message in enumerate(rest) if message["role"] == "user"]
    if len(starts) <= max_history:
        return messages
    return [*system, *rest[starts[-max_history] :]]


def _format_tool(
    tool: llm.Tool, custom_serializer: Callable[[Any], Any] | None
) -> dict[str, Any]:
    """Describe one Home Assistant tool the way an OpenAI endpoint expects."""
    specification: dict[str, Any] = {
        "name": tool.name,
        "parameters": convert(tool.parameters, custom_serializer=custom_serializer),
    }
    if tool.description:
        specification["description"] = tool.description
    return {"type": "function", "function": specification}
