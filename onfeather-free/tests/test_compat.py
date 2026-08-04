"""Tool and schema translation, which is where an OpenAI client gets literal."""

from __future__ import annotations

import json

import pytest

from onfeather_free import compat

# 20260725 RG What AutoHeal sends: draft-07, strict, nullable as a union.
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


# -- schemas --------------------------------------------------------------


def test_meta_keywords_are_stripped_for_everyone():
    """`$schema` is the one keyword that is never information to a provider."""
    adapted = compat.adapt_schema(HEAL_SCHEMA)
    assert "$schema" not in adapted
    assert adapted["required"] == ["ref", "confidence", "reasoning"]
    assert adapted["additionalProperties"] is False


def test_the_default_dialect_sends_the_schema_as_written():
    """Rewriting a schema a provider would have accepted only loses information."""
    adapted = compat.adapt_schema(HEAL_SCHEMA)

    assert adapted["properties"]["ref"]["anyOf"] == [{"type": "string"}, {"type": "null"}]
    assert adapted["properties"]["confidence"]["minimum"] == 0
    assert adapted["properties"]["confidence"]["maximum"] == 1


def test_defs_survive_because_ref_still_needs_them():
    """Stripping $defs while leaving a $ref pointing at it produces a broken schema."""
    schema = {"$schema": "http://json-schema.org/draft-07/schema#",
              "$defs": {"ref": {"type": "string"}},
              "type": "object", "properties": {"a": {"$ref": "#/$defs/ref"}}}
    adapted = compat.adapt_schema(schema)

    assert adapted["$defs"] == {"ref": {"type": "string"}}
    assert adapted["properties"]["a"] == {"$ref": "#/$defs/ref"}


def test_strict_dialect_drops_the_keywords_openai_rejects():
    """Under strict: true these are a 400, not a keyword quietly ignored."""
    adapted = compat.adapt_schema(HEAL_SCHEMA, compat.DIALECT_OPENAI_STRICT)

    assert "minimum" not in adapted["properties"]["confidence"]
    assert "maximum" not in adapted["properties"]["confidence"]
    # 20260725 RG Dropped constraints are still enforced on the way back.
    assert compat.validate({"confidence": 4}, HEAL_SCHEMA["properties"]["confidence"], "$.c")


def test_strict_dialect_keeps_what_openai_does_support():
    adapted = compat.adapt_schema(HEAL_SCHEMA, compat.DIALECT_OPENAI_STRICT)

    assert adapted["additionalProperties"] is False
    assert adapted["required"] == ["ref", "confidence", "reasoning"]
    assert adapted["properties"]["ref"]["anyOf"] == [{"type": "string"}, {"type": "null"}]


def test_tool_parameters_are_adapted_too():
    tools = [{
        "type": "function",
        "function": {
            "name": "click",
            "parameters": {"$schema": "http://json-schema.org/draft-07/schema#",
                           "type": "object", "properties": {}, "additionalProperties": False},
        },
    }]
    adapted = compat.adapt_tools(tools, compat.DIALECT_OPENAI_STRICT)

    assert "$schema" not in adapted[0]["function"]["parameters"]
    assert adapted[0]["function"]["parameters"]["additionalProperties"] is False


def test_adapting_does_not_mutate_the_callers_tools():
    tools = [{"function": {"name": "click", "parameters": dict(HEAL_SCHEMA)}}]
    compat.adapt_tools(tools, compat.DIALECT_OPENAI_STRICT)
    assert "$schema" in tools[0]["function"]["parameters"]


def test_empty_tools_is_treated_as_no_tools():
    assert compat.adapt_tools([]) is None
    assert compat.adapt_tools(None) is None


# -- response_format ------------------------------------------------------


def test_schema_is_recognised_only_in_the_json_schema_form():
    assert compat.schema_of(RESPONSE_FORMAT) == HEAL_SCHEMA
    assert compat.schema_of({"type": "json_object"}) is None
    assert compat.schema_of(None) is None


def test_adapted_response_format_keeps_its_envelope():
    adapted = compat.adapt_response_format(RESPONSE_FORMAT, compat.DIALECT_OPENAI_STRICT)

    assert adapted["type"] == "json_schema"
    assert adapted["json_schema"]["name"] == "response"
    assert adapted["json_schema"]["strict"] is True
    assert "$schema" not in adapted["json_schema"]["schema"]


def test_instruction_carries_the_whole_schema():
    """The caller does not repeat the schema in its prompt, so this is the only copy."""
    instruction = compat.schema_instruction(HEAL_SCHEMA)
    assert "confidence" in instruction and "reasoning" in instruction
    assert json.loads(instruction[instruction.index("{"):])["required"]


def test_instruction_is_folded_into_the_existing_system_message():
    messages = [{"role": "system", "content": "be terse"}, {"role": "user", "content": "hi"}]
    out = compat.with_instruction(messages, "OBEY")

    assert out[0]["content"].startswith("be terse")
    assert "OBEY" in out[0]["content"]
    assert len(out) == 2
    assert messages[0]["content"] == "be terse"


def test_instruction_gets_its_own_message_when_there_is_no_system_one():
    out = compat.with_instruction([{"role": "user", "content": "hi"}], "OBEY")
    assert out[0] == {"role": "system", "content": "OBEY"}
    assert out[1]["role"] == "user"


# -- reading the answer ---------------------------------------------------


def test_plain_json_parses():
    assert compat.extract_json('{"a": 1}') == {"a": 1}


def test_fenced_json_parses():
    assert compat.extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_json_buried_in_prose_parses():
    assert compat.extract_json('Sure! {"a": 1} hope that helps') == {"a": 1}


@pytest.mark.parametrize("text", ["", "   ", "no json here"])
def test_unparsable_answers_raise(text):
    with pytest.raises(ValueError):
        compat.extract_json(text)


def test_extra_keys_are_pruned_when_the_schema_closes_the_object():
    value = {"ref": "x", "confidence": 0.5, "reasoning": "y", "notes": "chatty"}
    assert compat.prune(value, HEAL_SCHEMA) == {"ref": "x", "confidence": 0.5, "reasoning": "y"}


def test_extra_keys_survive_an_open_object():
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    assert compat.prune({"a": "x", "b": 1}, schema) == {"a": "x", "b": 1}


def test_a_valid_answer_validates():
    assert compat.validate({"ref": None, "confidence": 1, "reasoning": "y"}, HEAL_SCHEMA) == []


def test_a_missing_required_key_is_an_error():
    errors = compat.validate({"ref": "x", "confidence": 0.5}, HEAL_SCHEMA)
    assert any("reasoning" in error for error in errors)


def test_a_wrong_type_is_an_error():
    errors = compat.validate({"ref": "x", "confidence": "high", "reasoning": "y"}, HEAL_SCHEMA)
    assert any("confidence" in error for error in errors)


def test_a_bound_is_enforced():
    errors = compat.validate({"ref": "x", "confidence": 4, "reasoning": "y"}, HEAL_SCHEMA)
    assert any("maximum" in error for error in errors)


def test_an_undeclared_key_is_an_error_before_pruning():
    value = {"ref": "x", "confidence": 0.5, "reasoning": "y", "notes": "n"}
    assert any("notes" in error for error in compat.validate(value, HEAL_SCHEMA))


def test_a_boolean_is_not_a_number():
    """`True` passes `isinstance(x, int)`, and a confidence of True is not 1."""
    assert compat.validate({"ref": "x", "confidence": True, "reasoning": "y"}, HEAL_SCHEMA)


# -- the assistant message ------------------------------------------------


def test_object_arguments_become_a_json_string():
    """Ollama returns the parsed object; every client's validator wants the string."""
    message = compat.normalise_message({
        "role": "assistant", "content": None,
        "tool_calls": [{"id": "call_1", "function": {"name": "click", "arguments": {"ref": "e7"}}}],
    })
    arguments = message["tool_calls"][0]["function"]["arguments"]

    assert isinstance(arguments, str)
    assert json.loads(arguments) == {"ref": "e7"}


def test_string_arguments_are_left_exactly_as_they_are():
    raw = '{"ref": "e7"}'
    message = compat.normalise_message({
        "tool_calls": [{"id": "c", "function": {"name": "click", "arguments": raw}}]
    })
    assert message["tool_calls"][0]["function"]["arguments"] == raw


def test_a_missing_id_is_synthesised_and_deterministic():
    """That id comes back as `tool_call_id`, so a call without one ends the run."""
    call = {"function": {"name": "click", "arguments": "{}"}}
    first = compat.normalise_message({"tool_calls": [dict(call)]}, seed="abc")
    again = compat.normalise_message({"tool_calls": [dict(call)]}, seed="abc")

    assert first["tool_calls"][0]["id"].startswith("call_")
    assert first["tool_calls"][0]["id"] == again["tool_calls"][0]["id"]


def test_parallel_calls_without_ids_do_not_collide():
    message = compat.normalise_message({"tool_calls": [
        {"function": {"name": "click", "arguments": "{}"}},
        {"function": {"name": "click", "arguments": "{}"}},
    ]})
    assert message["tool_calls"][0]["id"] != message["tool_calls"][1]["id"]


def test_null_content_survives():
    message = compat.normalise_message({"role": "assistant", "content": None})
    assert message["content"] is None


def test_absent_content_becomes_null_not_empty_string():
    assert compat.normalise_message({"role": "assistant"})["content"] is None


def test_an_empty_tool_call_array_is_dropped():
    """An empty array reads as a tool step with no calls, and stalls the loop."""
    assert "tool_calls" not in compat.normalise_message({"content": "hi", "tool_calls": []})


def test_a_nameless_tool_call_is_reported_as_unusable():
    message = compat.normalise_message({"tool_calls": [{"function": {"arguments": "{}"}}]})
    assert compat.message_problem(message)


def test_an_ordinary_message_has_no_problem():
    assert compat.message_problem({"role": "assistant", "content": "hi"}) is None


def test_finish_reason_reports_tool_calls_whatever_the_provider_said():
    message = {"tool_calls": [{"function": {"name": "click"}}]}
    assert compat.finish_reason(message, "stop") == "tool_calls"


def test_finish_reason_passes_through_otherwise():
    assert compat.finish_reason({"content": "hi"}, "length") == "length"
    assert compat.finish_reason({"content": "hi"}, None) == "stop"
