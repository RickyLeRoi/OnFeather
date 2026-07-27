"""Speaking OpenAI to clients that mean it literally.

An agentic client sends two things a plain chat proxy has no reason to think
about: tool definitions, and a schema the answer must obey. Both have to survive
the trip to a provider whose OpenAI compatibility is approximate, and come back
in exactly the shape the client's validator expects -- not merely close to it.

The asymmetry worth knowing about is that these two fail differently. A dropped
tool array produces prose where a tool call was needed, and an agent that loops
until it gives up. A dropped `response_format` produces well-formed JSON of the
wrong shape, which the caller rejects outright. Neither shows up as an HTTP
error, so both are handled here rather than left to the provider's goodwill.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

#: 20260726 ** RG Provider takes JSON Schema as written, bar the meta keywords.
DIALECT_OPENAI = "openai"
#: 20260726 ** RG OpenAI's own strict mode accepts a documented subset and 400s on the rest.
DIALECT_OPENAI_STRICT = "openai_strict"

DIALECTS = (DIALECT_OPENAI, DIALECT_OPENAI_STRICT)

#: 20260726 ** RG Identity keywords, unlike $defs, which $ref still needs.
_META_KEYWORDS = frozenset({"$schema", "$id"})

#: 20260726 ** RG Keywords OpenAI's structured outputs reject under strict: true.
_STRICT_UNSUPPORTED = frozenset({
    "minLength", "maxLength", "pattern", "format",
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
    "minItems", "maxItems", "uniqueItems", "contains", "minContains", "maxContains",
    "minProperties", "maxProperties", "patternProperties", "propertyNames",
    "unevaluatedItems", "unevaluatedProperties",
})

_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "boolean": (bool,),
    "object": (dict,),
    "array": (list,),
}


# -- tools ----------------------------------------------------------------


def adapt_tools(tools: Any, dialect: str = DIALECT_OPENAI) -> list[dict] | None:
    """Rewrite a `tools` array for one provider's idea of JSON Schema."""
    if not isinstance(tools, list) or not tools:
        return None

    adapted = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        entry = dict(tool)
        entry.setdefault("type", "function")
        function = entry.get("function")
        if isinstance(function, dict):
            function = dict(function)
            if "parameters" in function:
                function["parameters"] = adapt_schema(function["parameters"], dialect)
            entry["function"] = function
        adapted.append(entry)
    return adapted or None


def adapt_schema(schema: Any, dialect: str = DIALECT_OPENAI) -> Any:
    """Strip what the provider cannot read, keeping the meaning intact.

    Almost nothing needs stripping. Providers that route to an OpenAI-shaped
    constrained decoder take draft-07 as written, and rewriting a schema that
    would have been accepted only loses information -- `nullable: true` in place
    of `anyOf: [T, null]` says the same thing, but dropping `maximum` to get
    there does not.

    The exception is OpenAI's own strict mode, which documents a subset and
    answers 400 rather than ignoring the rest. What is dropped for it is still
    enforced on the way back by `validate`, so the constraint holds even where
    the provider never saw it.
    """
    if isinstance(schema, list):
        return [adapt_schema(item, dialect) for item in schema]
    if not isinstance(schema, dict):
        return schema

    unsupported = _STRICT_UNSUPPORTED if dialect == DIALECT_OPENAI_STRICT else frozenset()
    return {
        key: adapt_schema(value, dialect)
        for key, value in schema.items()
        if key not in _META_KEYWORDS and key not in unsupported
    }


# -- response_format ------------------------------------------------------


def schema_of(response_format: Any) -> dict | None:
    """The schema a `response_format` demands, if it demands one."""
    if not isinstance(response_format, dict):
        return None
    if response_format.get("type") != "json_schema":
        return None
    block = response_format.get("json_schema")
    if not isinstance(block, dict):
        return None
    schema = block.get("schema")
    return schema if isinstance(schema, dict) else None


def adapt_response_format(response_format: dict, dialect: str = DIALECT_OPENAI) -> dict:
    """Rewrite a `response_format` block for one provider."""
    out = dict(response_format)
    block = out.get("json_schema")
    if isinstance(block, dict) and isinstance(block.get("schema"), dict):
        block = dict(block)
        block["schema"] = adapt_schema(block["schema"], dialect)
        block.setdefault("name", "response")
        out["json_schema"] = block
    return out


def schema_instruction(schema: dict) -> str:
    """A prompt that stands in for a `response_format` the provider lacks.

    Only reached when the model cannot be constrained natively. The schema is
    never in the caller's own prompt -- that is the whole point of sending it as
    `response_format` -- so without this the answer is not degraded, it is the
    wrong object entirely.
    """
    return (
        "Reply with a single JSON object and nothing else: no prose, no code fence, "
        "no explanation. It must validate against this JSON Schema exactly, with every "
        "required key present and no key that the schema does not define.\n\n"
        + json.dumps(schema, indent=2, ensure_ascii=False)
    )


def with_instruction(messages: list[dict], instruction: str) -> list[dict]:
    """Fold an instruction into the system prompt, without touching the caller's list."""
    out = [dict(message) if isinstance(message, dict) else message for message in messages]
    for message in out:
        if isinstance(message, dict) and message.get("role") == "system":
            content = message.get("content")
            if isinstance(content, str):
                message["content"] = f"{content}\n\n{instruction}"
                return out
            break
    return [{"role": "system", "content": instruction}, *out]


# -- reading the answer ---------------------------------------------------


def extract_json(text: str) -> Any:
    """Parse JSON out of a reply that may be wrapped in prose or a code fence.

    Raises ValueError when there is nothing parsable, which the caller treats as
    a failed attempt and fails over rather than returning junk.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("empty response")

    candidate = text.strip()
    try:
        return json.loads(candidate)
    except ValueError:
        pass

    if candidate.startswith("```"):
        candidate = candidate.split("```")[1] if candidate.count("```") >= 2 else candidate[3:]
        if candidate.lstrip().lower().startswith("json"):
            candidate = candidate.lstrip()[4:]
        try:
            return json.loads(candidate.strip())
        except ValueError:
            pass

    start, end = candidate.find("{"), candidate.rfind("}")
    if start != -1 and end > start:
        return json.loads(candidate[start : end + 1])

    raise ValueError("no JSON object in response")


def prune(value: Any, schema: Any) -> Any:
    """Drop keys the schema forbids, recursively.

    `strict: true` means the caller will reject an object carrying anything it
    did not ask for. Weaker models add a `notes` field roughly as often as they
    get the rest right, and dropping it is a far better outcome than failing the
    whole call over it.
    """
    if not isinstance(schema, dict):
        return value

    branches = schema.get("anyOf") or schema.get("oneOf")
    if isinstance(branches, list):
        for branch in branches:
            if isinstance(branch, dict) and not validate(value, branch):
                return prune(value, branch)
        return value

    if isinstance(value, dict) and schema.get("type") == "object":
        properties = schema.get("properties")
        if not isinstance(properties, dict):
            return value
        closed = schema.get("additionalProperties") is False
        out = {}
        for key, item in value.items():
            if key in properties:
                out[key] = prune(item, properties[key])
            elif not closed:
                out[key] = item
        return out

    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        return [prune(item, schema["items"]) for item in value]

    return value


def validate(value: Any, schema: Any, path: str = "$") -> list[str]:
    """Check a value against the draft-07 subset that actually gets sent.

    Not a conformance implementation: enough of the language to catch a model
    that returned the right JSON with the wrong contents, which is the failure
    this exists to turn into a retry.
    """
    if not isinstance(schema, dict) or schema == {}:
        return []

    branches = schema.get("anyOf") or schema.get("oneOf")
    if isinstance(branches, list) and branches:
        if any(not validate(value, branch, path) for branch in branches):
            return []
        return [f"{path} matches none of the permitted types"]

    errors: list[str] = []
    declared = schema.get("type")
    permitted = [declared] if isinstance(declared, str) else list(declared or [])
    if permitted and not any(_is_type(value, expected) for expected in permitted):
        return [f"{path} should be {declared}, got {_name_of(value)}"]

    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path} is not one of {schema['enum']}")

    if isinstance(value, dict):
        errors += _validate_object(value, schema, path)
    elif isinstance(value, list):
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                errors += validate(item, item_schema, f"{path}[{index}]")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path} is below the minimum of {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path} is above the maximum of {schema['maximum']}")
    elif isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append(f"{path} is shorter than {schema['minLength']}")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(f"{path} is longer than {schema['maxLength']}")

    return errors


def _validate_object(value: dict, schema: dict, path: str) -> list[str]:
    errors: list[str] = []
    properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}

    for name in schema.get("required") or []:
        if name not in value:
            errors.append(f"{path}.{name} is required and missing")

    if schema.get("additionalProperties") is False:
        for name in value:
            if name not in properties:
                errors.append(f"{path}.{name} is not allowed")

    for name, item in value.items():
        sub = properties.get(name)
        if isinstance(sub, dict):
            errors += validate(item, sub, f"{path}.{name}")
    return errors


def _is_type(value: Any, expected: str) -> bool:
    if expected == "null":
        return value is None
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    types = _JSON_TYPES.get(expected)
    if types is None:
        return True  # 20260726 ** RG Unknown type keyword: not our business to reject it.
    if expected == "string" and isinstance(value, bool):
        return False
    return isinstance(value, types)


def _name_of(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


# -- the assistant message ------------------------------------------------


def normalise_message(message: Any, *, seed: str = "") -> dict:
    """Put an upstream assistant message into the shape clients validate against.

    Two things vary between providers and both break a tool loop silently:
    `arguments` comes back as a parsed object from some of them where the spec
    says string, and `id` is occasionally absent -- and that id is what the next
    request echoes as `tool_call_id`, so without it the conversation cannot
    continue.
    """
    out = dict(message) if isinstance(message, dict) else {}
    out["role"] = out.get("role") or "assistant"
    if "content" not in out:
        out["content"] = None

    calls = out.get("tool_calls")
    if isinstance(calls, list) and calls:
        out["tool_calls"] = [_normalise_call(call, index, seed) for index, call in enumerate(calls)]
    elif "tool_calls" in out:
        # 20260726 ** RG An empty array reads as "a tool step with no calls" and stalls the loop.
        out.pop("tool_calls")
    return out


def _normalise_call(call: Any, index: int, seed: str) -> dict:
    out = dict(call) if isinstance(call, dict) else {}
    out["type"] = out.get("type") or "function"

    function = dict(out.get("function") or {})
    arguments = function.get("arguments")
    if arguments is None:
        function["arguments"] = "{}"
    elif not isinstance(arguments, str):
        # 20260726 ** RG Ollama and a few others hand back a parsed object here.
        function["arguments"] = json.dumps(arguments, ensure_ascii=False)
    out["function"] = function

    if not out.get("id"):
        out["id"] = _synthetic_id(seed, index, function)
    return out


def _synthetic_id(seed: str, index: int, function: dict) -> str:
    material = f"{seed}|{index}|{function.get('name')}|{function.get('arguments')}"
    return f"call_{hashlib.sha1(material.encode()).hexdigest()[:24]}"


def message_problem(message: Any) -> str | None:
    """Why this message is unusable to a tool-calling client, if it is.

    A tool call without a name is worse than no tool call: the client's schema
    rejects the whole response, and a provider that does this does it every
    time, so the useful answer is to fail this attempt and try the next one.
    """
    if not isinstance(message, dict):
        return "message is not an object"
    for call in message.get("tool_calls") or []:
        function = call.get("function") if isinstance(call, dict) else None
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            return "tool call is missing a function name"
        if not function["name"].strip():
            return "tool call has an empty function name"
    return None


def finish_reason(message: dict, upstream: Any) -> str:
    """Report `tool_calls` when there are some, whatever the provider said."""
    if message.get("tool_calls"):
        return "tool_calls"
    return upstream if isinstance(upstream, str) and upstream else "stop"
