"""Reusable Gemini function-calling loop for agents that bypass ADK.

Provides a ReAct-style loop using google.genai directly:
  1. Call generate_content with tool declarations.
  2. If the model returns FunctionCall parts, execute the tools and feed
     FunctionResponse parts back.
  3. Repeat until the model returns a plain text response (JSON).

Also exposes record_llm_usage() for token cost tracking.
"""

from __future__ import annotations

import inspect
import logging
from datetime import datetime
from typing import Any, Callable

import google.genai as genai
from google.genai import types as genai_types

from app.agents.llm_factory import default_model_name, get_gemini_client
from app.agents.llm_json import parse_llm_json

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 5


# ── Token usage recording ────────────────────────────────────────────────────

def record_llm_usage(
    response: genai_types.GenerateContentResponse,
    *,
    model_name: str,
    started_at: datetime,
    ended_at: datetime,
) -> None:
    """Persist token counts and cost from a Gemini response to the cost ledger."""
    try:
        from app.observability.cost import record_gemini_call
        record_gemini_call(
            response=response,
            model_name=model_name,
            started_at=started_at,
            ended_at=ended_at,
        )
    except Exception:
        logger.debug("Cost recording skipped (no active run or DB unavailable)")


# ── Tool declaration builder ─────────────────────────────────────────────────

def _build_tool_declaration(fn: Callable) -> genai_types.FunctionDeclaration:
    """Convert a plain Python function into a Gemini FunctionDeclaration."""
    sig = inspect.signature(fn)
    doc = inspect.getdoc(fn) or fn.__name__

    # Parse Args section from docstring for descriptions
    arg_docs: dict[str, str] = {}
    in_args = False
    for line in doc.splitlines():
        stripped = line.strip()
        if stripped == "Args:":
            in_args = True
            continue
        if in_args:
            if stripped and not stripped.startswith(" ") and stripped.endswith(":"):
                break
            if ":" in stripped:
                arg_name, _, arg_desc = stripped.partition(":")
                arg_docs[arg_name.strip()] = arg_desc.strip()

    # Strip Args section from main description
    description = doc.split("\n\nArgs:")[0].strip()

    properties: dict[str, genai_types.Schema] = {}
    required: list[str] = []

    _type_map = {
        str: genai_types.Type.STRING,
        int: genai_types.Type.INTEGER,
        float: genai_types.Type.NUMBER,
        bool: genai_types.Type.BOOLEAN,
    }

    for param_name, param in sig.parameters.items():
        annotation = param.annotation
        # Unwrap Optional[X] → X
        origin = getattr(annotation, "__origin__", None)
        args = getattr(annotation, "__args__", ())
        if origin is type(None):
            continue
        if origin is not None and len(args) == 2 and type(None) in args:
            annotation = next(a for a in args if a is not type(None))

        schema_type = _type_map.get(annotation, genai_types.Type.STRING)
        properties[param_name] = genai_types.Schema(
            type=schema_type,
            description=arg_docs.get(param_name, ""),
        )
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    return genai_types.FunctionDeclaration(
        name=fn.__name__,
        description=description,
        parameters=genai_types.Schema(
            type=genai_types.Type.OBJECT,
            properties=properties,
            required=required if required else None,
        ),
    )


def _build_tools_config(fns: list[Callable]) -> list[genai_types.Tool]:
    declarations = [_build_tool_declaration(fn) for fn in fns]
    return [genai_types.Tool(function_declarations=declarations)]


def _evidence_flags(tools_called: set[str]) -> dict[str, bool]:
    return {name: True for name in tools_called}


# ── Main loop ────────────────────────────────────────────────────────────────

def run_agent_with_tools(
    system_prompt: str,
    user_message: str,
    tools: list[Callable],
    max_rounds: int = MAX_TOOL_ROUNDS,
    return_evidence: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], dict[str, bool]]:
    """Run a Gemini agent with tools in a ReAct-style loop.

    Parameters
    ----------
    system_prompt : str
        System instruction for the agent.
    user_message : str
        User message describing the task.
    tools : list[Callable]
        Plain Python functions the agent can call.
    max_rounds : int
        Maximum function-call rounds before forcing a text response.
    return_evidence : bool
        When True, return (parsed_json, evidence_flags).
    """
    client = get_gemini_client()
    model = default_model_name()
    tool_map = {fn.__name__: fn for fn in tools}
    tools_config = _build_tools_config(tools) if tools else []
    tools_called: set[str] = set()

    contents: list[genai_types.Content] = [
        genai_types.Content(role="user", parts=[genai_types.Part(text=user_message)])
    ]

    config = genai_types.GenerateContentConfig(
        system_instruction=system_prompt,
        tools=tools_config if tools_config else None,
    )

    for round_num in range(max_rounds):
        started_at = datetime.utcnow()
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )
        ended_at = datetime.utcnow()
        record_llm_usage(response, model_name=model, started_at=started_at, ended_at=ended_at)

        candidate = response.candidates[0] if response.candidates else None
        if candidate is None:
            break

        # Separate function calls from text parts
        fn_calls = [p for p in candidate.content.parts if p.function_call]
        text_parts = [p for p in candidate.content.parts if p.text]

        if not fn_calls:
            # No tool calls — final text response
            text = "".join(p.text for p in text_parts)
            parsed = parse_llm_json(text)
            if return_evidence:
                return parsed, _evidence_flags(tools_called)
            return parsed

        # Append model turn to history
        contents.append(candidate.content)

        # Execute each function call and build response parts
        fn_response_parts: list[genai_types.Part] = []
        for fn_call in fn_calls:
            name = fn_call.function_call.name
            args = dict(fn_call.function_call.args or {})
            tools_called.add(name)
            logger.debug("Agent calling tool: %s(%s)", name, args)

            if name not in tool_map:
                result = f"Error: Unknown tool '{name}'"
            else:
                try:
                    result = tool_map[name](**args)
                except Exception as exc:
                    result = f"Error calling {name}: {exc}"
                    logger.warning("Tool %s failed: %s", name, exc)

            fn_response_parts.append(
                genai_types.Part(
                    function_response=genai_types.FunctionResponse(
                        name=name,
                        response={"result": str(result)},
                    )
                )
            )

        contents.append(
            genai_types.Content(role="user", parts=fn_response_parts)
        )

    # Exhausted rounds — force final text response without tools
    logger.warning("Agent exhausted %d tool rounds, forcing final response", max_rounds)
    config_no_tools = genai_types.GenerateContentConfig(
        system_instruction=system_prompt,
    )
    started_at = datetime.utcnow()
    final = client.models.generate_content(
        model=model,
        contents=contents,
        config=config_no_tools,
    )
    ended_at = datetime.utcnow()
    record_llm_usage(final, model_name=model, started_at=started_at, ended_at=ended_at)

    text = final.text or ""
    try:
        parsed = parse_llm_json(text)
    except Exception as first_error:
        logger.warning("Final response was not valid JSON; requesting JSON repair")
        repair_msg = (
            "Return ONLY a valid JSON object for the previously requested schema. "
            "Do not call tools, do not include markdown, prose, or tags."
        )
        contents.append(genai_types.Content(role="model", parts=[genai_types.Part(text=text)]))
        contents.append(genai_types.Content(role="user", parts=[genai_types.Part(text=repair_msg)]))
        started_at = datetime.utcnow()
        repaired = client.models.generate_content(
            model=model, contents=contents, config=config_no_tools
        )
        ended_at = datetime.utcnow()
        record_llm_usage(repaired, model_name=model, started_at=started_at, ended_at=ended_at)
        try:
            parsed = parse_llm_json(repaired.text or "")
        except Exception:
            raise first_error

    if return_evidence:
        return parsed, _evidence_flags(tools_called)
    return parsed
