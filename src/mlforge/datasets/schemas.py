"""Convert Saara rows into the training-data schemas used across the industry.

Every converter takes one Saara row and returns the target record, or raises SchemaError with a
short reason (the export engine counts reasons instead of failing the whole export).
"""
from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any


class SchemaError(ValueError):
    """A row cannot be represented in the requested schema."""


# ---------- canonical form: chat messages ----------

def to_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize any Saara row shape into chat messages (system/user/assistant/tool)."""
    messages = row.get("messages")
    if isinstance(messages, list) and messages:
        result = [dict(m) for m in messages if isinstance(m, dict) and m.get("role")]
    else:
        source = row.get("input") if isinstance(row.get("input"), dict) else row
        output = row.get("output") if isinstance(row.get("output"), dict) else row
        instruction = str(source.get("instruction") or source.get("prompt") or source.get("question") or "")
        context = source.get("input") if isinstance(source.get("input"), str) else source.get("context")
        answer = output.get("output") if isinstance(output.get("output"), str) else output.get("response") or output.get("answer")
        if not instruction or not answer:
            raise SchemaError("no_messages_or_instruction_output")
        user = instruction + (f"\n\n{context}" if context else "")
        result = [{"role": "user", "content": user}, {"role": "assistant", "content": str(answer)}]
        system = source.get("system")
        if system:
            result.insert(0, {"role": "system", "content": str(system)})
    calls = row.get("tool_calls")
    if calls and not any(m.get("tool_calls") for m in result):
        for message in reversed(result):
            if message["role"] == "assistant":
                message["tool_calls"] = calls
                break
    if not any(m["role"] == "assistant" for m in result):
        raise SchemaError("no_assistant_turn")
    return result


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # multimodal parts: keep text, mark images
        return "".join(p.get("text", "") if p.get("type") == "text" else "<image>" for p in content if isinstance(p, dict))
    return "" if content is None else str(content)


def _images(content: Any) -> list[str]:
    if not isinstance(content, list):
        return []
    return [str(p.get("path") or p.get("url") or p.get("image")) for p in content if isinstance(p, dict) and p.get("type") == "image"]


def _call_parts(call: dict[str, Any]) -> tuple[str, Any]:
    function = call.get("function", call)
    args = function.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError as exc:
            raise SchemaError("tool_arguments_not_json") from exc
    return str(function["name"]), args


def _tools(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [t if "function" in t else {"type": "function", "function": t} for t in row.get("tools") or []]


# ---------- target schemas ----------

def openai_chat(row: dict[str, Any]) -> dict[str, Any]:
    """OpenAI fine-tuning chat format (also accepted by Together, Fireworks, Azure, Unsloth, TRL)."""
    out = []
    for message in to_messages(row):
        item: dict[str, Any] = {"role": message["role"], "content": _text(message.get("content"))}
        if message.get("tool_calls"):
            item["tool_calls"] = []
            for index, call in enumerate(message["tool_calls"]):
                name, args = _call_parts(call)
                item["tool_calls"].append({"id": call.get("id", f"call_{index}"), "type": "function",
                                           "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}})
        if message["role"] == "tool":
            item["tool_call_id"] = message.get("tool_call_id", "")
        out.append(item)
    record: dict[str, Any] = {"messages": out}
    if row.get("tools"):
        record["tools"] = _tools(row)
    return record


SHAREGPT_ROLES = {"system": "system", "user": "human", "assistant": "gpt", "tool": "observation"}


def sharegpt(row: dict[str, Any]) -> dict[str, Any]:
    """ShareGPT as used by LLaMA-Factory/Axolotl (function_call/observation turns for tools)."""
    conversations, system, images = [], None, []
    for message in to_messages(row):
        images += _images(message.get("content"))
        if message["role"] == "system":
            system = _text(message.get("content"))
            continue
        text = _text(message.get("content"))
        if message["role"] == "assistant" and message.get("tool_calls"):
            if text:
                conversations.append({"from": "gpt", "value": text})
            calls = [dict(zip(("name", "arguments"), _call_parts(c))) for c in message["tool_calls"]]
            conversations.append({"from": "function_call", "value": json.dumps(calls[0] if len(calls) == 1 else calls, ensure_ascii=False)})
            continue
        conversations.append({"from": SHAREGPT_ROLES[message["role"]], "value": text})
    record: dict[str, Any] = {"conversations": conversations}
    if system:
        record["system"] = system
    if row.get("tools"):
        record["tools"] = json.dumps([t["function"] for t in _tools(row)], ensure_ascii=False)
    if images:
        record["images"] = images
    return record


def alpaca(row: dict[str, Any]) -> dict[str, Any]:
    """Alpaca instruction format; earlier turns go to `history` as [user, assistant] pairs."""
    messages = to_messages(row)
    if any(m.get("tool_calls") or m["role"] == "tool" for m in messages):
        raise SchemaError("alpaca_cannot_hold_tool_calls")
    system = next((_text(m["content"]) for m in messages if m["role"] == "system"), None)
    turns = [m for m in messages if m["role"] in ("user", "assistant")]
    if len(turns) < 2 or turns[-1]["role"] != "assistant" or turns[-2]["role"] != "user":
        raise SchemaError("alpaca_needs_user_then_assistant")
    history = [[_text(turns[i]["content"]), _text(turns[i + 1]["content"])] for i in range(0, len(turns) - 2, 2)]
    record: dict[str, Any] = {"instruction": _text(turns[-2]["content"]), "input": "", "output": _text(turns[-1]["content"])}
    if system:
        record["system"] = system
    if history:
        record["history"] = history
    return record


def chatml(row: dict[str, Any]) -> dict[str, Any]:
    """Single `text` field rendered with ChatML markers (for raw-text SFT trainers)."""
    parts = []
    for message in to_messages(row):
        content = _text(message.get("content"))
        if message.get("tool_calls"):
            calls = [dict(zip(("name", "arguments"), _call_parts(c))) for c in message["tool_calls"]]
            content += "".join(f"\n<tool_call>{json.dumps(c, ensure_ascii=False)}</tool_call>" for c in calls)
        parts.append(f"<|im_start|>{message['role']}\n{content}<|im_end|>")
    return {"text": "\n".join(parts)}


def vertex_gemini(row: dict[str, Any]) -> dict[str, Any]:
    """Vertex AI / Gemini supervised tuning format (contents with user/model roles and parts)."""
    contents, system = [], None
    names: dict[str, str] = {}
    for message in to_messages(row):
        role = message["role"]
        if role == "system":
            system = {"role": "system", "parts": [{"text": _text(message.get("content"))}]}
            continue
        if role == "tool":
            response = message.get("content")
            try:
                response = json.loads(response) if isinstance(response, str) else response
            except json.JSONDecodeError:
                pass
            name = names.get(message.get("tool_call_id", ""), message.get("name", "tool"))
            contents.append({"role": "user", "parts": [{"functionResponse": {
                "name": name, "response": response if isinstance(response, dict) else {"content": response}}}]})
            continue
        parts = [{"text": _text(message.get("content"))}] if _text(message.get("content")) else []
        for call in message.get("tool_calls") or []:
            name, args = _call_parts(call)
            names[call.get("id", "")] = name
            parts.append({"functionCall": {"name": name, "args": args}})
        if _images(message.get("content")):
            raise SchemaError("vertex_needs_uploaded_image_uris")
        contents.append({"role": "model" if role == "assistant" else "user", "parts": parts})
    record: dict[str, Any] = {"contents": contents}
    if system:
        record["systemInstruction"] = system
    return record


def dpo(row: dict[str, Any]) -> dict[str, Any]:
    """TRL preference format (prompt/chosen/rejected), from Saara `distill --method dpo` rows."""
    if not (row.get("chosen") and row.get("rejected")):
        raise SchemaError("dpo_needs_chosen_and_rejected")
    if row["chosen"] == row["rejected"]:
        raise SchemaError("dpo_chosen_equals_rejected")
    return {"prompt": _text(row.get("prompt")), "chosen": _text(row["chosen"]), "rejected": _text(row["rejected"])}


def text(row: dict[str, Any]) -> dict[str, Any]:
    """Plain-text pretraining format."""
    if isinstance(row.get("text"), str) and row["text"]:
        return {"text": row["text"]}
    body = "\n\n".join(_text(m.get("content")) for m in to_messages(row) if m["role"] != "system")
    return {"text": body}


def saara(row: dict[str, Any]) -> dict[str, Any]:
    return dict(row)


SCHEMAS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "saara": saara, "openai-chat": openai_chat, "sharegpt": sharegpt, "alpaca": alpaca, "chatml": chatml,
    "vertex-gemini": vertex_gemini, "dpo": dpo, "text": text,
}
