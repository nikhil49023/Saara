"""
Dataset Output Formats Module
Unified format converters for different training use cases.

Format Selection Guide:
-----------------------
| Use Case                      | Format     |
|-------------------------------|------------|
| Domain adaptation (PDFs)      | Alpaca     |
| Chatbot / assistant           | ChatML     |
| Multi-turn with history       | ShareGPT   |
| Base model continuation       | Completion |
| Quality alignment / RLHF      | DPO        |
| Function calling / tools      | ChatML     |
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class FormatType(Enum):
    """Supported dataset output formats."""
    ALPACA = "alpaca"
    CHATML = "chatml"
    SHAREGPT = "sharegpt"
    COMPLETION = "completion"
    DPO = "dpo"
    CHATML_TOOLS = "chatml_tools"


@dataclass
class FormatConfig:
    """Configuration for format conversion."""
    system_prompt: str = ""
    include_system: bool = True
    tool_schemas: List[Dict] = field(default_factory=list)
    add_eos: bool = False
    eos_token: str = "<|im_end|>"


# =============================================================================
# Base Format Converter
# =============================================================================

class BaseFormatConverter:
    """Base class for all format converters."""

    format_type: FormatType = None

    def __init__(self, config: FormatConfig = None):
        self.config = config or FormatConfig()

    def convert(self, data: List[Dict]) -> List[Dict]:
        """Convert data to target format."""
        raise NotImplementedError

    def convert_single(self, item: Dict) -> Dict:
        """Convert a single item."""
        raise NotImplementedError

    def save(self, data: List[Dict], output_path: Union[str, Path]) -> str:
        """Save converted data to JSONL file."""
        output_path = Path(output_path)
        converted = self.convert(data)

        with open(output_path, 'w', encoding='utf-8') as f:
            for item in converted:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        logger.info(f"Saved {len(converted)} items to {output_path}")
        return str(output_path)


# =============================================================================
# Alpaca Format (Domain Adaptation, Instruction Tuning)
# =============================================================================

class AlpacaFormat(BaseFormatConverter):
    """
    Alpaca format for instruction tuning and domain adaptation.

    Best for: Converting PDFs/documents to domain knowledge.

    Output format:
    {
        "instruction": "What is...",
        "input": "",           # Optional context
        "output": "The answer..."
    }
    """

    format_type = FormatType.ALPACA

    def convert(self, data: List[Dict]) -> List[Dict]:
        """Convert to Alpaca format."""
        result = []
        for item in data:
            converted = self.convert_single(item)
            if converted:
                result.append(converted)
        return result

    def convert_single(self, item: Dict) -> Optional[Dict]:
        """Convert single item to Alpaca format."""
        # Extract instruction (handles multiple key names)
        instruction = (
            item.get('instruction') or
            item.get('question') or
            item.get('q') or
            item.get('prompt', '')
        )

        # Extract output/response
        output = (
            item.get('output') or
            item.get('response') or
            item.get('answer') or
            item.get('a', '')
        )

        # Extract optional input/context
        input_text = (
            item.get('input') or
            item.get('context', '')
        )

        if not instruction or not output:
            return None

        return {
            "instruction": instruction.strip(),
            "input": input_text.strip(),
            "output": output.strip()
        }


# =============================================================================
# ChatML Format (Chatbot/Assistant, Function Calling)
# =============================================================================

class ChatMLFormat(BaseFormatConverter):
    """
    ChatML format for chatbot/assistant training.

    Best for: Building conversational AI assistants.

    Output format:
    {
        "messages": [
            {"role": "system", "content": "You are..."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
    }
    """

    format_type = FormatType.CHATML

    def convert(self, data: List[Dict]) -> List[Dict]:
        """Convert to ChatML format."""
        result = []
        for item in data:
            converted = self.convert_single(item)
            if converted:
                result.append(converted)
        return result

    def convert_single(self, item: Dict) -> Optional[Dict]:
        """Convert single item to ChatML format."""
        messages = []

        # Add system message if configured
        if self.config.include_system and self.config.system_prompt:
            messages.append({
                "role": "system",
                "content": self.config.system_prompt
            })

        # Extract user message
        user_content = (
            item.get('instruction') or
            item.get('question') or
            item.get('q') or
            item.get('user') or
            item.get('prompt', '')
        )

        # Extract assistant response
        assistant_content = (
            item.get('output') or
            item.get('response') or
            item.get('answer') or
            item.get('a') or
            item.get('assistant', '')
        )

        if not user_content or not assistant_content:
            return None

        # Add context to user message if present
        context = item.get('input') or item.get('context', '')
        if context:
            user_content = f"{context}\n\n{user_content}"

        messages.append({"role": "user", "content": user_content.strip()})
        messages.append({"role": "assistant", "content": assistant_content.strip()})

        return {"messages": messages}


class ChatMLToolsFormat(BaseFormatConverter):
    """
    ChatML format with function/tool calling support.

    Best for: Building AI agents with tool use capabilities.

    Output format:
    {
        "messages": [...],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather info",
                    "parameters": {...}
                }
            }
        ]
    }
    """

    format_type = FormatType.CHATML_TOOLS

    def convert(self, data: List[Dict]) -> List[Dict]:
        """Convert to ChatML with tools format."""
        result = []
        for item in data:
            converted = self.convert_single(item)
            if converted:
                result.append(converted)
        return result

    def convert_single(self, item: Dict) -> Optional[Dict]:
        """Convert single item to ChatML tools format."""
        messages = []

        # System message with tool instructions
        if self.config.include_system:
            system_content = self.config.system_prompt or "You are a helpful assistant with access to tools."
            messages.append({"role": "system", "content": system_content})

        # User message
        user_content = item.get('instruction') or item.get('question') or item.get('user', '')
        if not user_content:
            return None
        messages.append({"role": "user", "content": user_content.strip()})

        # Tool call (if present)
        tool_call = item.get('tool_call') or item.get('function_call')
        if tool_call:
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [tool_call] if isinstance(tool_call, dict) else tool_call
            })

            # Tool response
            tool_response = item.get('tool_response') or item.get('function_response', '')
            if tool_response:
                messages.append({
                    "role": "tool",
                    "content": str(tool_response),
                    "tool_call_id": tool_call.get('id', 'call_0') if isinstance(tool_call, dict) else 'call_0'
                })

        # Final assistant response
        assistant_content = item.get('output') or item.get('response') or item.get('assistant', '')
        if assistant_content:
            messages.append({"role": "assistant", "content": assistant_content.strip()})

        result = {"messages": messages}

        # Add tool schemas if configured
        if self.config.tool_schemas:
            result["tools"] = self.config.tool_schemas
        elif item.get('tools'):
            result["tools"] = item['tools']

        return result


# =============================================================================
# ShareGPT Format (Multi-turn Conversations)
# =============================================================================

class ShareGPTFormat(BaseFormatConverter):
    """
    ShareGPT format for multi-turn conversation training.

    Best for: Training on conversation history with multiple turns.

    Output format:
    {
        "conversations": [
            {"from": "system", "value": "You are..."},
            {"from": "human", "value": "Hello"},
            {"from": "gpt", "value": "Hi there!"},
            {"from": "human", "value": "How are you?"},
            {"from": "gpt", "value": "I'm doing well!"}
        ]
    }
    """

    format_type = FormatType.SHAREGPT

    def convert(self, data: List[Dict]) -> List[Dict]:
        """Convert to ShareGPT format."""
        result = []
        for item in data:
            converted = self.convert_single(item)
            if converted:
                result.append(converted)
        return result

    def convert_single(self, item: Dict) -> Optional[Dict]:
        """Convert single item to ShareGPT format."""
        conversations = []

        # System message
        if self.config.include_system and self.config.system_prompt:
            conversations.append({
                "from": "system",
                "value": self.config.system_prompt
            })

        # Check if item already has conversations array
        if 'conversations' in item:
            # Normalize existing conversations
            for conv in item['conversations']:
                role = conv.get('from') or conv.get('role', '')
                value = conv.get('value') or conv.get('content', '')

                # Normalize role names
                if role in ['user', 'human']:
                    role = 'human'
                elif role in ['assistant', 'gpt', 'bot']:
                    role = 'gpt'

                conversations.append({"from": role, "value": value})
            return {"conversations": conversations}

        # Convert single Q&A to conversation
        human_val = (
            item.get('question') or
            item.get('q') or
            item.get('instruction') or
            item.get('user', '')
        )

        gpt_val = (
            item.get('answer') or
            item.get('a') or
            item.get('output') or
            item.get('response') or
            item.get('assistant', '')
        )

        if not human_val or not gpt_val:
            return None

        conversations.append({"from": "human", "value": human_val.strip()})
        conversations.append({"from": "gpt", "value": gpt_val.strip()})

        return {"conversations": conversations}


# =============================================================================
# Completion Format (Base Model Continuation)
# =============================================================================

class CompletionFormat(BaseFormatConverter):
    """
    Completion format for base model continuation training.

    Best for: Continued pretraining, raw text completion.

    Output format:
    {
        "text": "The full text content for training..."
    }
    """

    format_type = FormatType.COMPLETION

    def convert(self, data: List[Dict]) -> List[Dict]:
        """Convert to completion format."""
        result = []
        for item in data:
            converted = self.convert_single(item)
            if converted:
                result.append(converted)
        return result

    def convert_single(self, item: Dict) -> Optional[Dict]:
        """Convert single item to completion format."""
        # Just raw text
        if 'text' in item:
            text = item['text']
        else:
            # Combine instruction + response into completion text
            instruction = item.get('instruction') or item.get('question', '')
            response = item.get('output') or item.get('response') or item.get('answer', '')

            if instruction and response:
                text = f"{instruction}\n\n{response}"
            elif instruction:
                text = instruction
            elif response:
                text = response
            else:
                return None

        if not text.strip():
            return None

        result = {"text": text.strip()}

        if self.config.add_eos:
            result["text"] += self.config.eos_token

        return result


# =============================================================================
# DPO Format (Direct Preference Optimization / RLHF)
# =============================================================================

class DPOFormat(BaseFormatConverter):
    """
    DPO (Direct Preference Optimization) format for RLHF alignment.

    Best for: Quality alignment, preference-based training.

    Output format:
    {
        "prompt": "User question...",
        "chosen": "Good response...",
        "rejected": "Bad response..."
    }
    """

    format_type = FormatType.DPO

    def convert(self, data: List[Dict]) -> List[Dict]:
        """Convert to DPO format."""
        result = []
        for item in data:
            converted = self.convert_single(item)
            if converted:
                result.append(converted)
        return result

    def convert_single(self, item: Dict) -> Optional[Dict]:
        """Convert single item to DPO format."""
        # Extract prompt
        prompt = (
            item.get('prompt') or
            item.get('instruction') or
            item.get('question') or
            item.get('input', '')
        )

        # Extract chosen (good) response
        chosen = (
            item.get('chosen') or
            item.get('preferred') or
            item.get('good_response') or
            item.get('output') or
            item.get('response', '')
        )

        # Extract rejected (bad) response
        rejected = (
            item.get('rejected') or
            item.get('dispreferred') or
            item.get('bad_response', '')
        )

        if not prompt or not chosen:
            return None

        result = {
            "prompt": prompt.strip(),
            "chosen": chosen.strip(),
        }

        # Rejected is optional (can be generated later)
        if rejected:
            result["rejected"] = rejected.strip()

        return result


# =============================================================================
# Format Registry & Factory
# =============================================================================

class FormatRegistry:
    """Registry of all available format converters."""

    _formats: Dict[str, type] = {
        "alpaca": AlpacaFormat,
        "chatml": ChatMLFormat,
        "sharegpt": ShareGPTFormat,
        "completion": CompletionFormat,
        "dpo": DPOFormat,
        "chatml_tools": ChatMLToolsFormat,
        # Aliases
        "messages": ChatMLFormat,
        "openai": ChatMLFormat,
        "preference": DPOFormat,
        "rlhf": DPOFormat,
    }

    @classmethod
    def get(cls, format_name: str, config: FormatConfig = None) -> BaseFormatConverter:
        """Get a format converter by name."""
        format_name = format_name.lower().strip()

        if format_name not in cls._formats:
            available = list(set(cls._formats.keys()) - {'messages', 'openai', 'preference', 'rlhf'})
            raise ValueError(
                f"Unknown format: '{format_name}'. "
                f"Available formats: {', '.join(sorted(available))}"
            )

        return cls._formats[format_name](config)

    @classmethod
    def list_formats(cls) -> List[str]:
        """List all available format names (excluding aliases)."""
        return ["alpaca", "chatml", "sharegpt", "completion", "dpo", "chatml_tools"]

    @classmethod
    def get_recommendation(cls, use_case: str) -> str:
        """Get recommended format for a use case."""
        recommendations = {
            "domain_adaptation": "alpaca",
            "pdf": "alpaca",
            "knowledge": "alpaca",
            "chatbot": "chatml",
            "assistant": "chatml",
            "multi_turn": "sharegpt",
            "conversation": "sharegpt",
            "history": "sharegpt",
            "pretraining": "completion",
            "continuation": "completion",
            "base_model": "completion",
            "rlhf": "dpo",
            "alignment": "dpo",
            "preference": "dpo",
            "function_calling": "chatml_tools",
            "tools": "chatml_tools",
            "agent": "chatml_tools",
        }

        use_case_lower = use_case.lower().replace(" ", "_").replace("-", "_")
        return recommendations.get(use_case_lower, "alpaca")


# =============================================================================
# Convenience Functions
# =============================================================================

def convert_dataset(
    data: List[Dict],
    target_format: str,
    system_prompt: str = "",
    tool_schemas: List[Dict] = None,
    output_path: Union[str, Path] = None
) -> Union[List[Dict], str]:
    """
    Convert dataset to a target format.

    Args:
        data: List of data items
        target_format: One of: alpaca, chatml, sharegpt, completion, dpo, chatml_tools
        system_prompt: Optional system prompt for chat formats
        tool_schemas: Optional tool definitions for chatml_tools format
        output_path: Optional path to save JSONL file

    Returns:
        Converted data list, or file path if output_path is provided
    """
    config = FormatConfig(
        system_prompt=system_prompt,
        tool_schemas=tool_schemas or []
    )

    converter = FormatRegistry.get(target_format, config)

    if output_path:
        return converter.save(data, output_path)
    else:
        return converter.convert(data)


def load_and_convert(
    input_path: Union[str, Path],
    target_format: str,
    output_path: Union[str, Path] = None,
    system_prompt: str = ""
) -> Union[List[Dict], str]:
    """
    Load JSONL file and convert to target format.

    Args:
        input_path: Path to input JSONL file
        target_format: Target format name
        output_path: Optional output path
        system_prompt: Optional system prompt

    Returns:
        Converted data or output file path
    """
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    return convert_dataset(
        data,
        target_format,
        system_prompt=system_prompt,
        output_path=output_path
    )


# =============================================================================
# Format Documentation
# =============================================================================

FORMAT_GUIDE = """
Dataset Format Selection Guide
==============================

| Use Case                      | Recommended Format | Description                        |
|-------------------------------|-------------------|------------------------------------|
| Domain adaptation (PDFs)      | alpaca            | Simple instruction/input/output    |
| Chatbot / assistant           | chatml            | OpenAI-style messages array        |
| Multi-turn with history       | sharegpt          | Conversation format with turns     |
| Base model continuation       | completion        | Raw text for pretraining           |
| Quality alignment / RLHF      | dpo               | Preference pairs (chosen/rejected) |
| Function calling / tools      | chatml_tools      | Messages with tool schemas         |

Example Usage:
--------------
    from saara.formats import convert_dataset, FormatRegistry

    # Convert Q&A data to Alpaca format
    alpaca_data = convert_dataset(qa_pairs, "alpaca")

    # Convert to ChatML with system prompt
    chatml_data = convert_dataset(
        data,
        "chatml",
        system_prompt="You are a helpful assistant."
    )

    # Get format recommendation
    format_name = FormatRegistry.get_recommendation("chatbot")
"""
