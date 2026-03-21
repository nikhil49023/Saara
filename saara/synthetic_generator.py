"""
Synthetic Data Generator Module
Generates high-quality training data with multiple types:
- Factual QA (fact retrieval)
- Reasoning QA (why/how questions)
- Conversational (multi-turn simulation)
"""

import logging
import json
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum

from .ollama_client import OllamaClient

logger = logging.getLogger(__name__)


class DataType(Enum):
    """Types of synthetic data to generate."""
    FACTUAL = "factual"         # Simple fact retrieval
    REASONING = "reasoning"      # Why/How questions requiring logic
    CONVERSATIONAL = "conversational"  # Multi-turn scenarios
    INSTRUCTION = "instruction"  # Task-based instructions
    ALL = "all"


@dataclass
class GeneratedSample:
    """A single generated training sample."""
    instruction: str
    input_context: str
    output: str
    data_type: str
    source_chunk: str
    quality_score: float = 0.0
    passed_filter: bool = True
    rejection_reason: str = ""


@dataclass
class GenerationResult:
    """Result of synthetic data generation."""
    samples: List[GeneratedSample] = field(default_factory=list)
    total_generated: int = 0
    total_passed: int = 0
    total_rejected: int = 0
    rejection_stats: Dict[str, int] = field(default_factory=dict)


class SyntheticDataGenerator:
    """
    Generates high-quality synthetic training data using a Teacher model.
    Implements multi-type generation and quality filtering.
    """
    
    # Prompts for different data types
    PROMPTS = {
        DataType.FACTUAL: """Based on the following document section, generate {count} factual question-answer pairs.

RULES:
- Questions should ask for specific facts, definitions, or data points
- Answers must be directly supported by the text
- DO NOT use phrases like "according to the text" or "the document states"
- Questions should sound natural, as if asked by a curious user
- Answers should be complete but concise

DOCUMENT SECTION:
{text}

Generate exactly {count} pairs in this JSON format:
[
  {{"instruction": "question here", "input": "", "output": "answer here"}},
  ...
]

JSON OUTPUT:""",

        DataType.REASONING: """Based on the following document section, generate {count} reasoning question-answer pairs.

RULES:
- Questions should require logical thinking, analysis, or inference
- Use "why", "how", "what would happen if", "explain the relationship"
- Answers should show step-by-step reasoning
- DO NOT reference "the text" or "the document" - just answer naturally
- Answers should demonstrate deep understanding

DOCUMENT SECTION:
{text}

Generate exactly {count} pairs in this JSON format:
[
  {{"instruction": "reasoning question", "input": "", "output": "detailed reasoning answer"}},
  ...
]

JSON OUTPUT:""",

        DataType.CONVERSATIONAL: """Based on the following document section, generate {count} conversational scenarios.

RULES:
- Frame as a user with a practical problem or question
- Example: "I'm trying to understand X..." or "I need help with Y..."
- Responses should be helpful, practical, and conversational
- DO NOT sound like a textbook - sound like a knowledgeable assistant
- Include context when helpful

DOCUMENT SECTION:
{text}

Generate exactly {count} pairs in this JSON format:
[
  {{"instruction": "user's conversational question/problem", "input": "", "output": "helpful assistant response"}},
  ...
]

JSON OUTPUT:""",

        DataType.INSTRUCTION: """Based on the following document section, generate {count} instruction-following pairs.

RULES:
- Frame as tasks the user wants to accomplish
- Example: "Summarize...", "List the steps to...", "Compare X and Y..."
- Responses should complete the task accurately
- Be specific and actionable

DOCUMENT SECTION:
{text}

Generate exactly {count} pairs in this JSON format:
[
  {{"instruction": "task instruction", "input": "", "output": "completed task response"}},
  ...
]

JSON OUTPUT:"""
    }
    
    # Quality filter patterns
    REJECT_PATTERNS = [
        r"(?i)according to (the|this) (text|document|passage|article)",
        r"(?i)the (text|document|passage) (mentions|states|says|describes)",
        r"(?i)as (stated|mentioned|described) in",
        r"(?i)based on (the|this) (text|document|passage)",
        r"(?i)the author (says|states|mentions)",
        r"(?i)in (the|this) (section|passage|document)",
    ]
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.client = OllamaClient(self.config.get('ollama', {}))
        
        gen_config = self.config.get('generation', {})
        self.pairs_per_type = gen_config.get('pairs_per_type', 3)
        self.min_answer_words = gen_config.get('min_answer_words', 5)
        self.max_answer_words = gen_config.get('max_answer_words', 500)
        self.enable_quality_filter = gen_config.get('enable_quality_filter', True)
        
    def generate(
        self, 
        chunk: str, 
        data_types: List[DataType] = None,
        pairs_per_type: int = None
    ) -> GenerationResult:
        """
        Generate synthetic training data from a text chunk.
        
        Args:
            chunk: Source text to generate from
            data_types: Types of data to generate (default: all)
            pairs_per_type: Number of pairs per type (default: from config)
            
        Returns:
            GenerationResult with samples and statistics
        """
        if data_types is None:
            data_types = [DataType.FACTUAL, DataType.REASONING, DataType.CONVERSATIONAL]
        elif DataType.ALL in data_types:
            data_types = [DataType.FACTUAL, DataType.REASONING, DataType.CONVERSATIONAL, DataType.INSTRUCTION]
            
        if pairs_per_type is None:
            pairs_per_type = self.pairs_per_type
            
        result = GenerationResult()
        
        for dtype in data_types:
            samples = self._generate_type(chunk, dtype, pairs_per_type)
            
            for sample in samples:
                result.total_generated += 1
                
                # Apply quality filter
                if self.enable_quality_filter:
                    passed, reason = self._quality_check(sample)
                    sample.passed_filter = passed
                    sample.rejection_reason = reason
                    
                    if passed:
                        result.total_passed += 1
                        result.samples.append(sample)
                    else:
                        result.total_rejected += 1
                        result.rejection_stats[reason] = result.rejection_stats.get(reason, 0) + 1
                else:
                    result.samples.append(sample)
                    result.total_passed += 1
        
        return result
    
    def _generate_type(self, chunk: str, dtype: DataType, count: int) -> List[GeneratedSample]:
        """Generate samples of a specific type."""
        prompt_template = self.PROMPTS.get(dtype)
        if not prompt_template:
            logger.warning(f"No prompt template for type: {dtype}")
            return []
        
        prompt = prompt_template.format(text=chunk[:3000], count=count)
        
        response = self.client.generate(
            prompt=prompt,
            system_prompt="You are an expert at creating high-quality training data for AI models. Output valid JSON only."
        )
        
        if not response.success:
            logger.warning(f"Generation failed for {dtype}: {response.error}")
            return []
        
        # Parse JSON response
        samples = self._parse_response(response.content, dtype, chunk)
        return samples
    
    def _parse_response(self, content: str, dtype: DataType, source_chunk: str) -> List[GeneratedSample]:
        """Parse LLM response into GeneratedSample objects."""
        samples = []
        
        # Try to extract JSON from response
        try:
            # Find JSON array in response
            json_match = re.search(r'\[[\s\S]*\]', content)
            if json_match:
                data = json.loads(json_match.group())
            else:
                # Try parsing entire content as JSON
                data = json.loads(content)
            
            if not isinstance(data, list):
                data = [data]
                
            for item in data:
                if isinstance(item, dict):
                    sample = GeneratedSample(
                        instruction=item.get('instruction', ''),
                        input_context=item.get('input', ''),
                        output=item.get('output', item.get('response', '')),
                        data_type=dtype.value,
                        source_chunk=source_chunk[:500]  # Truncate for storage
                    )
                    samples.append(sample)
                    
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            # Try line-by-line extraction as fallback
            samples = self._fallback_parse(content, dtype, source_chunk)
        
        return samples
    
    def _fallback_parse(self, content: str, dtype: DataType, source_chunk: str) -> List[GeneratedSample]:
        """Fallback parser for non-JSON responses."""
        samples = []
        
        # Try to find Q/A patterns
        qa_pattern = r'(?:Q|Question|Instruction)[:\s]*(.+?)(?:\n|$)[\s]*(?:A|Answer|Response|Output)[:\s]*(.+?)(?:\n\n|$)'
        matches = re.findall(qa_pattern, content, re.IGNORECASE | re.DOTALL)
        
        for q, a in matches:
            sample = GeneratedSample(
                instruction=q.strip(),
                input_context="",
                output=a.strip(),
                data_type=dtype.value,
                source_chunk=source_chunk[:500]
            )
            samples.append(sample)
        
        return samples
    
    def _quality_check(self, sample: GeneratedSample) -> tuple[bool, str]:
        """
        Check sample quality and return (passed, rejection_reason).
        """
        instruction = sample.instruction
        output = sample.output
        
        # Rule 1: Check minimum answer length
        word_count = len(output.split())
        if word_count < self.min_answer_words:
            return False, "answer_too_short"
        
        if word_count > self.max_answer_words:
            return False, "answer_too_long"
        
        # Rule 2: Check for document reference patterns
        for pattern in self.REJECT_PATTERNS:
            if re.search(pattern, instruction):
                return False, "instruction_references_document"
            if re.search(pattern, output):
                return False, "output_references_document"
        
        # Rule 3: Check for empty or placeholder content
        if not instruction.strip() or not output.strip():
            return False, "empty_content"
        
        if instruction.lower() in ['question', 'q', 'instruction']:
            return False, "placeholder_instruction"
        
        # Rule 4: Check for hallucination markers
        hallucination_markers = [
            r"(?i)i (don't|do not) have (access|information)",
            r"(?i)i cannot (access|see|read)",
            r"(?i)as an ai",
            r"(?i)i'm (just|only) an ai",
        ]
        for pattern in hallucination_markers:
            if re.search(pattern, output):
                return False, "hallucination_marker"
        
        return True, ""
    
    def generate_from_chunks(
        self, 
        chunks: List[Dict[str, Any]], 
        data_types: List[DataType] = None
    ) -> GenerationResult:
        """
        Generate training data from multiple chunks.
        
        Args:
            chunks: List of chunk dictionaries with 'content' key
            data_types: Types of data to generate
            
        Returns:
            Combined GenerationResult
        """
        combined = GenerationResult()
        
        for chunk in chunks:
            content = chunk.get('content', chunk.get('text', ''))
            if not content:
                continue
                
            result = self.generate(content, data_types)
            
            combined.samples.extend(result.samples)
            combined.total_generated += result.total_generated
            combined.total_passed += result.total_passed
            combined.total_rejected += result.total_rejected
            
            for reason, count in result.rejection_stats.items():
                combined.rejection_stats[reason] = combined.rejection_stats.get(reason, 0) + count
        
        return combined


class QualityJudge:
    """
    Additional quality control using LLM-as-Judge pattern.
    Evaluates samples on multiple criteria.
    """
    
    JUDGE_PROMPT = """You are a quality control judge for AI training data.

Evaluate the following instruction-response pair on these criteria:
1. RELEVANCE (1-10): Does the response actually answer the instruction?
2. ACCURACY (1-10): Is the information factually correct?
3. NATURALNESS (1-10): Does it sound like a real conversation?
4. COMPLETENESS (1-10): Is the response thorough enough?

INSTRUCTION: {instruction}
RESPONSE: {response}

Provide your evaluation as JSON:
{{"relevance": X, "accuracy": X, "naturalness": X, "completeness": X, "average": X, "passed": true/false, "issues": ["issue1", ...]}}

Only mark passed=false if average < 6 or if there are critical issues.

JSON:"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.client = OllamaClient(self.config.get('ollama', {}))
        self.min_score = self.config.get('quality', {}).get('min_score', 6.0)
        
    def judge(self, sample: GeneratedSample) -> Dict[str, Any]:
        """
        Judge a single sample's quality.
        
        Returns:
            Dictionary with scores and pass/fail status
        """
        prompt = self.JUDGE_PROMPT.format(
            instruction=sample.instruction,
            response=sample.output
        )
        
        response = self.client.generate(
            prompt=prompt,
            system_prompt="You are a strict but fair quality judge. Output only valid JSON."
        )
        
        if not response.success:
            return {"passed": True, "error": response.error}
        
        try:
            # Extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response.content)
            if json_match:
                result = json.loads(json_match.group())
                return result
        except:
            pass
        
        return {"passed": True, "error": "parse_failed"}
    
    def filter_samples(
        self, 
        samples: List[GeneratedSample], 
        use_llm_judge: bool = False
    ) -> List[GeneratedSample]:
        """
        Filter samples through quality checks.
        
        Args:
            samples: List of samples to filter
            use_llm_judge: Whether to use LLM-as-Judge (slower but more thorough)
            
        Returns:
            Filtered list of high-quality samples
        """
        passed = []
        
        for sample in samples:
            if not sample.passed_filter:
                continue
            
            if use_llm_judge:
                judgment = self.judge(sample)
                if judgment.get('passed', True):
                    sample.quality_score = judgment.get('average', 7.0)
                    passed.append(sample)
            else:
                passed.append(sample)
        
        return passed
