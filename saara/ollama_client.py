"""
Ollama Integration Module
Handles communication with Ollama API and Granite 4.0 model.
"""

import ollama
import httpx
import json
import logging
from typing import Dict, Any, List, Optional, Generator
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Response from LLM."""
    content: str
    model: str
    total_duration: float = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    success: bool = True
    error: Optional[str] = None


class OllamaClient:
    """
    Client for Ollama API with Granite 4.0 model support.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        self.base_url = config.get('base_url', 'http://localhost:11434')
        self.model = config.get('model', 'granite4')
        self.timeout = config.get('timeout', 300)
        self.max_retries = config.get('max_retries', 3)
        
        # Initialize ollama client
        self.client = ollama.Client(host=self.base_url)
        
    def generate(self, 
                 prompt: str, 
                 system_prompt: str = None,
                 temperature: float = 0.7,
                 max_tokens: int = 2048) -> LLMResponse:
        """
        Generate a response from the model.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            LLMResponse object
        """
        messages = []
        
        if system_prompt:
            messages.append({
                'role': 'system',
                'content': system_prompt
            })
            
        messages.append({
            'role': 'user',
            'content': prompt
        })
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat(
                    model=self.model,
                    messages=messages,
                    options={
                        'temperature': temperature,
                        'num_predict': max_tokens,
                    }
                )
                
                return LLMResponse(
                    content=response['message']['content'],
                    model=self.model,
                    total_duration=response.get('total_duration', 0) / 1e9,  # Convert to seconds
                    prompt_tokens=response.get('prompt_eval_count', 0),
                    completion_tokens=response.get('eval_count', 0),
                    success=True
                )
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    return LLMResponse(
                        content="",
                        model=self.model,
                        success=False,
                        error=str(e)
                    )
    
    def generate_json(self, 
                      prompt: str, 
                      system_prompt: str = None,
                      schema: Dict = None) -> Dict[str, Any]:
        """
        Generate a JSON response from the model.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt (will add JSON instruction)
            schema: Optional JSON schema for validation
            
        Returns:
            Parsed JSON dictionary
        """
        json_instruction = "\n\nRespond with valid JSON only. No markdown, no explanations."
        
        if system_prompt:
            system_prompt += json_instruction
        else:
            system_prompt = "You are a helpful assistant that responds in JSON format." + json_instruction
            
        if schema:
            system_prompt += f"\n\nFollow this JSON schema:\n{json.dumps(schema, indent=2)}"
        
        response = self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.3  # Lower temperature for more consistent JSON
        )
        
        if not response.success:
            return {'error': response.error}
            
        try:
            # Try to extract JSON from response
            content = response.content.strip()
            
            # Find the first { or [ and last } or ]
            start_idx = -1
            end_idx = -1
            
            # Find start
            for i, char in enumerate(content):
                if char in ['{', '[']:
                    start_idx = i
                    break
            
            # Find end
            for i in range(len(content) - 1, -1, -1):
                if content[i] in ['}', ']']:
                    end_idx = i + 1
                    break
            
            if start_idx != -1 and end_idx != -1:
                content = content[start_idx:end_idx]
            
            return json.loads(content)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            logger.debug(f"Raw response: {response.content}")
            # Instead of returning error dict, return empty result to allow graceful degradation
            if schema and isinstance(schema, list) or (isinstance(schema, dict) and 'type' in schema and schema['type'] == 'array'):
                 return []
            return {}
    
    def stream_generate(self, 
                        prompt: str,
                        system_prompt: str = None) -> Generator[str, None, None]:
        """
        Stream responses from the model.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            
        Yields:
            Response chunks
        """
        messages = []
        
        if system_prompt:
            messages.append({
                'role': 'system',
                'content': system_prompt
            })
            
        messages.append({
            'role': 'user',
            'content': prompt
        })
        
        try:
            stream = self.client.chat(
                model=self.model,
                messages=messages,
                stream=True
            )
            
            for chunk in stream:
                yield chunk['message']['content']
                
        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"[Error: {e}]"
    
    def check_health(self) -> bool:
        """Check if Ollama is running and model is available."""
        try:
            models = self.client.list()
            # Handle response whether it's a dict or object
            if hasattr(models, 'models'):
                model_list = models.models
            else:
                model_list = models.get('models', [])
            
            available_models = []
            for m in model_list:
                # Handle model entry whether it's a dict or object
                if hasattr(m, 'model'):
                    name = m.model
                elif isinstance(m, dict) and 'name' in m:
                    name = m['name']
                elif isinstance(m, dict) and 'model' in m:
                    name = m['model']
                else:
                    logger.warning(f"Unexpected model format: {m}")
                    continue
                    
                available_models.append(name)
                # Also add base name (e.g., granite4 from granite4:latest)
                if ':' in name:
                    available_models.append(name.split(':')[0])
            
            if self.model not in available_models and f"{self.model}:latest" not in available_models:
                logger.warning(f"Model {self.model} not found. Available: {available_models}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        try:
            info = self.client.show(self.model)
            return {
                'name': self.model,
                'size': info.get('size', 0),
                'parameter_size': info.get('details', {}).get('parameter_size', 'unknown'),
                'family': info.get('details', {}).get('family', 'unknown'),
                'format': info.get('details', {}).get('format', 'unknown'),
                'quantization': info.get('details', {}).get('quantization_level', 'unknown')
            }
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {'error': str(e)}


class PromptTemplates:
    """Pre-defined prompt templates for common data labeling and analysis tasks."""

    CLASSIFY_DOCUMENT = """Analyze the following document excerpt and classify it.

Document:
{text}

Classify into one of these categories:
- research_paper: Academic research with methodology, results, citations
- textbook: Educational content with structured explanations and examples
- technical_documentation: Technical guides, API docs, system manuals
- tutorial: Step-by-step how-to guides and walkthroughs
- reference_material: Reference specifications, standards, lookup tables
- medical_clinical: Clinical guidelines, medical literature, health protocols
- legal: Legal documents, contracts, regulations, compliance material
- financial: Financial reports, economic analysis, investment material
- scientific: Scientific papers, lab reports, experimental data
- general_knowledge: General informational or encyclopedic content

Respond with JSON:
{{
    "category": "<category>",
    "confidence": <0.0-1.0>,
    "reasoning": "<one sentence explanation>"
}}"""

    EXTRACT_TOPICS = """Extract the main topics and concepts from this text.

Text:
{text}

Respond with JSON:
{{
    "main_topic": "<primary topic in 3-5 words>",
    "subtopics": ["<topic1>", "<topic2>", "<topic3>"],
    "keywords": ["<keyword1>", "<keyword2>", "<keyword3>", "<keyword4>", "<keyword5>"],
    "domain": "<field/domain>",
    "complexity_level": "<basic|intermediate|advanced|expert>"
}}"""

    GENERATE_QA = """You are a rigorous training data specialist. Your task is to extract every distinct, verifiable fact from the text below into well-formed question-answer pairs for LLM fine-tuning.

STRICT RULES:
1. Extract EVERY separate fact — do not merge or summarize multiple facts into one pair.
2. Questions must be self-contained: do not use vague pronouns like "it", "they", "this" — always name the specific subject.
3. Answers must be factually grounded in the source text. Do not add, infer, or hallucinate information.
4. NEVER use phrases like "according to the text", "the document states", "as mentioned above" — write as if the answer is simply known.
5. Questions should sound natural, as if asked by a knowledgeable user seeking information.
6. Answers should be complete but concise — no unnecessary filler or hedging.
7. Output ONLY valid JSON — no markdown, no commentary.

TEXT:
{text}

Output format — JSON array:
[
   {{"question": "...", "answer": "..."}},
   {{"question": "...", "answer": "..."}}
]"""

    SUMMARIZE = """Summarize the following text faithfully and concisely. Preserve all key facts without adding, inferring, or hallucinating information that is not present in the source.

Text:
{text}

Respond with JSON:
{{
    "summary": "<concise, faithful summary — no invented facts>",
    "key_points": ["<point1>", "<point2>", "<point3>"],
    "word_count": <original approximate word count>,
    "compression_ratio": <summary word count divided by original, as a decimal>
}}"""

    EXTRACT_ENTITIES = """Extract all named entities from the following text with careful attention to context.

Text:
{text}

Respond with JSON:
{{
    "entities": [
        {{"text": "<entity>", "type": "<PERSON|ORGANIZATION|LOCATION|DATE|CONCEPT|TECHNOLOGY|PRODUCT|EVENT|OTHER>", "context": "<one sentence describing how this entity appears in the text>"}}
    ]
}}"""

    CREATE_INSTRUCTION = """You are an expert LLM fine-tuning data engineer. Generate a single, high-quality instruction-response pair from the content below, suitable for supervised fine-tuning.

Content:
{text}

Choose the most natural instruction type for this content:
- EXPLANATION: "Explain ...", "Describe how ...", "What is ..."
- PROCEDURE: "List the steps to ...", "How do you ...", "Walk me through ..."
- COMPARISON: "Compare X and Y ...", "What is the difference between ..."
- APPLICATION: "Give an example of ...", "How would you apply ..."
- ANALYSIS: "Why does ...", "What are the implications of ..."

Rules:
- The instruction must be self-contained — no references to 'the text' or 'the passage'.
- The response must be accurate, detailed, and directly address the instruction using information from the content.
- The response should demonstrate understanding, not just copy the source.
- Minimum response length: 30 words.
- NEVER start responses with 'I' or 'As an AI'.

Respond with JSON:
{{
    "instruction": "<natural user instruction>",
    "response": "<thorough, accurate response>",
    "category": "<explanation|procedure|comparison|application|analysis>"
}}"""

    ASSESS_QUALITY = """You are a quality gatekeeper for LLM training data. Evaluate the text below across 6 dimensions and decide whether it is suitable for inclusion in a training dataset.

Text:
{text}

Scoring dimensions (1-10 each):
1. INFORMATIVENESS: Does it convey meaningful, substantive information?
2. FACTUAL_COHERENCE: Is the content internally consistent and non-contradictory?
3. SPECIFICITY: Does it contain concrete details rather than vague generalities?
4. LANGUAGE_QUALITY: Is it grammatically sound and clearly written?
5. TRAINING_UTILITY: Would an AI model learn something useful from this text?
6. NOISE_FREEDOM: Is it free from OCR errors, formatting artifacts, or garbled content?

Mark is_suitable=false if ANY of the following is true:
- average score < 5.0
- noise_freedom < 4 (heavily corrupted text)
- language_quality < 3 (unintelligible)

Respond with JSON:
{{
    "informativeness": <1-10>,
    "factual_coherence": <1-10>,
    "specificity": <1-10>,
    "language_quality": <1-10>,
    "training_utility": <1-10>,
    "noise_freedom": <1-10>,
    "quality_score": <average of all 6, rounded to 1 decimal>,
    "issues": ["<issue1>"],
    "is_suitable": <true|false>,
    "language": "<detected language code, e.g. en>",
    "contains_code": <true|false>,
    "contains_math": <true|false>,
    "readability": "<easy|medium|hard>"
}}"""
