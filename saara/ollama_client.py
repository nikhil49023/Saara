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
    """Pre-defined prompt templates for common tasks."""
    
    CLASSIFY_DOCUMENT = """Analyze the following document excerpt and classify it.

Document:
{text}

Classify into one of these categories:
- research_paper: Academic research papers with methodology, results, citations
- textbook: Educational content with explanations and examples
- technical_documentation: Technical guides, API docs, manuals
- tutorial: Step-by-step guides and how-to content
- reference_material: Reference docs, specifications, standards
- general_knowledge: General informational content

Respond with JSON:
{{
    "category": "<category>",
    "confidence": <0.0-1.0>,
    "reasoning": "<brief explanation>"
}}"""

    EXTRACT_TOPICS = """Extract the main topics and concepts from this text.

Text:
{text}

Respond with JSON:
{{
    "main_topic": "<primary topic>",
    "subtopics": ["<topic1>", "<topic2>", ...],
    "keywords": ["<keyword1>", "<keyword2>", ...],
    "domain": "<field/domain>",
    "complexity_level": "<basic|intermediate|advanced|expert>"
}}"""

    GENERATE_QA = """You are an expert Ayurvedic Data Analyst.
Read the following text chunk from an Ayurveda textbook.
Your goal is to extract EVERY valid piece of medical information into Q&A pairs.

Rules:
1. Don't summarize. If the text lists 10 herbs, create 10 separate Q&A pairs.
2. Structure:
   - Q: What is the effect of [Herb Name] on [Disease/Dosha]?
   - A: [Herb Name] acts as... (Cite the text directly).
3. Context: If the text says "It cures fever," replace "It" with the actual herb name from the previous sentences.

Text:
{text}

Output format: JSON List of objects:
[
   {{"question": "...", "answer": "..."}},
   ...
]"""

    SUMMARIZE = """Summarize the following text concisely while retaining key information.

Text:
{text}

Respond with JSON:
{{
    "summary": "<concise summary>",
    "key_points": ["<point1>", "<point2>", ...],
    "word_count": <original word count>,
    "compression_ratio": <summary words / original words>
}}"""

    EXTRACT_ENTITIES = """Extract named entities from the following text.

Text:
{text}

Respond with JSON:
{{
    "entities": [
        {{"text": "<entity>", "type": "<PERSON|ORG|LOCATION|DATE|CONCEPT|TECHNOLOGY|OTHER>", "context": "<brief context>"}}
    ]
}}"""

    CREATE_INSTRUCTION = """Transform the following content into an instruction-following format for training.

Content:
{text}

Create an instruction and response pair where:
- The instruction asks about or requests something related to the content
- The response provides accurate information based on the content

Respond with JSON:
{{
    "instruction": "<user instruction/question>",
    "response": "<helpful response based on content>",
    "category": "<category of instruction>"
}}"""

    ASSESS_QUALITY = """Assess the quality of this text for training data.

Text:
{text}

Evaluate and respond with JSON:
{{
    "quality_score": <1-10>,
    "issues": ["<issue1>", ...],
    "is_suitable": <true|false>,
    "language": "<detected language>",
    "contains_code": <true|false>,
    "contains_math": <true|false>,
    "readability": "<easy|medium|hard>"
}}"""
