"""
Text Cleaner Module
Sanitizes OCR output by removing conversational filler and enforcing strict Markdown.
"""

import logging
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from .ollama_client import OllamaClient

logger = logging.getLogger(__name__)


@dataclass
class CleanedText:
    """Result of text cleaning."""
    original: str
    cleaned: str
    removed_phrases: List[str]
    confidence: float


class TextCleaner:
    """
    Cleans and sanitizes OCR output from vision models.
    Removes conversational filler and enforces strict Markdown format.
    """
    
    # Common filler phrases from vision models
    FILLER_PATTERNS = [
        r"(?i)^(here is|the image shows|this image contains|the text reads|i can see)[^.]*[.:]?\s*",
        r"(?i)^(the document shows|this document contains|the page shows)[^.]*[.:]?\s*",
        r"(?i)^(let me|i'll|i will|allow me to)[^.]*[.:]?\s*",
        r"(?i)^(based on the image|from the image|looking at)[^.]*[.:]?\s*",
        r"(?i)^(the following|below is|here's)[^.]*[.:]?\s*",
        r"(?i)(hope this helps|let me know if)[^.]*$",
        r"(?i)(as shown in the image|as visible in)[^.]*",
    ]
    
    # Patterns to preserve (headers, tables, etc.)
    PRESERVE_PATTERNS = [
        r"^#{1,6}\s+.+$",  # Markdown headers
        r"^\|.+\|$",       # Table rows
        r"^[-*]\s+.+$",    # List items
        r"^\d+\.\s+.+$",   # Numbered lists
        r"^```[\s\S]*?```$",  # Code blocks
    ]
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.client = OllamaClient(self.config.get('ollama', {}))
        self.use_llm = self.config.get('cleaner', {}).get('use_llm', True)
        
    def clean(self, text: str, use_llm: bool = None) -> CleanedText:
        """
        Clean text using rule-based and optionally LLM-based cleaning.
        
        Args:
            text: Raw OCR text to clean
            use_llm: Whether to use LLM for deep cleaning (overrides config)
            
        Returns:
            CleanedText with original and cleaned versions
        """
        if use_llm is None:
            use_llm = self.use_llm
            
        # Step 1: Rule-based cleaning (fast)
        cleaned, removed = self._rule_based_clean(text)
        
        # Step 2: LLM-based cleaning (optional, more thorough)
        if use_llm and len(cleaned) > 100:
            cleaned = self._llm_clean(cleaned)
            
        # Step 3: Format normalization
        cleaned = self._normalize_format(cleaned)
        
        # Calculate confidence based on how much was changed
        original_len = len(text)
        cleaned_len = len(cleaned)
        confidence = min(cleaned_len / max(original_len, 1), 1.0)
        
        return CleanedText(
            original=text,
            cleaned=cleaned,
            removed_phrases=removed,
            confidence=confidence
        )
    
    def _rule_based_clean(self, text: str) -> tuple[str, List[str]]:
        """Apply rule-based cleaning patterns."""
        cleaned = text
        removed = []
        
        # Remove filler phrases
        for pattern in self.FILLER_PATTERNS:
            matches = re.findall(pattern, cleaned, re.MULTILINE)
            for match in matches:
                if match:
                    removed.append(match.strip())
            cleaned = re.sub(pattern, '', cleaned, flags=re.MULTILINE)
        
        # Clean up excessive whitespace
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        cleaned = re.sub(r' {2,}', ' ', cleaned)
        
        # Remove empty lines at start/end
        cleaned = cleaned.strip()
        
        return cleaned, removed
    
    def _llm_clean(self, text: str) -> str:
        """Use LLM for deep cleaning of conversational filler."""
        prompt = f"""You are a strict text cleaner. The following text was extracted from a document image using OCR.

TASK: Remove ALL conversational filler and return ONLY the document content.

REMOVE these patterns:
- "Here is the text..." / "The image shows..."
- "I can see..." / "Looking at the document..."
- Any meta-commentary about the extraction process
- Polite phrases like "Hope this helps"

PRESERVE:
- All headers (# ## ###)
- All tables (| col | col |)
- All lists (- item or 1. item)
- Technical content, formulas, data

Return ONLY the cleaned document content in proper Markdown format.
Do NOT add any explanation or commentary.

---
TEXT TO CLEAN:
{text[:3000]}
---

CLEANED OUTPUT:"""

        response = self.client.generate(
            prompt=prompt,
            system_prompt="You are a text cleaning utility. Output only cleaned text, nothing else."
        )
        
        if response.success:
            return response.content.strip()
        else:
            logger.warning(f"LLM cleaning failed: {response.error}")
            return text
    
    def _normalize_format(self, text: str) -> str:
        """Normalize Markdown formatting."""
        lines = text.split('\n')
        normalized = []
        
        for line in lines:
            # Ensure headers have space after #
            if line.startswith('#') and not line.startswith('# '):
                line = re.sub(r'^(#+)(\S)', r'\1 \2', line)
            
            # Ensure list items have space after marker
            if re.match(r'^[-*]\S', line):
                line = re.sub(r'^([-*])(\S)', r'\1 \2', line)
            
            normalized.append(line)
        
        return '\n'.join(normalized)
    
    def clean_batch(self, texts: List[str]) -> List[CleanedText]:
        """Clean multiple texts."""
        return [self.clean(text) for text in texts]


class SemanticChunker:
    """
    Chunks text by semantic boundaries (Markdown headers) rather than character count.
    This preserves logical document structure for better training data.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.min_chunk_size = self.config.get('text', {}).get('min_chunk_size', 200)
        self.max_chunk_size = self.config.get('text', {}).get('max_chunk_size', 4000)
        
    def chunk_by_headers(self, text: str) -> List[Dict[str, Any]]:
        """
        Split text by Markdown headers, preserving hierarchy.
        
        Returns list of chunks with metadata:
        - content: The chunk text
        - header: The header that starts this section
        - level: Header level (1-6)
        - index: Chunk index
        """
        chunks = []
        current_chunk = {"content": "", "header": "", "level": 0}
        
        for line in text.split('\n'):
            # Check if this is a header
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            
            if header_match:
                # Save previous chunk if it has content
                if current_chunk["content"].strip():
                    chunks.append(current_chunk.copy())
                
                # Start new chunk
                level = len(header_match.group(1))
                header_text = header_match.group(2)
                current_chunk = {
                    "content": line + "\n",
                    "header": header_text,
                    "level": level
                }
            else:
                current_chunk["content"] += line + "\n"
        
        # Don't forget the last chunk
        if current_chunk["content"].strip():
            chunks.append(current_chunk)
        
        # Post-process: merge tiny chunks, split huge ones
        chunks = self._balance_chunks(chunks)
        
        # Add indices
        for i, chunk in enumerate(chunks):
            chunk["index"] = i
            chunk["word_count"] = len(chunk["content"].split())
        
        return chunks
    
    def _balance_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Merge small chunks and split large ones for balanced sizes."""
        balanced = []
        buffer = {"content": "", "header": "", "level": 0}
        
        for chunk in chunks:
            content_len = len(chunk["content"])
            
            # If chunk is too small, merge with buffer
            if content_len < self.min_chunk_size:
                if buffer["content"]:
                    buffer["content"] += "\n" + chunk["content"]
                else:
                    buffer = chunk.copy()
            # If chunk is too large, split it
            elif content_len > self.max_chunk_size:
                # First, flush buffer
                if buffer["content"]:
                    balanced.append(buffer)
                    buffer = {"content": "", "header": "", "level": 0}
                
                # Split large chunk by paragraphs
                paragraphs = chunk["content"].split('\n\n')
                sub_chunk = {"content": "", "header": chunk["header"], "level": chunk["level"]}
                
                for para in paragraphs:
                    if len(sub_chunk["content"]) + len(para) < self.max_chunk_size:
                        sub_chunk["content"] += para + "\n\n"
                    else:
                        if sub_chunk["content"]:
                            balanced.append(sub_chunk)
                        sub_chunk = {"content": para + "\n\n", "header": "", "level": 0}
                
                if sub_chunk["content"]:
                    balanced.append(sub_chunk)
            else:
                # Flush buffer and add chunk
                if buffer["content"]:
                    balanced.append(buffer)
                    buffer = {"content": "", "header": "", "level": 0}
                balanced.append(chunk)
        
        # Flush remaining buffer
        if buffer["content"]:
            balanced.append(buffer)
        
        return balanced
    
    def chunk_by_paragraphs(self, text: str, target_size: int = 1500) -> List[Dict[str, Any]]:
        """
        Alternative chunking by paragraphs with target size.
        Useful for documents without clear header structure.
        """
        paragraphs = text.split('\n\n')
        chunks = []
        current = {"content": "", "index": 0}
        
        for para in paragraphs:
            if len(current["content"]) + len(para) < target_size:
                current["content"] += para + "\n\n"
            else:
                if current["content"]:
                    current["word_count"] = len(current["content"].split())
                    chunks.append(current)
                current = {"content": para + "\n\n", "index": len(chunks)}
        
        if current["content"]:
            current["word_count"] = len(current["content"].split())
            chunks.append(current)
        
        return chunks
