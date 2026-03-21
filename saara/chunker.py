"""
Text Chunking Module
Intelligently splits documents into chunks suitable for LLM processing.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import re
import logging

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """Represents a chunk of text from a document."""
    chunk_id: int
    text: str
    start_pos: int
    end_pos: int
    page_numbers: List[int] = field(default_factory=list)
    section: Optional[str] = None
    word_count: int = 0
    char_count: int = 0
    
    def __post_init__(self):
        self.word_count = len(self.text.split())
        self.char_count = len(self.text)


class TextChunker:
    """
    Splits text into semantic chunks for LLM processing.
    Uses intelligent boundary detection for natural splits.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        self.chunk_size = config.get('chunk_size', 1500)
        self.chunk_overlap = config.get('chunk_overlap', 200)
        self.min_chunk_size = config.get('min_chunk_size', 100)
        self.max_chunk_size = config.get('max_chunk_size', 3000)
        
        # Sentence ending patterns
        self.sentence_endings = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
        
        # Paragraph patterns
        self.paragraph_pattern = re.compile(r'\n\s*\n')
        
    def chunk_document(self, text: str, sections: List[Dict] = None) -> List[TextChunk]:
        """
        Split document into chunks.
        
        Args:
            text: Full document text
            sections: Optional list of detected sections
            
        Returns:
            List of TextChunk objects
        """
        if not text or len(text.strip()) < self.min_chunk_size:
            return []
            
        # If sections are provided, chunk by section first
        if sections:
            return self._chunk_by_sections(text, sections)
        
        # Otherwise, use semantic chunking
        return self._semantic_chunk(text)
    
    def _semantic_chunk(self, text: str) -> List[TextChunk]:
        """Chunk text using semantic boundaries."""
        chunks = []
        
        # First, split by paragraphs
        paragraphs = self.paragraph_pattern.split(text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        current_chunk = ""
        current_start = 0
        chunk_id = 0
        
        for para in paragraphs:
            # If adding this paragraph would exceed chunk size
            if len(current_chunk) + len(para) > self.chunk_size:
                # Save current chunk if it meets minimum size
                if len(current_chunk) >= self.min_chunk_size:
                    chunks.append(TextChunk(
                        chunk_id=chunk_id,
                        text=current_chunk.strip(),
                        start_pos=current_start,
                        end_pos=current_start + len(current_chunk)
                    ))
                    chunk_id += 1
                    
                    # Start new chunk with overlap
                    overlap_text = self._get_overlap(current_chunk)
                    current_start = current_start + len(current_chunk) - len(overlap_text)
                    current_chunk = overlap_text
                
                # If paragraph itself is too long, split it
                if len(para) > self.max_chunk_size:
                    para_chunks = self._split_long_paragraph(para)
                    for pc in para_chunks:
                        if len(current_chunk) + len(pc) > self.chunk_size:
                            if len(current_chunk) >= self.min_chunk_size:
                                chunks.append(TextChunk(
                                    chunk_id=chunk_id,
                                    text=current_chunk.strip(),
                                    start_pos=current_start,
                                    end_pos=current_start + len(current_chunk)
                                ))
                                chunk_id += 1
                                overlap_text = self._get_overlap(current_chunk)
                                current_start = current_start + len(current_chunk) - len(overlap_text)
                                current_chunk = overlap_text + pc + "\n\n"
                            else:
                                current_chunk += pc + "\n\n"
                        else:
                            current_chunk += pc + "\n\n"
                else:
                    current_chunk += para + "\n\n"
            else:
                current_chunk += para + "\n\n"
        
        # Don't forget the last chunk
        if len(current_chunk.strip()) >= self.min_chunk_size:
            chunks.append(TextChunk(
                chunk_id=chunk_id,
                text=current_chunk.strip(),
                start_pos=current_start,
                end_pos=current_start + len(current_chunk)
            ))
        
        logger.info(f"Created {len(chunks)} chunks from document")
        return chunks
    
    def _chunk_by_sections(self, text: str, sections: List[Dict]) -> List[TextChunk]:
        """Chunk text by detected sections."""
        chunks = []
        chunk_id = 0
        
        # Sort sections by position
        sorted_sections = sorted(sections, key=lambda x: x.get('position', 0))
        
        for i, section in enumerate(sorted_sections):
            start_pos = section.get('position', 0)
            
            # End position is the start of next section or end of text
            if i + 1 < len(sorted_sections):
                end_pos = sorted_sections[i + 1].get('position', len(text))
            else:
                end_pos = len(text)
            
            section_text = text[start_pos:end_pos].strip()
            
            if not section_text:
                continue
                
            # If section is small enough, keep as one chunk
            if len(section_text) <= self.max_chunk_size:
                if len(section_text) >= self.min_chunk_size:
                    chunks.append(TextChunk(
                        chunk_id=chunk_id,
                        text=section_text,
                        start_pos=start_pos,
                        end_pos=end_pos,
                        section=section.get('title')
                    ))
                    chunk_id += 1
            else:
                # Split section into smaller chunks
                section_chunks = self._semantic_chunk(section_text)
                for sc in section_chunks:
                    sc.chunk_id = chunk_id
                    sc.section = section.get('title')
                    sc.start_pos += start_pos
                    sc.end_pos += start_pos
                    chunks.append(sc)
                    chunk_id += 1
        
        # Handle text before first section
        if sorted_sections and sorted_sections[0].get('position', 0) > 0:
            preamble = text[:sorted_sections[0]['position']].strip()
            if len(preamble) >= self.min_chunk_size:
                preamble_chunks = self._semantic_chunk(preamble)
                # Prepend to chunks
                for pc in preamble_chunks:
                    pc.section = "Preamble"
                chunks = preamble_chunks + chunks
                # Renumber chunk IDs
                for i, c in enumerate(chunks):
                    c.chunk_id = i
        
        return chunks
    
    def _split_long_paragraph(self, text: str) -> List[str]:
        """Split a long paragraph by sentences."""
        sentences = self.sentence_endings.split(text)
        
        result = []
        current = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            if len(current) + len(sentence) > self.chunk_size:
                if current:
                    result.append(current.strip())
                current = sentence
            else:
                current += " " + sentence if current else sentence
        
        if current:
            result.append(current.strip())
            
        return result
    
    def _get_overlap(self, text: str) -> str:
        """Get overlap text from end of chunk."""
        if len(text) <= self.chunk_overlap:
            return text
            
        # Try to find a sentence boundary near the overlap point
        overlap_start = len(text) - self.chunk_overlap
        
        # Look for sentence ending in the overlap region
        overlap_text = text[overlap_start:]
        
        # Find first sentence start if possible
        match = re.search(r'(?<=[.!?])\s+', overlap_text)
        if match:
            return overlap_text[match.end():]
            
        return overlap_text


class ChunkProcessor:
    """Additional processing for chunks."""
    
    @staticmethod
    def add_context(chunks: List[TextChunk], document_title: str = None) -> List[TextChunk]:
        """Add context information to chunks."""
        total_chunks = len(chunks)
        
        for chunk in chunks:
            # Add position context
            chunk.metadata = {
                'document_title': document_title,
                'position': f"{chunk.chunk_id + 1}/{total_chunks}",
                'is_first': chunk.chunk_id == 0,
                'is_last': chunk.chunk_id == total_chunks - 1
            }
            
        return chunks
    
    @staticmethod
    def filter_chunks(chunks: List[TextChunk], 
                      min_words: int = 20,
                      max_words: int = 2000) -> List[TextChunk]:
        """Filter chunks by word count."""
        filtered = [
            c for c in chunks 
            if min_words <= c.word_count <= max_words
        ]
        
        # Renumber filtered chunks
        for i, chunk in enumerate(filtered):
            chunk.chunk_id = i
            
        return filtered
