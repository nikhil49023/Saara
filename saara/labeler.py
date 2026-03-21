"""
Data Labeler Module
Uses Granite 4.0 to automatically label and categorize document content.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from .ollama_client import OllamaClient, PromptTemplates, LLMResponse
from .chunker import TextChunk
from .pdf_extractor import ExtractedDocument

logger = logging.getLogger(__name__)


@dataclass
class LabeledChunk:
    """A text chunk with labels and metadata."""
    chunk: TextChunk
    labels: Dict[str, Any] = field(default_factory=dict)
    qa_pairs: List[Dict[str, str]] = field(default_factory=list)
    summary: str = ""
    entities: List[Dict[str, str]] = field(default_factory=list)
    instruction_pairs: List[Dict[str, str]] = field(default_factory=list)
    quality_score: float = 0.0
    is_suitable: bool = True
    

@dataclass
class LabeledDocument:
    """A complete labeled document."""
    source_path: str
    title: str
    category: str
    confidence: float
    topics: Dict[str, Any]
    chunks: List[LabeledChunk]
    metadata: Dict[str, Any] = field(default_factory=dict)
    total_qa_pairs: int = 0
    total_instruction_pairs: int = 0


class DataLabeler:
    """
    Automated data labeling using Granite 4.0.
    Processes documents and generates labeled training data.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        ollama_config = self.config.get('ollama', {})
        self.client = OllamaClient(ollama_config)
        
        self.labeling_config = self.config.get('labeling', {})
        self.dataset_types = self.config.get('dataset_types', [])
        
        # What to generate
        self.generate_qa = self._is_enabled('instruction_tuning')
        self.generate_summaries = self._is_enabled('summarization')
        self.generate_instructions = self._is_enabled('instruction_tuning')
        self.generate_classification = self._is_enabled('classification')
        self.extract_entities = self.labeling_config.get('extract_entities', True)
        
    def _is_enabled(self, dataset_type: str) -> bool:
        """Check if a dataset type is enabled."""
        for dt in self.dataset_types:
            if dt.get('name') == dataset_type:
                return dt.get('enabled', True)
        return True
    
    def label_document(self, 
                       document: ExtractedDocument,
                       chunks: List[TextChunk],
                       progress_callback=None) -> LabeledDocument:
        """
        Label an entire document with all its chunks.
        
        Args:
            document: Extracted document
            chunks: Text chunks from the document
            progress_callback: Optional callback for progress updates
            
        Returns:
            LabeledDocument with all labels and generated data
        """
        logger.info(f"Labeling document: {document.metadata.title}")
        
        # First, classify the document type
        classification = self._classify_document(document.full_text[:3000])
        
        # Extract topics from the document
        topics = self._extract_topics(document.full_text[:3000])
        
        # Label each chunk
        labeled_chunks = []
        total_qa = 0
        total_instructions = 0
        
        for i, chunk in enumerate(tqdm(chunks, desc="Labeling chunks")):
            if progress_callback:
                progress_callback(i + 1, len(chunks))
                
            labeled = self._label_chunk(chunk)
            labeled_chunks.append(labeled)
            
            total_qa += len(labeled.qa_pairs)
            total_instructions += len(labeled.instruction_pairs)
        
        return LabeledDocument(
            source_path=document.metadata.file_path,
            title=document.metadata.title or "Untitled",
            category=classification.get('category', 'general_knowledge'),
            confidence=classification.get('confidence', 0.5),
            topics=topics,
            chunks=labeled_chunks,
            metadata={
                'author': document.metadata.author,
                'page_count': document.metadata.page_count,
                'word_count': sum(c.word_count for c in chunks),
                'chunk_count': len(chunks)
            },
            total_qa_pairs=total_qa,
            total_instruction_pairs=total_instructions
        )
    
    def _classify_document(self, text: str) -> Dict[str, Any]:
        """Classify document type."""
        prompt = PromptTemplates.CLASSIFY_DOCUMENT.format(text=text[:2000])
        
        result = self.client.generate_json(
            prompt=prompt,
            system_prompt="You are a document classification expert."
        )
        
        if 'error' in result:
            logger.warning(f"Classification failed: {result['error']}")
            return {'category': 'general_knowledge', 'confidence': 0.5}
            
        return result
    
    def _extract_topics(self, text: str) -> Dict[str, Any]:
        """Extract topics and keywords."""
        prompt = PromptTemplates.EXTRACT_TOPICS.format(text=text[:2000])
        
        result = self.client.generate_json(
            prompt=prompt,
            system_prompt="You are an expert at identifying topics and themes in documents."
        )
        
        if 'error' in result:
            logger.warning(f"Topic extraction failed: {result['error']}")
            return {'main_topic': 'unknown', 'subtopics': [], 'keywords': []}
            
        return result
    
    def _label_chunk(self, chunk: TextChunk) -> LabeledChunk:
        """Label a single chunk with all configured labeling tasks."""
        labeled = LabeledChunk(chunk=chunk)
        
        # Assess quality first
        quality = self._assess_quality(chunk.text)
        labeled.quality_score = quality.get('quality_score', 5)
        labeled.is_suitable = quality.get('is_suitable', True)
        labeled.labels['quality'] = quality
        
        # Skip low-quality chunks
        if labeled.quality_score < 4 or not labeled.is_suitable:
            logger.debug(f"Skipping low-quality chunk {chunk.chunk_id}")
            return labeled
        
        # Generate Q&A pairs
        if self.generate_qa:
            # High density: 30 pairs per chunk (~120 per page)
            qa_pairs = self._generate_qa_pairs(chunk.text, num_pairs=30)
            labeled.qa_pairs = qa_pairs
        
        # Generate summary
        if self.generate_summaries:
            summary = self._generate_summary(chunk.text)
            labeled.summary = summary.get('summary', '')
            labeled.labels['summary_data'] = summary
        
        # Extract entities
        if self.extract_entities:
            entities = self._extract_entities(chunk.text)
            labeled.entities = entities.get('entities', [])
        
        # Generate instruction pairs
        if self.generate_instructions:
            instructions = self._generate_instructions(chunk.text)
            labeled.instruction_pairs = instructions
        
        return labeled
    
    def _assess_quality(self, text: str) -> Dict[str, Any]:
        """Assess text quality for training."""
        prompt = PromptTemplates.ASSESS_QUALITY.format(text=text[:1500])
        
        result = self.client.generate_json(
            prompt=prompt,
            system_prompt="You are a data quality assessor for ML training data."
        )
        
        if 'error' in result:
            return {'quality_score': 5, 'is_suitable': True}
            
        return result
    
    def _generate_qa_pairs(self, text: str, num_pairs: int = 15) -> List[Dict[str, str]]:
        """Generate question-answer pairs from text."""
        # Note: num_pairs is used in the prompt but default is now lower per chunk 
        # (since we are "forcing" 8+ per small chunk)
        
        prompt = PromptTemplates.GENERATE_QA.format(text=text)
        
        result = self.client.generate_json(
            prompt=prompt,
            system_prompt="You are a rigorous medical data analyst. Output JSON list only."
        )
        
        if 'error' in result:
            return []
            
        # Result should be a list directly
        if isinstance(result, list):
            return result
        elif isinstance(result, dict) and 'qa_pairs' in result:
            return result['qa_pairs']
            
        return []
    
    def _generate_summary(self, text: str) -> Dict[str, Any]:
        """Generate a summary of the text."""
        prompt = PromptTemplates.SUMMARIZE.format(text=text)
        
        result = self.client.generate_json(
            prompt=prompt,
            system_prompt="You are an expert summarizer."
        )
        
        if 'error' in result:
            return {'summary': '', 'key_points': []}
            
        return result
    
    def _extract_entities(self, text: str) -> Dict[str, Any]:
        """Extract named entities from text."""
        prompt = PromptTemplates.EXTRACT_ENTITIES.format(text=text[:1500])
        
        result = self.client.generate_json(
            prompt=prompt,
            system_prompt="You are a named entity recognition expert."
        )
        
        if 'error' in result:
            return {'entities': []}
            
        return result
    
    def _generate_instructions(self, text: str, num_pairs: int = 2) -> List[Dict[str, str]]:
        """Generate instruction-response pairs."""
        instructions = []
        
        for _ in range(num_pairs):
            prompt = PromptTemplates.CREATE_INSTRUCTION.format(text=text)
            
            result = self.client.generate_json(
                prompt=prompt,
                system_prompt="You are an expert at creating instruction-following training data."
            )
            
            if 'error' not in result:
                instructions.append({
                    'instruction': result.get('instruction', ''),
                    'response': result.get('response', ''),
                    'category': result.get('category', 'general')
                })
        
        return instructions


class BatchLabeler:
    """Process multiple documents in batch."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.labeler = DataLabeler(config)
        self.max_workers = config.get('max_workers', 2)  # Conservative for LLM calls
        
    def label_documents(self, 
                        documents: List[tuple],  # (ExtractedDocument, List[TextChunk])
                        progress_callback=None) -> List[LabeledDocument]:
        """
        Label multiple documents.
        
        Args:
            documents: List of (document, chunks) tuples
            progress_callback: Optional callback
            
        Returns:
            List of LabeledDocument objects
        """
        labeled_docs = []
        
        for i, (doc, chunks) in enumerate(documents):
            logger.info(f"Processing document {i+1}/{len(documents)}: {doc.metadata.title}")
            
            if progress_callback:
                progress_callback(i + 1, len(documents), doc.metadata.title)
            
            labeled = self.labeler.label_document(doc, chunks)
            labeled_docs.append(labeled)
        
        return labeled_docs
