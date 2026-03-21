"""
RAG (Retrieval-Augmented Generation) Engine for SAARA.

🔍 Build intelligent agents that can search and reason over your documents.

Features:
- Multi-format document ingestion (PDF, TXT, JSONL, Markdown)
- Semantic chunking with overlap
- Vector embeddings (local or API-based)
- ChromaDB vector store
- Hybrid search (semantic + keyword)
- Context-aware generation
- Citation tracking
- Multi-agent support

© 2025-2026 Kilani Sai Nikhil. All Rights Reserved.
"""

import json
import logging
import os
import re
import hashlib
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Generator, Tuple, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
import time

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

console = Console()
logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class Document:
    """Represents a document in the knowledge base."""
    doc_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    doc_type: str = "text"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        return cls(**data)


@dataclass
class Chunk:
    """Represents a text chunk from a document."""
    chunk_id: str
    doc_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # Don't serialize embedding to save space in metadata
        data.pop('embedding', None)
        return data


@dataclass
class SearchResult:
    """Result from a search query."""
    chunk: Chunk
    score: float
    doc_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGResponse:
    """Response from RAG query."""
    answer: str
    sources: List[SearchResult]
    query: str
    model: str
    latency_ms: float
    confidence: float = 0.0
    citations: List[str] = field(default_factory=list)


@dataclass
class KnowledgeBaseConfig:
    """Configuration for a knowledge base."""
    name: str
    description: str = ""
    embedding_model: str = "all-MiniLM-L6-v2"  # Sentence transformers model
    chunk_size: int = 512
    chunk_overlap: int = 50
    llm_model: str = "granite4"  # Ollama model for generation
    search_type: str = "hybrid"  # semantic, keyword, or hybrid
    top_k: int = 5
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeBaseConfig":
        return cls(**data)


# ============================================================================
# EMBEDDINGS
# ============================================================================

class EmbeddingProvider:
    """Base class for embedding providers."""
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError
    
    def embed_query(self, query: str) -> List[float]:
        embeddings = self.embed([query])
        return embeddings[0] if embeddings else []


class SentenceTransformerEmbeddings(EmbeddingProvider):
    """Embeddings using sentence-transformers (local, free)."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
    
    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                logger.info(f"Loaded embedding model: {self.model_name}")
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for local embeddings.\n"
                    "Install with: pip install sentence-transformers"
                )
        return self._model
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()
    
    def get_dimension(self) -> int:
        # Get embedding dimension
        return self.model.get_sentence_embedding_dimension()


class OllamaEmbeddings(EmbeddingProvider):
    """Embeddings using Ollama (local, requires Ollama server)."""
    
    def __init__(self, model_name: str = "nomic-embed-text"):
        self.model_name = model_name
        try:
            import ollama
            self.client = ollama.Client()
        except ImportError:
            raise ImportError("ollama is required. Install with: pip install ollama")
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            response = self.client.embeddings(model=self.model_name, prompt=text)
            embeddings.append(response['embedding'])
        return embeddings


# ============================================================================
# VECTOR STORE
# ============================================================================

class VectorStore:
    """ChromaDB-based vector store for document chunks."""
    
    def __init__(self, persist_directory: str, collection_name: str = "default"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self._client = None
        self._collection = None
    
    @property
    def client(self):
        if self._client is None:
            try:
                import chromadb
                from chromadb.config import Settings
                
                self._client = chromadb.PersistentClient(
                    path=self.persist_directory,
                    settings=Settings(anonymized_telemetry=False)
                )
                logger.info(f"Connected to ChromaDB at {self.persist_directory}")
            except ImportError:
                raise ImportError(
                    "chromadb is required for vector storage.\n"
                    "Install with: pip install chromadb"
                )
        return self._client
    
    @property
    def collection(self):
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        return self._collection
    
    def add(self, 
            ids: List[str], 
            embeddings: List[List[float]], 
            documents: List[str],
            metadatas: List[Dict[str, Any]] = None):
        """Add documents to the vector store."""
        # Ensure metadata values are serializable
        if metadatas:
            clean_metadatas = []
            for meta in metadatas:
                clean_meta = {}
                for k, v in meta.items():
                    if isinstance(v, (str, int, float, bool)):
                        clean_meta[k] = v
                    elif isinstance(v, list):
                        clean_meta[k] = json.dumps(v)
                    else:
                        clean_meta[k] = str(v)
                clean_metadatas.append(clean_meta)
            metadatas = clean_metadatas
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
    
    def query(self, 
              query_embedding: List[float], 
              n_results: int = 5,
              where: Dict = None,
              where_document: Dict = None) -> Dict[str, Any]:
        """Query the vector store."""
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            where_document=where_document,
            include=["documents", "metadatas", "distances"]
        )
    
    def delete(self, ids: List[str] = None, where: Dict = None):
        """Delete documents from the store."""
        if ids:
            self.collection.delete(ids=ids)
        elif where:
            self.collection.delete(where=where)
    
    def count(self) -> int:
        """Get number of documents in the store."""
        return self.collection.count()
    
    def get_all(self, limit: int = 100) -> Dict[str, Any]:
        """Get all documents (with limit)."""
        return self.collection.get(limit=limit, include=["documents", "metadatas"])


# ============================================================================
# DOCUMENT PROCESSOR
# ============================================================================

class DocumentProcessor:
    """Process and chunk documents for indexing."""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def process_file(self, file_path: str) -> List[Document]:
        """Process a file and return documents."""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if path.suffix.lower() == '.pdf':
            return self._process_pdf(path)
        elif path.suffix.lower() in ['.txt', '.md']:
            return self._process_text(path)
        elif path.suffix.lower() == '.jsonl':
            return self._process_jsonl(path)
        elif path.suffix.lower() == '.json':
            return self._process_json(path)
        else:
            # Try as text
            return self._process_text(path)
    
    def process_directory(self, dir_path: str, recursive: bool = True) -> List[Document]:
        """Process all files in a directory."""
        path = Path(dir_path)
        documents = []
        
        pattern = "**/*" if recursive else "*"
        
        for file_path in path.glob(pattern):
            if file_path.is_file():
                try:
                    docs = self.process_file(str(file_path))
                    documents.extend(docs)
                except Exception as e:
                    logger.warning(f"Failed to process {file_path}: {e}")
        
        return documents
    
    def _process_pdf(self, path: Path) -> List[Document]:
        """Process a PDF file."""
        try:
            import fitz  # PyMuPDF
            
            doc = fitz.open(str(path))
            text_parts = []
            
            for page_num, page in enumerate(doc):
                text = page.get_text()
                if text.strip():
                    text_parts.append(f"[Page {page_num + 1}]\n{text}")
            
            doc.close()
            
            full_text = "\n\n".join(text_parts)
            
            return [Document(
                doc_id=self._generate_id(str(path)),
                content=full_text,
                source=str(path),
                doc_type="pdf",
                metadata={
                    "filename": path.name,
                    "pages": len(text_parts)
                }
            )]
            
        except ImportError:
            raise ImportError("PyMuPDF is required for PDF processing. Install with: pip install pymupdf")
    
    def _process_text(self, path: Path) -> List[Document]:
        """Process a text or markdown file."""
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        return [Document(
            doc_id=self._generate_id(str(path)),
            content=content,
            source=str(path),
            doc_type="text" if path.suffix.lower() == '.txt' else "markdown",
            metadata={"filename": path.name}
        )]
    
    def _process_jsonl(self, path: Path) -> List[Document]:
        """Process a JSONL file (each line is a document)."""
        documents = []
        
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    data = json.loads(line)
                    
                    # Extract content from common fields
                    content = ""
                    if "text" in data:
                        content = data["text"]
                    elif "content" in data:
                        content = data["content"]
                    elif "conversations" in data:
                        # ShareGPT format
                        parts = []
                        for conv in data["conversations"]:
                            role = conv.get("from", conv.get("role", "user"))
                            value = conv.get("value", conv.get("content", ""))
                            parts.append(f"{role}: {value}")
                        content = "\n".join(parts)
                    else:
                        content = json.dumps(data)
                    
                    if content.strip():
                        documents.append(Document(
                            doc_id=self._generate_id(f"{path}_{i}"),
                            content=content,
                            source=str(path),
                            doc_type="jsonl",
                            metadata={"filename": path.name, "line": i, **data.get("metadata", {})}
                        ))
                except json.JSONDecodeError:
                    continue
        
        return documents
    
    def _process_json(self, path: Path) -> List[Document]:
        """Process a JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            documents = []
            for i, item in enumerate(data):
                content = item.get("text", item.get("content", json.dumps(item)))
                documents.append(Document(
                    doc_id=self._generate_id(f"{path}_{i}"),
                    content=content,
                    source=str(path),
                    doc_type="json",
                    metadata={"filename": path.name, "index": i}
                ))
            return documents
        else:
            content = data.get("text", data.get("content", json.dumps(data)))
            return [Document(
                doc_id=self._generate_id(str(path)),
                content=content,
                source=str(path),
                doc_type="json",
                metadata={"filename": path.name}
            )]
    
    def chunk_document(self, document: Document) -> List[Chunk]:
        """Split a document into chunks."""
        text = document.content
        chunks = []
        
        # Split by paragraphs first
        paragraphs = re.split(r'\n\s*\n', text)
        
        current_chunk = ""
        chunk_idx = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            if len(current_chunk) + len(para) <= self.chunk_size:
                current_chunk += ("\n\n" if current_chunk else "") + para
            else:
                if current_chunk:
                    chunks.append(self._create_chunk(
                        document, current_chunk, chunk_idx
                    ))
                    chunk_idx += 1
                    
                    # Start new chunk with overlap
                    overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else ""
                    current_chunk = overlap_text + ("\n\n" if overlap_text else "") + para
                else:
                    # Paragraph itself is too long - split by sentences
                    sentences = re.split(r'(?<=[.!?])\s+', para)
                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) <= self.chunk_size:
                            current_chunk += (" " if current_chunk else "") + sentence
                        else:
                            if current_chunk:
                                chunks.append(self._create_chunk(
                                    document, current_chunk, chunk_idx
                                ))
                                chunk_idx += 1
                            current_chunk = sentence[:self.chunk_size]
        
        # Don't forget the last chunk
        if current_chunk.strip():
            chunks.append(self._create_chunk(document, current_chunk, chunk_idx))
        
        return chunks
    
    def _create_chunk(self, document: Document, content: str, index: int) -> Chunk:
        """Create a chunk from content."""
        return Chunk(
            chunk_id=f"{document.doc_id}_chunk_{index}",
            doc_id=document.doc_id,
            content=content.strip(),
            metadata={
                "source": document.source,
                "doc_type": document.doc_type,
                "chunk_index": index,
                **document.metadata
            }
        )
    
    def _generate_id(self, content: str) -> str:
        """Generate a unique ID for content."""
        return hashlib.md5(content.encode()).hexdigest()[:16]


# ============================================================================
# RAG ENGINE
# ============================================================================

class RAGEngine:
    """Main RAG engine for knowledge base operations."""
    
    def __init__(self, 
                 knowledge_base_path: str,
                 config: KnowledgeBaseConfig = None):
        self.kb_path = Path(knowledge_base_path)
        self.kb_path.mkdir(parents=True, exist_ok=True)
        
        # Load or create config
        config_path = self.kb_path / "config.json"
        if config_path.exists() and config is None:
            with open(config_path, 'r') as f:
                self.config = KnowledgeBaseConfig.from_dict(json.load(f))
        else:
            self.config = config or KnowledgeBaseConfig(name="default")
            self._save_config()
        
        # Initialize components
        self._embeddings = None
        self._vector_store = None
        self._llm_client = None
        self._processor = None
    
    @property
    def embeddings(self) -> EmbeddingProvider:
        if self._embeddings is None:
            if self.config.embedding_model.startswith("nomic") or self.config.embedding_model.startswith("mxbai"):
                self._embeddings = OllamaEmbeddings(self.config.embedding_model)
            else:
                self._embeddings = SentenceTransformerEmbeddings(self.config.embedding_model)
        return self._embeddings
    
    @property
    def vector_store(self) -> VectorStore:
        if self._vector_store is None:
            self._vector_store = VectorStore(
                persist_directory=str(self.kb_path / "vectors"),
                collection_name=self.config.name.replace(" ", "_").lower()
            )
        return self._vector_store
    
    @property
    def processor(self) -> DocumentProcessor:
        if self._processor is None:
            self._processor = DocumentProcessor(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
        return self._processor
    
    @property
    def llm_client(self):
        if self._llm_client is None:
            from .ollama_client import OllamaClient
            self._llm_client = OllamaClient({"model": self.config.llm_model})
        return self._llm_client
    
    def _save_config(self):
        """Save configuration to disk."""
        with open(self.kb_path / "config.json", 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
    
    def add_documents(self, 
                      source: str, 
                      show_progress: bool = True) -> int:
        """Add documents from a file or directory to the knowledge base."""
        from rich.live import Live
        from rich.table import Table
        from rich.text import Text
        
        source_path = Path(source)
        
        if source_path.is_file():
            documents = self.processor.process_file(str(source_path))
        elif source_path.is_dir():
            documents = self.processor.process_directory(str(source_path))
        else:
            raise ValueError(f"Source not found: {source}")
        
        if not documents:
            return 0
        
        # Chunk all documents
        all_chunks = []
        
        if show_progress:
            console.print(f"\n[cyan]📄 Processing {len(documents)} document(s)...[/cyan]")
        
        for doc in documents:
            chunks = self.processor.chunk_document(doc)
            all_chunks.extend(chunks)
        
        if not all_chunks:
            return 0
        
        if show_progress:
            console.print(f"[green]✓[/green] Created [bold]{len(all_chunks)}[/bold] chunks from documents")
        
        # Create embeddings in batches with colorful progress
        batch_size = 32
        
        if show_progress:
            # Multi-stage colorful progress
            console.print(f"\n[magenta]🧠 Generating embeddings using {self.config.embedding_model}...[/magenta]\n")
            
            with Progress(
                SpinnerColumn(style="cyan"),
                TextColumn("[bold blue]{task.description}[/bold blue]"),
                BarColumn(bar_width=40, style="yellow", complete_style="green", finished_style="bright_green"),
                TextColumn("[bold]{task.percentage:>3.0f}%[/bold]"),
                TextColumn("•"),
                TextColumn("[cyan]{task.completed}/{task.total} chunks[/cyan]"),
                console=console,
                expand=False
            ) as progress:
                # Main indexing task
                task = progress.add_task(
                    "[bold]Indexing[/bold]", 
                    total=len(all_chunks)
                )
                
                indexed_count = 0
                for i in range(0, len(all_chunks), batch_size):
                    batch = all_chunks[i:i+batch_size]
                    self._index_chunks(batch)
                    indexed_count += len(batch)
                    progress.update(task, completed=indexed_count)
            
            # Show completion stats
            console.print(f"\n[green]✅ Indexing complete![/green]")
            
            # Stats table
            stats_table = Table(show_header=False, box=None, padding=(0, 2))
            stats_table.add_column("Metric", style="dim")
            stats_table.add_column("Value", style="bold green")
            stats_table.add_row("Documents processed", str(len(documents)))
            stats_table.add_row("Chunks created", str(len(all_chunks)))
            stats_table.add_row("Embedding model", self.config.embedding_model)
            console.print(stats_table)
        else:
            for i in range(0, len(all_chunks), batch_size):
                batch = all_chunks[i:i+batch_size]
                self._index_chunks(batch)
        
        # Save document metadata
        self._save_document_metadata(documents)
        
        return len(all_chunks)
    
    def _index_chunks(self, chunks: List[Chunk]):
        """Index a batch of chunks."""
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embeddings.embed(texts)
        
        self.vector_store.add(
            ids=[chunk.chunk_id for chunk in chunks],
            embeddings=embeddings,
            documents=texts,
            metadatas=[chunk.metadata for chunk in chunks]
        )
    
    def _save_document_metadata(self, documents: List[Document]):
        """Save document metadata for reference."""
        docs_file = self.kb_path / "documents.jsonl"
        with open(docs_file, 'a', encoding='utf-8') as f:
            for doc in documents:
                f.write(json.dumps(doc.to_dict()) + "\n")
    
    def search(self, 
               query: str, 
               top_k: int = None,
               filters: Dict = None) -> List[SearchResult]:
        """Search the knowledge base."""
        top_k = top_k or self.config.top_k
        
        # Get query embedding
        query_embedding = self.embeddings.embed_query(query)
        
        # Semantic search
        results = self.vector_store.query(
            query_embedding=query_embedding,
            n_results=top_k,
            where=filters
        )
        
        # Convert to SearchResult objects
        search_results = []
        
        if results and results.get('documents'):
            documents = results['documents'][0]
            metadatas = results.get('metadatas', [[]])[0]
            distances = results.get('distances', [[]])[0]
            ids = results.get('ids', [[]])[0]
            
            for i, (doc, meta, dist, chunk_id) in enumerate(zip(documents, metadatas, distances, ids)):
                chunk = Chunk(
                    chunk_id=chunk_id,
                    doc_id=meta.get('doc_id', ''),
                    content=doc,
                    metadata=meta
                )
                
                # Convert distance to similarity score (cosine distance to similarity)
                score = 1 - dist
                
                search_results.append(SearchResult(
                    chunk=chunk,
                    score=score,
                    doc_metadata=meta
                ))
        
        return search_results
    
    def query(self, 
              question: str, 
              top_k: int = None,
              stream: bool = False) -> Union[RAGResponse, Generator[str, None, None]]:
        """Query the knowledge base and generate an answer."""
        start_time = time.time()
        
        # Search for relevant context
        search_results = self.search(question, top_k=top_k)
        
        if not search_results:
            return RAGResponse(
                answer="I couldn't find any relevant information in the knowledge base.",
                sources=[],
                query=question,
                model=self.config.llm_model,
                latency_ms=(time.time() - start_time) * 1000,
                confidence=0.0
            )
        
        # Build context from search results
        context_parts = []
        citations = []
        
        for i, result in enumerate(search_results):
            source = result.doc_metadata.get('source', result.doc_metadata.get('filename', 'Unknown'))
            context_parts.append(f"[Source {i+1}: {source}]\n{result.chunk.content}")
            citations.append(source)
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Build RAG prompt
        system_prompt = """You are a helpful assistant with access to a knowledge base. 
Answer questions based ONLY on the provided context. If the answer isn't in the context, say so.
Be concise but thorough. Cite sources when relevant using [Source N] notation."""
        
        user_prompt = f"""Context from knowledge base:

{context}

---

Question: {question}

Please provide a comprehensive answer based on the context above."""
        
        if stream:
            return self._stream_response(system_prompt, user_prompt, search_results, question, start_time)
        
        # Generate response
        response = self.llm_client.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.3
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Calculate confidence based on search scores
        avg_score = sum(r.score for r in search_results) / len(search_results)
        
        return RAGResponse(
            answer=response.content,
            sources=search_results,
            query=question,
            model=self.config.llm_model,
            latency_ms=latency_ms,
            confidence=avg_score,
            citations=list(set(citations))
        )
    
    def _stream_response(self, system_prompt: str, user_prompt: str, 
                         search_results: List[SearchResult], question: str,
                         start_time: float) -> Generator[str, None, None]:
        """Stream the response."""
        for chunk in self.llm_client.stream_generate(user_prompt, system_prompt):
            yield chunk
    
    def chat(self, 
             conversation: List[Dict[str, str]],
             top_k: int = None) -> RAGResponse:
        """Chat with the knowledge base, maintaining conversation history."""
        # Get the latest user message
        user_message = ""
        for msg in reversed(conversation):
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                break
        
        if not user_message:
            return RAGResponse(
                answer="Please ask a question.",
                sources=[],
                query="",
                model=self.config.llm_model,
                latency_ms=0
            )
        
        # Search for relevant context
        search_results = self.search(user_message, top_k=top_k)
        
        # Build context
        context_parts = []
        for i, result in enumerate(search_results):
            source = result.doc_metadata.get('source', 'Unknown')
            context_parts.append(f"[Source {i+1}]: {result.chunk.content}")
        
        context = "\n\n".join(context_parts)
        
        # Build messages with context
        system_prompt = f"""You are a helpful assistant with access to a knowledge base.

Relevant context from knowledge base:
{context}

Answer based on the context when relevant. Be conversational and helpful."""
        
        # Generate with conversation history
        messages_text = ""
        for msg in conversation:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            messages_text += f"{role.capitalize()}: {content}\n"
        
        response = self.llm_client.generate(
            prompt=messages_text,
            system_prompt=system_prompt,
            temperature=0.5
        )
        
        return RAGResponse(
            answer=response.content,
            sources=search_results,
            query=user_message,
            model=self.config.llm_model,
            latency_ms=0
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        chunk_count = self.vector_store.count()
        
        # Count documents
        docs_file = self.kb_path / "documents.jsonl"
        doc_count = 0
        if docs_file.exists():
            with open(docs_file, 'r') as f:
                doc_count = sum(1 for _ in f)
        
        return {
            "name": self.config.name,
            "description": self.config.description,
            "documents": doc_count,
            "chunks": chunk_count,
            "embedding_model": self.config.embedding_model,
            "llm_model": self.config.llm_model,
            "chunk_size": self.config.chunk_size,
            "created_at": self.config.created_at
        }
    
    def clear(self):
        """Clear all documents from the knowledge base."""
        # Delete vector store
        vectors_path = self.kb_path / "vectors"
        if vectors_path.exists():
            shutil.rmtree(vectors_path)
        
        # Clear documents file
        docs_file = self.kb_path / "documents.jsonl"
        if docs_file.exists():
            docs_file.unlink()
        
        # Reset vector store
        self._vector_store = None
        
        logger.info(f"Cleared knowledge base: {self.config.name}")
    
    def delete(self):
        """Delete the entire knowledge base."""
        if self.kb_path.exists():
            shutil.rmtree(self.kb_path)
        logger.info(f"Deleted knowledge base: {self.config.name}")


# ============================================================================
# RAG MANAGER
# ============================================================================

class RAGManager:
    """Manages multiple RAG knowledge bases."""
    
    DEFAULT_BASE_PATH = "knowledge_bases"
    
    def __init__(self, base_path: str = None):
        self.base_path = Path(base_path or self.DEFAULT_BASE_PATH)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def create(self, 
               name: str, 
               description: str = "",
               **config_kwargs) -> RAGEngine:
        """Create a new knowledge base."""
        kb_path = self.base_path / name.replace(" ", "_").lower()
        
        if kb_path.exists():
            raise ValueError(f"Knowledge base '{name}' already exists")
        
        config = KnowledgeBaseConfig(
            name=name,
            description=description,
            **config_kwargs
        )
        
        engine = RAGEngine(str(kb_path), config)
        logger.info(f"Created knowledge base: {name}")
        
        return engine
    
    def get(self, name: str) -> RAGEngine:
        """Get an existing knowledge base."""
        kb_path = self.base_path / name.replace(" ", "_").lower()
        
        if not kb_path.exists():
            raise ValueError(f"Knowledge base '{name}' not found")
        
        return RAGEngine(str(kb_path))
    
    def list(self) -> List[Dict[str, Any]]:
        """List all knowledge bases."""
        kbs = []
        
        for path in self.base_path.iterdir():
            if path.is_dir():
                config_path = path / "config.json"
                if config_path.exists():
                    try:
                        with open(config_path, 'r') as f:
                            config = json.load(f)
                        
                        # Get stats
                        engine = RAGEngine(str(path))
                        stats = engine.get_stats()
                        kbs.append(stats)
                    except Exception as e:
                        logger.warning(f"Failed to load KB at {path}: {e}")
        
        return kbs
    
    def delete(self, name: str):
        """Delete a knowledge base."""
        kb_path = self.base_path / name.replace(" ", "_").lower()
        
        if not kb_path.exists():
            raise ValueError(f"Knowledge base '{name}' not found")
        
        engine = RAGEngine(str(kb_path))
        engine.delete()
        
        logger.info(f"Deleted knowledge base: {name}")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_rag_engine(name: str = "default", **kwargs) -> RAGEngine:
    """Create a RAG engine with sensible defaults."""
    manager = RAGManager()
    
    try:
        # Try to get existing
        return manager.get(name)
    except ValueError:
        # Create new
        return manager.create(name, **kwargs)


def quick_rag(documents: Union[str, List[str]], 
              query: str,
              kb_name: str = "quick_rag") -> RAGResponse:
    """Quick one-shot RAG query on documents."""
    engine = create_rag_engine(kb_name)
    
    # Add documents if they're files
    if isinstance(documents, str):
        documents = [documents]
    
    for doc in documents:
        if Path(doc).exists():
            engine.add_documents(doc, show_progress=False)
    
    # Query
    return engine.query(query)
