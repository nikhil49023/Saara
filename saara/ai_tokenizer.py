"""
AI-Enhanced Tokenizer Module
Uses LLMs to create intelligent, domain-aware tokenizers.

Features:
- Domain-specific vocabulary extraction using AI
- Semantic-aware subword segmentation
- Multi-lingual support with AI language detection
- Smart handling of technical terms, formulas, code
- Vocabulary optimization based on semantic importance

© 2025-2026 Kilani Sai Nikhil. All Rights Reserved.
"""

import os
import sys
import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Set
from dataclasses import dataclass, field
from collections import Counter
import hashlib

try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich.panel import Panel
    from rich.table import Table
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

logger = logging.getLogger(__name__)
console = Console() if RICH_AVAILABLE else None


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class AITokenizerConfig:
    """Configuration for AI-enhanced tokenizer."""
    vocab_size: int = 32000
    min_frequency: int = 2
    special_tokens: List[str] = field(default_factory=lambda: [
        "<pad>", "<unk>", "<bos>", "<eos>", "<sep>", "<mask>", "<cls>"
    ])
    
    # AI settings
    use_ai_extraction: bool = True
    use_ai_segmentation: bool = True
    domain: str = "general"  # general, medical, legal, code, scientific
    languages: List[str] = field(default_factory=lambda: ["en"])
    
    # Provider settings
    provider: str = "auto"  # auto, ollama, gemini, openai
    model: str = None
    
    # Processing
    max_samples_for_vocab: int = 10000
    batch_size: int = 100
    cache_dir: str = "tokenizers/.cache"


# ============================================================================
# AI Domain Analyzer
# ============================================================================

class AIDomainAnalyzer:
    """Uses AI to analyze text domain and extract key terms."""
    
    DOMAIN_PROMPTS = {
        "general": """Analyze this text and extract important terms that should be kept as single tokens.
Focus on: proper nouns, technical terms, compound words, acronyms, and domain-specific vocabulary.

Text:
{text}

Return a JSON object with:
{{
    "domain": "detected domain (general/medical/legal/code/scientific)",
    "language": "detected primary language code",
    "key_terms": ["list of important terms to keep as single tokens"],
    "compound_words": ["list of compound words that shouldn't be split"],
    "abbreviations": {{"abbr": "full form"}},
    "special_patterns": ["regex patterns for domain-specific tokens"]
}}

JSON:""",

        "medical": """You are a medical terminology expert. Analyze this medical text and extract important terms.

Text:
{text}

Return JSON with:
{{
    "medical_terms": ["disease names, drug names, procedures, anatomical terms"],
    "drug_names": ["specific drug/medicine names"],
    "abbreviations": {{"abbr": "full medical term"}},
    "compound_terms": ["multi-word medical terms that should stay together"],
    "dosage_patterns": ["patterns for dosages like '500mg', '2x daily'"]
}}

JSON:""",

        "code": """Analyze this code/technical text and identify tokens that should not be split.

Text:
{text}

Return JSON with:
{{
    "identifiers": ["function names, variable names, class names"],
    "keywords": ["programming language keywords"],
    "operators": ["operators and symbols to keep together"],
    "string_patterns": ["patterns for strings, comments"],
    "special_tokens": ["tokens unique to this codebase"]
}}

JSON:""",

        "scientific": """Analyze this scientific text and extract domain-specific terminology.

Text:
{text}

Return JSON with:
{{
    "scientific_terms": ["technical terms, formulas, equations"],
    "units": ["measurement units"],
    "chemical_formulas": ["H2O, CO2, etc."],
    "mathematical_notation": ["symbols and notation to preserve"],
    "citations": ["citation patterns"]
}}

JSON:""",

        "legal": """Analyze this legal text and extract legal terminology.

Text:
{text}

Return JSON with:
{{
    "legal_terms": ["legal terminology, Latin phrases"],
    "case_citations": ["case citation patterns"],
    "statute_references": ["statute/law references"],
    "defined_terms": ["terms with specific legal definitions"],
    "latin_phrases": ["Latin legal phrases to keep intact"]
}}

JSON:"""
    }
    
    def __init__(self, config: AITokenizerConfig):
        self.config = config
        self._client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the AI client based on config."""
        provider = self.config.provider
        
        if provider == "auto":
            # Try cloud first, then ollama
            try:
                from saara.cloud_runtime import UnifiedCloudClient
                client = UnifiedCloudClient()
                if client.is_available():
                    self._client = client
                    self._provider = "cloud"
                    return
            except:
                pass
            
            # Try Ollama
            try:
                import requests
                response = requests.get("http://localhost:11434/api/tags", timeout=2)
                if response.ok:
                    self._provider = "ollama"
                    return
            except:
                pass
            
            logger.warning("No AI provider available. Will use rule-based extraction.")
            self._provider = "none"
        else:
            self._provider = provider
    
    def _call_ai(self, prompt: str) -> str:
        """Call AI to get response."""
        if self._provider == "cloud" and self._client:
            return self._client.generate(prompt, temperature=0.3, max_tokens=2048)
        
        elif self._provider == "ollama":
            import requests
            model = self.config.model or "granite3.1-dense:8b"
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3}
                },
                timeout=120
            )
            
            if response.ok:
                return response.json().get("response", "")
        
        return ""
    
    def analyze_domain(self, texts: List[str]) -> Dict[str, Any]:
        """Analyze texts to extract domain-specific information."""
        if self._provider == "none":
            return self._rule_based_analysis(texts)
        
        # Sample texts for analysis
        sample_size = min(10, len(texts))
        samples = texts[:sample_size] if len(texts) <= sample_size else \
                  [texts[i * len(texts) // sample_size] for i in range(sample_size)]
        
        combined_text = "\n---\n".join(samples[:5])[:4000]  # Limit context
        
        prompt = self.DOMAIN_PROMPTS.get(
            self.config.domain, 
            self.DOMAIN_PROMPTS["general"]
        ).format(text=combined_text)
        
        try:
            response = self._call_ai(prompt)
            
            # Parse JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                result = json.loads(json_match.group())
                return result
        except Exception as e:
            logger.warning(f"AI analysis failed: {e}")
        
        return self._rule_based_analysis(texts)
    
    def _rule_based_analysis(self, texts: List[str]) -> Dict[str, Any]:
        """Fallback rule-based analysis."""
        combined = " ".join(texts[:100])
        
        # Extract potential terms
        words = re.findall(r'\b[A-Za-z][A-Za-z0-9_-]*[A-Za-z0-9]\b', combined)
        word_counts = Counter(words)
        
        # Find capitalized terms (likely proper nouns)
        proper_nouns = [w for w, c in word_counts.items() 
                       if w[0].isupper() and c >= 2]
        
        # Find technical terms (camelCase, snake_case)
        technical = [w for w in word_counts if '_' in w or 
                    (any(c.isupper() for c in w[1:]))]
        
        # Find abbreviations (all caps)
        abbreviations = [w for w, c in word_counts.items() 
                        if w.isupper() and len(w) >= 2 and c >= 2]
        
        return {
            "domain": self.config.domain,
            "key_terms": proper_nouns[:50],
            "compound_words": technical[:30],
            "abbreviations": {a: a for a in abbreviations[:20]},
            "special_patterns": []
        }
    
    def suggest_segmentation(self, word: str) -> List[str]:
        """Use AI to suggest how to segment a complex word."""
        if self._provider == "none" or len(word) < 6:
            return [word]
        
        prompt = f"""How should this word be segmented into meaningful subwords for a tokenizer?
Word: {word}

Return only the segmentation as space-separated parts, e.g.: "un reason able"
Segmentation:"""
        
        try:
            response = self._call_ai(prompt)
            parts = response.strip().split()
            if parts and ''.join(parts).lower() == word.lower():
                return parts
        except:
            pass
        
        return [word]


# ============================================================================
# AI-Enhanced BPE Tokenizer
# ============================================================================

class AIEnhancedTokenizer:
    """
    BPE Tokenizer enhanced with AI-driven vocabulary optimization.
    
    Features:
    - AI-extracted domain vocabulary
    - Semantic-aware merge priorities
    - Protected tokens for domain terms
    - Smart handling of special patterns
    """
    
    def __init__(self, config: AITokenizerConfig = None):
        self.config = config or AITokenizerConfig()
        self.vocab: Dict[str, int] = {}
        self.merges: List[Tuple[str, str]] = []
        self.special_tokens: Dict[str, int] = {}
        self.protected_tokens: Set[str] = set()
        self.domain_info: Dict[str, Any] = {}
        
        # Initialize special tokens
        for i, token in enumerate(self.config.special_tokens):
            self.special_tokens[token] = i
        
        self.analyzer = AIDomainAnalyzer(self.config)
        
        # Cache directory
        self.cache_dir = Path(self.config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_key(self, texts: List[str]) -> str:
        """Generate cache key for texts."""
        content = "".join(texts[:100])
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def train(self, texts: List[str], show_progress: bool = True) -> None:
        """
        Train the tokenizer on a corpus of texts.
        
        Args:
            texts: List of text strings to train on
            show_progress: Whether to show progress bar
        """
        if RICH_AVAILABLE and show_progress:
            console.print(Panel(
                f"[bold cyan]🤖 AI-Enhanced Tokenizer Training[/bold cyan]\n\n"
                f"Vocab Size: {self.config.vocab_size}\n"
                f"Domain: {self.config.domain}\n"
                f"AI Extraction: {'✅' if self.config.use_ai_extraction else '❌'}\n"
                f"Texts: {len(texts)}",
                title="Tokenizer Training",
                border_style="cyan"
            ))
        
        # Step 1: AI Domain Analysis
        if self.config.use_ai_extraction:
            if show_progress:
                console.print("\n[bold]Step 1/4:[/bold] AI Domain Analysis...")
            
            self.domain_info = self.analyzer.analyze_domain(texts)
            
            # Extract protected tokens from AI analysis
            self._extract_protected_tokens()
            
            if show_progress and self.protected_tokens:
                console.print(f"  [green]✓ Found {len(self.protected_tokens)} protected tokens[/green]")
        
        # Step 2: Build initial vocabulary
        if show_progress:
            console.print("\n[bold]Step 2/4:[/bold] Building character vocabulary...")
        
        word_freqs = self._get_word_frequencies(texts, show_progress)
        
        if show_progress:
            console.print(f"  [green]✓ Found {len(word_freqs)} unique words[/green]")
        
        # Step 3: Initialize vocab with characters
        self._initialize_vocab(word_freqs)
        
        if show_progress:
            console.print(f"  [green]✓ Initial vocab: {len(self.vocab)} characters[/green]")
        
        # Step 4: BPE Training with AI-guided merges
        if show_progress:
            console.print("\n[bold]Step 3/4:[/bold] BPE Training...")
        
        self._train_bpe(word_freqs, show_progress)
        
        if show_progress:
            console.print(f"  [green]✓ Final vocab: {len(self.vocab)} tokens[/green]")
            console.print(f"  [green]✓ Learned {len(self.merges)} merges[/green]")
        
        # Step 5: Add special tokens
        if show_progress:
            console.print("\n[bold]Step 4/4:[/bold] Adding special tokens...")
        
        self._add_special_tokens()
        
        if show_progress:
            console.print(f"  [green]✓ Added {len(self.special_tokens)} special tokens[/green]")
            console.print(Panel(
                f"[green]✅ Training Complete![/green]\n\n"
                f"Total Vocabulary: {len(self.vocab) + len(self.special_tokens)}\n"
                f"Protected Terms: {len(self.protected_tokens)}\n"
                f"BPE Merges: {len(self.merges)}",
                title="Training Summary",
                border_style="green"
            ))
    
    def _extract_protected_tokens(self):
        """Extract tokens that should not be split from AI analysis."""
        info = self.domain_info
        
        # Add key terms
        for term in info.get("key_terms", []):
            self.protected_tokens.add(term.lower())
        
        # Add compound words
        for term in info.get("compound_words", []):
            self.protected_tokens.add(term.lower())
        
        # Add medical/scientific terms
        for key in ["medical_terms", "drug_names", "scientific_terms", 
                    "legal_terms", "identifiers"]:
            for term in info.get(key, []):
                self.protected_tokens.add(term.lower())
        
        # Add abbreviations
        for abbr in info.get("abbreviations", {}).keys():
            self.protected_tokens.add(abbr.lower())
    
    def _get_word_frequencies(self, texts: List[str], show_progress: bool = True) -> Dict[str, int]:
        """Get word frequencies from texts with protected token handling."""
        word_freq = Counter()
        
        iterator = texts[:self.config.max_samples_for_vocab]
        if show_progress and RICH_AVAILABLE:
            iterator = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
            ).track(iterator, description="Counting words...")
        
        for text in iterator:
            # Tokenize while preserving protected tokens
            words = self._pretokenize(text)
            word_freq.update(words)
        
        return dict(word_freq)
    
    def _pretokenize(self, text: str) -> List[str]:
        """Pre-tokenize text, preserving protected tokens."""
        # First, protect special patterns
        protected_map = {}
        
        for i, token in enumerate(self.protected_tokens):
            if token in text.lower():
                placeholder = f"__PROT_{i}__"
                # Case-insensitive replacement
                pattern = re.compile(re.escape(token), re.IGNORECASE)
                for match in pattern.finditer(text):
                    original = match.group()
                    protected_map[placeholder] = original
                    text = text[:match.start()] + placeholder + text[match.end():]
        
        # Standard word tokenization
        words = re.findall(r'\S+', text)
        
        # Restore protected tokens
        result = []
        for word in words:
            if word in protected_map:
                result.append(protected_map[word])
            else:
                result.append(word)
        
        return result
    
    def _initialize_vocab(self, word_freqs: Dict[str, int]):
        """Initialize vocabulary with characters."""
        # Collect all characters
        chars = set()
        for word in word_freqs:
            chars.update(word)
        
        # Add to vocab
        offset = len(self.special_tokens)
        for i, char in enumerate(sorted(chars)):
            self.vocab[char] = offset + i
    
    def _train_bpe(self, word_freqs: Dict[str, int], show_progress: bool = True):
        """Train BPE with AI-guided merge priorities."""
        # Convert words to character sequences
        splits = {word: list(word) for word in word_freqs}
        
        target_vocab_size = self.config.vocab_size - len(self.special_tokens)
        current_vocab_size = len(self.vocab)
        
        if show_progress and RICH_AVAILABLE:
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
            )
            task = progress.add_task(
                "Learning merges...", 
                total=target_vocab_size - current_vocab_size
            )
            progress.start()
        
        while current_vocab_size < target_vocab_size:
            # Count pair frequencies
            pair_freqs = Counter()
            for word, chars in splits.items():
                freq = word_freqs[word]
                for i in range(len(chars) - 1):
                    pair = (chars[i], chars[i + 1])
                    pair_freqs[pair] += freq
            
            if not pair_freqs:
                break
            
            # Find best pair (with AI-guided priority for protected substrings)
            best_pair = self._select_best_pair(pair_freqs)
            
            if best_pair is None:
                break
            
            # Merge best pair
            new_token = best_pair[0] + best_pair[1]
            self.merges.append(best_pair)
            self.vocab[new_token] = len(self.vocab) + len(self.special_tokens)
            
            # Update splits
            for word in splits:
                chars = splits[word]
                new_chars = []
                i = 0
                while i < len(chars):
                    if i < len(chars) - 1 and chars[i] == best_pair[0] and chars[i + 1] == best_pair[1]:
                        new_chars.append(new_token)
                        i += 2
                    else:
                        new_chars.append(chars[i])
                        i += 1
                splits[word] = new_chars
            
            current_vocab_size += 1
            
            if show_progress and RICH_AVAILABLE:
                progress.advance(task)
        
        if show_progress and RICH_AVAILABLE:
            progress.stop()
    
    def _select_best_pair(self, pair_freqs: Counter) -> Optional[Tuple[str, str]]:
        """Select best pair with AI-guided priorities."""
        if not pair_freqs:
            return None
        
        # Score pairs
        scored_pairs = []
        
        for pair, freq in pair_freqs.most_common(100):
            merged = pair[0] + pair[1]
            score = freq
            
            # Boost score for pairs that form protected tokens
            if merged.lower() in self.protected_tokens:
                score *= 10
            
            # Boost for pairs that are substrings of protected tokens
            for protected in self.protected_tokens:
                if merged.lower() in protected:
                    score *= 1.5
                    break
            
            scored_pairs.append((pair, score))
        
        if not scored_pairs:
            return None
        
        # Return highest scored pair
        scored_pairs.sort(key=lambda x: x[1], reverse=True)
        return scored_pairs[0][0]
    
    def _add_special_tokens(self):
        """Add special tokens to vocabulary."""
        # Special tokens are already in self.special_tokens
        # Just ensure they have the lowest IDs
        pass
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        words = self._pretokenize(text)
        tokens = []
        
        for word in words:
            # Check if word is a protected token
            if word.lower() in self.protected_tokens and word in self.vocab:
                tokens.append(self.vocab[word])
                continue
            
            # Apply BPE
            word_tokens = list(word)
            
            for merge in self.merges:
                i = 0
                new_tokens = []
                while i < len(word_tokens):
                    if i < len(word_tokens) - 1 and \
                       word_tokens[i] == merge[0] and word_tokens[i + 1] == merge[1]:
                        new_tokens.append(merge[0] + merge[1])
                        i += 2
                    else:
                        new_tokens.append(word_tokens[i])
                        i += 1
                word_tokens = new_tokens
            
            # Convert to IDs
            for token in word_tokens:
                if token in self.vocab:
                    tokens.append(self.vocab[token])
                elif token in self.special_tokens:
                    tokens.append(self.special_tokens[token])
                else:
                    tokens.append(self.special_tokens.get("<unk>", 1))
        
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        # Build reverse vocab
        id_to_token = {v: k for k, v in self.vocab.items()}
        id_to_token.update({v: k for k, v in self.special_tokens.items()})
        
        tokens = [id_to_token.get(tid, "<unk>") for tid in token_ids]
        
        # Join tokens (simple space joining for now)
        text = "".join(tokens)
        return text
    
    def save(self, directory: str) -> None:
        """Save tokenizer to directory."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config_dict = {
            "vocab_size": self.config.vocab_size,
            "min_frequency": self.config.min_frequency,
            "special_tokens": self.config.special_tokens,
            "domain": self.config.domain,
            "languages": self.config.languages,
            "use_ai_extraction": self.config.use_ai_extraction,
        }
        
        with open(path / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)
        
        # Save vocab
        with open(path / "vocab.json", "w") as f:
            json.dump(self.vocab, f, indent=2)
        
        # Save special tokens
        with open(path / "special_tokens.json", "w") as f:
            json.dump(self.special_tokens, f, indent=2)
        
        # Save merges
        with open(path / "merges.txt", "w") as f:
            for merge in self.merges:
                f.write(f"{merge[0]} {merge[1]}\n")
        
        # Save protected tokens
        with open(path / "protected_tokens.json", "w") as f:
            json.dump(list(self.protected_tokens), f, indent=2)
        
        # Save domain info
        with open(path / "domain_info.json", "w") as f:
            json.dump(self.domain_info, f, indent=2)
        
        if RICH_AVAILABLE:
            console.print(f"[green]✓ Tokenizer saved to {directory}[/green]")
    
    @classmethod
    def load(cls, directory: str) -> "AIEnhancedTokenizer":
        """Load tokenizer from directory."""
        path = Path(directory)
        
        # Load config
        with open(path / "config.json") as f:
            config_dict = json.load(f)
        
        config = AITokenizerConfig(
            vocab_size=config_dict.get("vocab_size", 32000),
            min_frequency=config_dict.get("min_frequency", 2),
            special_tokens=config_dict.get("special_tokens", []),
            domain=config_dict.get("domain", "general"),
            languages=config_dict.get("languages", ["en"]),
            use_ai_extraction=config_dict.get("use_ai_extraction", True),
        )
        
        tokenizer = cls(config)
        
        # Load vocab
        with open(path / "vocab.json") as f:
            tokenizer.vocab = json.load(f)
        
        # Load special tokens
        with open(path / "special_tokens.json") as f:
            tokenizer.special_tokens = json.load(f)
        
        # Load merges
        tokenizer.merges = []
        with open(path / "merges.txt") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    tokenizer.merges.append((parts[0], parts[1]))
        
        # Load protected tokens
        if (path / "protected_tokens.json").exists():
            with open(path / "protected_tokens.json") as f:
                tokenizer.protected_tokens = set(json.load(f))
        
        # Load domain info
        if (path / "domain_info.json").exists():
            with open(path / "domain_info.json") as f:
                tokenizer.domain_info = json.load(f)
        
        return tokenizer
    
    def display_info(self):
        """Display tokenizer information."""
        if not RICH_AVAILABLE:
            print(f"Vocab size: {len(self.vocab)}")
            return
        
        table = Table(title="🤖 AI-Enhanced Tokenizer", show_header=True, header_style="bold cyan")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Vocabulary Size", str(len(self.vocab) + len(self.special_tokens)))
        table.add_row("BPE Merges", str(len(self.merges)))
        table.add_row("Special Tokens", str(len(self.special_tokens)))
        table.add_row("Protected Terms", str(len(self.protected_tokens)))
        table.add_row("Domain", self.config.domain)
        table.add_row("AI Extraction", "✅" if self.config.use_ai_extraction else "❌")
        
        console.print(table)
        
        if self.protected_tokens:
            console.print("\n[bold]Sample Protected Tokens:[/bold]")
            sample = list(self.protected_tokens)[:10]
            console.print(f"  {', '.join(sample)}" + ("..." if len(self.protected_tokens) > 10 else ""))


# ============================================================================
# Factory Functions
# ============================================================================

def create_ai_tokenizer(
    vocab_size: int = 32000,
    domain: str = "general",
    use_ai: bool = True,
    provider: str = "auto"
) -> AIEnhancedTokenizer:
    """
    Create an AI-enhanced tokenizer.
    
    Args:
        vocab_size: Target vocabulary size
        domain: Domain for specialized vocabulary (general, medical, legal, code, scientific)
        use_ai: Whether to use AI for vocabulary extraction
        provider: AI provider (auto, ollama, gemini, openai)
    
    Returns:
        AIEnhancedTokenizer instance
    """
    config = AITokenizerConfig(
        vocab_size=vocab_size,
        domain=domain,
        use_ai_extraction=use_ai,
        use_ai_segmentation=use_ai,
        provider=provider
    )
    
    return AIEnhancedTokenizer(config)


def train_tokenizer_on_files(
    input_path: str,
    output_dir: str,
    vocab_size: int = 32000,
    domain: str = "general",
    use_ai: bool = True
) -> AIEnhancedTokenizer:
    """
    Train tokenizer on text files.
    
    Args:
        input_path: Path to text file or directory of text files
        output_dir: Directory to save tokenizer
        vocab_size: Target vocabulary size
        domain: Domain specialization
        use_ai: Whether to use AI extraction
    
    Returns:
        Trained tokenizer
    """
    input_path = Path(input_path)
    
    # Load texts
    texts = []
    
    if input_path.is_file():
        if input_path.suffix == ".jsonl":
            with open(input_path) as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if "text" in data:
                            texts.append(data["text"])
                        elif "conversations" in data:
                            for conv in data["conversations"]:
                                texts.append(conv.get("value", ""))
                    except:
                        pass
        else:
            with open(input_path) as f:
                texts = [f.read()]
    else:
        # Directory
        for file in input_path.glob("**/*.txt"):
            with open(file) as f:
                texts.append(f.read())
        
        for file in input_path.glob("**/*.jsonl"):
            with open(file) as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if "text" in data:
                            texts.append(data["text"])
                    except:
                        pass
    
    if not texts:
        raise ValueError(f"No texts found in {input_path}")
    
    # Create and train tokenizer
    tokenizer = create_ai_tokenizer(
        vocab_size=vocab_size,
        domain=domain,
        use_ai=use_ai
    )
    
    tokenizer.train(texts)
    tokenizer.save(output_dir)
    
    return tokenizer


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    'AITokenizerConfig',
    'AIDomainAnalyzer',
    'AIEnhancedTokenizer',
    'create_ai_tokenizer',
    'train_tokenizer_on_files',
]
