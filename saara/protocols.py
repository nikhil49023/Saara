"""
Data-Centric AI Protocols for SAARA
Implements the 4-step refinery process: Clean, Dedupe, Verify, Secure.
"""

import logging
import re
import math
from typing import List, Dict, Any, Tuple, Set
from collections import defaultdict
import hashlib

# Optional dependencies
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from datasketch import MinHash, MinHashLSH
    DATASKETCH_AVAILABLE = True
except ImportError:
    DATASKETCH_AVAILABLE = False

try:
    import sympy
    from sympy.parsing.sympy_parser import parse_expr
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False

try:
    import scipy
    import networkx as nx
    GRAPH_AVAILABLE = True
except ImportError:
    GRAPH_AVAILABLE = False

logger = logging.getLogger(__name__)

class QualityFilter:
    """Protocol 1: Signal-to-Noise (Heuristic & Perplexity)"""
    
    def __init__(self, use_perplexity: bool = False, device: str = "cpu"):
        self.use_perplexity = use_perplexity
        self.device = device
        self.ppl_model = None
        self.ppl_tokenizer = None
        
        # Gopher-style heuristics
        self.min_words = 6
        self.max_word_length = 25
        self.symbol_ratio_threshold = 0.1
        self.min_avg_word_length = 3
        self.max_avg_word_length = 10
        
    def _load_ppl_model(self):
        if not self.ppl_model and TRANSFORMERS_AVAILABLE:
            logger.info("⏳ Loading GPT-2 for perplexity filtering...")
            self.ppl_tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self.ppl_model = AutoModelForCausalLM.from_pretrained("gpt2").to(self.device)
            self.ppl_model.eval()

    def check_heuristics(self, text: str) -> Tuple[bool, str]:
        """Apply Gopher-style rules. Returns (Pass/Fail, Reason)."""
        words = text.split()
        if not words:
            return False, "Empty text"
            
        num_words = len(words)
        if num_words < self.min_words:
            return False, "Too short"
            
        # Symbol ratio
        symbols = sum(1 for c in text if not c.isalnum() and not c.isspace())
        if symbols / len(text) > self.symbol_ratio_threshold:
            return False, "Too many symbols"
            
        # Word length checks
        lens = [len(w) for w in words]
        avg_len = sum(lens) / num_words
        if avg_len < self.min_avg_word_length or avg_len > self.max_avg_word_length:
            return False, "Abnormal word length"
            
        return True, "OK"

    def check_perplexity(self, text: str, threshold: float = 100.0) -> Tuple[bool, float]:
        """Calculate perplexity. Lower is better."""
        if not self.use_perplexity or not TRANSFORMERS_AVAILABLE:
            return True, 0.0
            
        self._load_ppl_model()
        
        try:
            encodings = self.ppl_tokenizer(text, return_tensors="pt")
            input_ids = encodings.input_ids.to(self.device)
            
            # Truncate if too long (GPT-2 ctx is 1024)
            if input_ids.shape[1] > 1024:
                input_ids = input_ids[:, :1024]
                
            with torch.no_grad():
                outputs = self.ppl_model(input_ids, labels=input_ids)
                loss = outputs.loss
                ppl = torch.exp(loss).item()
                
            if ppl > threshold:
                return False, ppl
            return True, ppl
        except Exception as e:
            logger.warning(f"Perplexity check failed: {e}")
            return True, 0.0

class Deduplicator:
    """Protocol 2: Decontamination (MinHash LSH)"""
    
    def __init__(self, threshold: float = 0.8, num_perm: int = 128):
        self.threshold = threshold
        self.num_perm = num_perm
        if not DATASKETCH_AVAILABLE:
            logger.warning("datasketch not installed. Deduplication will be limited.")
            
    def compute_minhash(self, text: str) -> Any:
        if not DATASKETCH_AVAILABLE:
            return None
            
        m = MinHash(num_perm=self.num_perm)
        # 3-shingles
        words = text.lower().split()
        for i in range(len(words) - 2):
            shingle = " ".join(words[i:i+3])
            m.update(shingle.encode('utf8'))
        return m
        
    def find_duplicates(self, texts: List[str]) -> List[int]:
        """Return indices of unique texts (removing duplicates)."""
        if not DATASKETCH_AVAILABLE:
            # Fallback to exact hash
            seen = set()
            unique_indices = []
            for i, text in enumerate(texts):
                h = hashlib.md5(text.encode()).hexdigest()
                if h not in seen:
                    seen.add(h)
                    unique_indices.append(i)
            return unique_indices

        lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        unique_indices = []
        
        # We need to process sequentially to respect order
        # Optimally we query LSH then insert
        for i, text in enumerate(texts):
            m = self.compute_minhash(text)
            result = lsh.query(m)
            
            if len(result) == 0:
                lsh.insert(str(i), m)
                unique_indices.append(i)
            else:
                # Duplicate found
                pass
                
        return unique_indices

class LogicVerifier:
    """Protocol 3: Verification (Symbolic Grounding)"""
    
    def __init__(self):
        pass
        
    def verify_math_consistency(self, text: str) -> bool:
        """
        Extract equations and verify strict syntax.
        (Full math solving is hard, we verify syntax first)
        """
        # Find LaTeX patterns
        matches = re.findall(r'\$\$(.+?)\$\$|\$(.+?)\$', text)
        valid_count = 0
        total_count = 0
        
        for m in matches:
            eq = m[0] or m[1]
            total_count += 1
            if self._check_sympy_syntax(eq):
                valid_count += 1
                
        if total_count == 0:
            return True # No math claim to fail
            
        # Pass if > 80% equations are valid syntax
        return (valid_count / total_count) > 0.8
        
    def _check_sympy_syntax(self, eq_str: str) -> bool:
        if not SYMPY_AVAILABLE:
            return True
        try:
            # Very basic check
            # Replace LaTeX symbols with SymPy compatible ones is complex
            # We just check for unbalanced braces or illegal chars here
            if eq_str.count('{') != eq_str.count('}'):
                return False
            return True
        except:
            return False

class SafetyGuard:
    """Protocol 4: Privacy & Safety (PII Redaction)"""
    
    def __init__(self):
        # Regex patterns for common PII
        self.patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b(\+\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b',
            'ip': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
            'api_key': r'(?i)(api[_-]?key|token|secret)[\s:=]+([a-zA-Z0-9_\-]{20,})',
        }
        
    def redact(self, text: str) -> str:
        redacted = text
        for ptype, pattern in self.patterns.items():
            redacted = re.sub(pattern, f"<{ptype.upper()}_REDACTED>", redacted)
        return redacted

class ProtocolEngine:
    """Main Refinery Engine"""
    
    def __init__(self):
        self.cleaner = QualityFilter()
        self.deduper = Deduplicator()
        self.verifier = LogicVerifier()
        self.guard = SafetyGuard()
        
    def refine_dataset(self, texts: List[str], strict_mode: bool = False) -> List[str]:
        """Run full pipeline."""
        
        # 1. Clean
        passed_clean = []
        for t in texts:
            ok, reason = self.cleaner.check_heuristics(t)
            if ok:
                passed_clean.append(t)
            elif not strict_mode:
                # In non-strict, maybe keep if length is the only issue?
                # No, better drop.
                pass
                
        # 2. Redact
        redacted = [self.guard.redact(t) for t in passed_clean]
        
        # 3. Verify (Math) - Only filtering if strictly bad syntax
        verified = []
        for t in redacted:
            if self.verifier.verify_math_consistency(t):
                verified.append(t)
                
        # 4. Dedupe
        unique_indices = self.deduper.find_duplicates(verified)
        final_set = [verified[i] for i in unique_indices]
        
        logger.info(f"Refinery: {len(texts)} -> {len(final_set)} docs")
        return final_set
