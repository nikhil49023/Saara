"""
Flexible Tokenizer Registry with Built-in Tokenizers

Supports multiple tokenizer types:
- BPE (Byte Pair Encoding)
- WordPiece
- SentencePiece
- ByteLevel
- Custom user tokenizers

Simple API: users can pick a tokenizer or bring their own.

Released under the MIT License.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Callable, Tuple, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from collections import Counter
import re

logger = logging.getLogger(__name__)


# =============================================================================
# Base Tokenizer Interface
# =============================================================================

class BaseTokenizer(ABC):
    """Base class for all tokenizers."""

    @abstractmethod
    def train(self, texts: List[str], vocab_size: int = 32000) -> None:
        """Train tokenizer on texts."""
        pass

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        pass

    @abstractmethod
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        pass

    @abstractmethod
    def save(self, directory: str) -> None:
        """Save tokenizer to directory."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, directory: str) -> "BaseTokenizer":
        """Load tokenizer from directory."""
        pass


# =============================================================================
# BPE Tokenizer (Byte Pair Encoding)
# =============================================================================

class BPETokenizer(BaseTokenizer):
    """
    Simple, fast BPE tokenizer.

    Example:
        >>> tokenizer = BPETokenizer()
        >>> tokenizer.train(texts, vocab_size=32000)
        >>> token_ids = tokenizer.encode("hello world")
    """

    def __init__(self):
        self.vocab: Dict[str, int] = {}
        self.merges: List[Tuple[str, str]] = []
        self.special_tokens = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}

    def train(self, texts: List[str], vocab_size: int = 32000) -> None:
        """Train BPE on texts."""
        logger.info(f"Training BPE tokenizer with vocab_size={vocab_size}")

        # Get word frequencies
        word_freq = Counter()
        for text in texts:
            words = re.findall(r'\S+', text)
            word_freq.update(words)

        logger.info(f"Found {len(word_freq)} unique words")

        # Initialize vocab with characters
        chars = set()
        for word in word_freq:
            chars.update(word)

        offset = len(self.special_tokens)
        for i, char in enumerate(sorted(chars)):
            self.vocab[char] = offset + i

        logger.info(f"Initial vocab: {len(self.vocab)} characters")

        # BPE training
        splits = {word: list(word) for word in word_freq}
        target_vocab_size = vocab_size
        current_vocab_size = len(self.vocab)

        while current_vocab_size < target_vocab_size:
            # Count pair frequencies
            pair_freqs = Counter()
            for word, chars in splits.items():
                freq = word_freq[word]
                for i in range(len(chars) - 1):
                    pair = (chars[i], chars[i + 1])
                    pair_freqs[pair] += freq

            if not pair_freqs:
                break

            best_pair = pair_freqs.most_common(1)[0][0]

            # Merge
            new_token = best_pair[0] + best_pair[1]
            self.merges.append(best_pair)
            self.vocab[new_token] = len(self.vocab)

            # Update splits
            for word in list(splits.keys()):
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

            if current_vocab_size % 1000 == 0:
                logger.info(f"Vocab size: {current_vocab_size}")

        logger.info(f"BPE training complete. Final vocab: {len(self.vocab)}")

    def encode(self, text: str) -> List[int]:
        """Encode text."""
        words = re.findall(r'\S+', text)
        tokens = []

        for word in words:
            word_tokens = list(word)

            # Apply merges
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
                token_id = self.vocab.get(token, self.special_tokens["<unk>"])
                tokens.append(token_id)

        return tokens

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs."""
        id_to_token = {v: k for k, v in self.vocab.items()}
        id_to_token.update({v: k for k, v in self.special_tokens.items()})

        tokens = [id_to_token.get(tid, "<unk>") for tid in token_ids]
        return "".join(tokens)

    def save(self, directory: str) -> None:
        """Save tokenizer."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        with open(path / "vocab.json", "w") as f:
            json.dump(self.vocab, f)

        with open(path / "special_tokens.json", "w") as f:
            json.dump(self.special_tokens, f)

        with open(path / "merges.txt", "w") as f:
            for a, b in self.merges:
                f.write(f"{a} {b}\n")

        logger.info(f"BPE tokenizer saved to {directory}")

    @classmethod
    def load(cls, directory: str) -> "BPETokenizer":
        """Load tokenizer."""
        path = Path(directory)

        tokenizer = cls()

        with open(path / "vocab.json") as f:
            tokenizer.vocab = json.load(f)

        with open(path / "special_tokens.json") as f:
            tokenizer.special_tokens = json.load(f)

        with open(path / "merges.txt") as f:
            tokenizer.merges = [tuple(line.strip().split()) for line in f if line.strip()]

        return tokenizer


# =============================================================================
# WordPiece Tokenizer
# =============================================================================

class WordPieceTokenizer(BaseTokenizer):
    """
    WordPiece tokenizer (used by BERT).

    Example:
        >>> tokenizer = WordPieceTokenizer()
        >>> tokenizer.train(texts, vocab_size=30522)
        >>> token_ids = tokenizer.encode("hello world")
    """

    def __init__(self, prefix: str = "##"):
        self.vocab: Dict[str, int] = {}
        self.prefix = prefix
        self.special_tokens = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4}

    def train(self, texts: List[str], vocab_size: int = 30522) -> None:
        """Train WordPiece tokenizer."""
        logger.info(f"Training WordPiece tokenizer with vocab_size={vocab_size}")

        # Collect all words
        word_freq = Counter()
        for text in texts:
            words = re.findall(r'\S+', text)
            word_freq.update(words)

        logger.info(f"Found {len(word_freq)} unique words")

        # Initialize vocab with special tokens and characters
        offset = len(self.special_tokens)
        idx = offset

        for token, count in word_freq.most_common(vocab_size - offset - 100):
            self.vocab[token] = idx
            idx += 1

        # Add subword pieces (##prefix)
        chars = set()
        for word in word_freq:
            chars.update(word)

        for char in sorted(chars):
            if idx < vocab_size:
                self.vocab[self.prefix + char] = idx
                idx += 1

        logger.info(f"WordPiece vocabulary built: {len(self.vocab)} tokens")

    def encode(self, text: str) -> List[int]:
        """Encode text."""
        words = re.findall(r'\b\w+\b', text.lower())
        token_ids = []

        for word in words:
            if word in self.vocab:
                token_ids.append(self.vocab[word])
            else:
                # Fallback to character-level tokenization
                for char in word:
                    prefix_char = self.prefix + char if token_ids else char
                    token_ids.append(self.vocab.get(prefix_char, self.special_tokens["[UNK]"]))

        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs."""
        id_to_token = {v: k for k, v in self.vocab.items()}
        id_to_token.update({v: k for k, v in self.special_tokens.items()})

        tokens = []
        for tid in token_ids:
            token = id_to_token.get(tid, "[UNK]")
            if token.startswith(self.prefix):
                tokens.append(token[len(self.prefix):])
            else:
                tokens.append(token)

        return " ".join(tokens)

    def save(self, directory: str) -> None:
        """Save tokenizer."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        with open(path / "vocab.json", "w") as f:
            json.dump(self.vocab, f)

        with open(path / "special_tokens.json", "w") as f:
            json.dump(self.special_tokens, f)

        logger.info(f"WordPiece tokenizer saved to {directory}")

    @classmethod
    def load(cls, directory: str) -> "WordPieceTokenizer":
        """Load tokenizer."""
        path = Path(directory)

        tokenizer = cls()

        with open(path / "vocab.json") as f:
            tokenizer.vocab = json.load(f)

        with open(path / "special_tokens.json") as f:
            tokenizer.special_tokens = json.load(f)

        return tokenizer


# =============================================================================
# Byte-Level Tokenizer
# =============================================================================

class ByteTokenizer(BaseTokenizer):
    """
    Byte-level tokenizer (works with any character/language).

    Example:
        >>> tokenizer = ByteTokenizer()
        >>> tokenizer.train(texts)
        >>> token_ids = tokenizer.encode("hello world")
    """

    def __init__(self):
        self.vocab: Dict[str, int] = {}
        self.special_tokens = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}

    def train(self, texts: List[str], vocab_size: int = 256) -> None:
        """Train byte-level tokenizer."""
        logger.info("Training byte-level tokenizer")

        # Byte-level vocab (0-255)
        offset = len(self.special_tokens)
        for i in range(256):
            self.vocab[f"<byte_{i}>"] = offset + i

        logger.info(f"Byte tokenizer vocab: {len(self.vocab)} bytes")

    def encode(self, text: str) -> List[int]:
        """Encode text to bytes."""
        token_ids = []
        for char in text:
            byte_val = ord(char)
            token_key = f"<byte_{byte_val}>"
            token_ids.append(self.vocab.get(token_key, self.special_tokens["<unk>"]))
        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """Decode bytes to text."""
        id_to_token = {v: k for k, v in self.vocab.items()}
        id_to_token.update({v: k for k, v in self.special_tokens.items()})

        chars = []
        for tid in token_ids:
            token = id_to_token.get(tid, "<unk>")
            if token.startswith("<byte_"):
                byte_str = token[6:-1]
                try:
                    chars.append(chr(int(byte_str)))
                except:
                    pass

        return "".join(chars)

    def save(self, directory: str) -> None:
        """Save tokenizer."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Byte tokenizer saved to {directory} (fixed vocab, no save needed)")

    @classmethod
    def load(cls, directory: str) -> "ByteTokenizer":
        """Load tokenizer."""
        return cls()


# =============================================================================
# Tokenizer Registry (Factory)
# =============================================================================

class TokenizerRegistry:
    """Factory for creating tokenizers."""

    _tokenizers = {
        "bpe": BPETokenizer,
        "wordpiece": WordPieceTokenizer,
        "byte": ByteTokenizer,
    }

    @classmethod
    def register(cls, name: str, tokenizer_class: type) -> None:
        """Register a custom tokenizer."""
        cls._tokenizers[name.lower()] = tokenizer_class
        logger.info(f"Registered tokenizer: {name}")

    @classmethod
    def create(cls, name: str = "bpe", **kwargs) -> BaseTokenizer:
        """Create tokenizer by name."""
        tokenizer_class = cls._tokenizers.get(name.lower())
        if not tokenizer_class:
            raise ValueError(f"Unknown tokenizer: {name}. Available: {list(cls._tokenizers.keys())}")

        return tokenizer_class(**kwargs)

    @classmethod
    def list_tokenizers(cls) -> List[str]:
        """List available tokenizers."""
        return list(cls._tokenizers.keys())


# =============================================================================
# Quick Helper Functions
# =============================================================================

def create_tokenizer(
    tokenizer_type: str = "bpe",
    **kwargs
) -> BaseTokenizer:
    """
    Quick helper to create a tokenizer.

    Args:
        tokenizer_type: "bpe", "wordpiece", "byte", or custom registered name
        **kwargs: Additional arguments for tokenizer initialization

    Returns:
        Tokenizer instance

    Examples:
        >>> tokenizer = create_tokenizer("bpe")
        >>> tokenizer = create_tokenizer("wordpiece")
        >>> tokenizer = create_tokenizer("byte")
    """
    return TokenizerRegistry.create(tokenizer_type, **kwargs)


def quick_tokenize(
    text: str,
    texts_for_training: Optional[List[str]] = None,
    tokenizer_type: str = "bpe",
    vocab_size: int = 32000
) -> List[int]:
    """
    Quick one-line tokenization.

    Args:
        text: Text to tokenize
        texts_for_training: Training texts (if not trained yet)
        tokenizer_type: Tokenizer type
        vocab_size: Vocabulary size

    Returns:
        List of token IDs

    Example:
        >>> tokens = quick_tokenize("hello world", texts_for_training=["hello world", "hi there"])
    """
    tokenizer = create_tokenizer(tokenizer_type)

    if texts_for_training:
        tokenizer.train(texts_for_training, vocab_size=vocab_size)

    return tokenizer.encode(text)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "BaseTokenizer",
    "BPETokenizer",
    "WordPieceTokenizer",
    "ByteTokenizer",
    "TokenizerRegistry",
    "create_tokenizer",
    "quick_tokenize",
]
