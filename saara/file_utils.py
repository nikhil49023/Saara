"""
File Handling Utilities for SAARA

Simple utilities for manual file operations:
- Loading data from various formats (JSON, JSONL, TXT, CSV, Parquet)
- Saving processed data
- Dataset preparation
- No automatic caching or saving - user has full control

Released under the MIT License.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import csv

logger = logging.getLogger(__name__)


# =============================================================================
# Data Loading
# =============================================================================

def load_text(file_path: str) -> str:
    """
    Load text from a file.

    Args:
        file_path: Path to text file

    Returns:
        File contents as string

    Example:
        >>> text = load_text("data.txt")
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def load_texts(file_path: str) -> List[str]:
    """
    Load lines from a text file.

    Args:
        file_path: Path to text file

    Returns:
        List of lines (stripped)

    Example:
        >>> lines = load_texts("data.txt")
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_json(file_path: str) -> Dict[str, Any]:
    """
    Load JSON file.

    Args:
        file_path: Path to JSON file

    Returns:
        Parsed JSON object

    Example:
        >>> data = load_json("data.json")
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    Load JSONL file (one JSON object per line).

    Args:
        file_path: Path to JSONL file

    Returns:
        List of parsed JSON objects

    Example:
        >>> records = load_jsonl("data.jsonl")
    """
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line: {e}")
                    continue

    logger.info(f"Loaded {len(records)} records from {file_path}")
    return records


def load_csv(file_path: str, delimiter: str = ",") -> List[Dict[str, Any]]:
    """
    Load CSV file.

    Args:
        file_path: Path to CSV file
        delimiter: Field delimiter (default: comma)

    Returns:
        List of dictionaries (one per row)

    Example:
        >>> data = load_csv("data.csv")
        >>> data = load_csv("data.tsv", delimiter="\\t")
    """
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            records.append(row)

    logger.info(f"Loaded {len(records)} records from {file_path}")
    return records


def load_parquet(file_path: str) -> List[Dict[str, Any]]:
    """
    Load Parquet file.

    Args:
        file_path: Path to Parquet file

    Returns:
        List of dictionaries

    Example:
        >>> data = load_parquet("data.parquet")

    Requires: pip install pyarrow
    """
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError("Install pyarrow: pip install pyarrow")

    table = pq.read_table(file_path)
    records = table.to_pylist()

    logger.info(f"Loaded {len(records)} records from {file_path}")
    return records


def load_from_file(file_path: str, **kwargs) -> Union[str, List, Dict]:
    """
    Auto-detect and load from file.

    Supports: .txt, .json, .jsonl, .csv, .tsv, .parquet

    Args:
        file_path: Path to file
        **kwargs: Additional arguments for specific loaders

    Returns:
        Loaded data

    Example:
        >>> data = load_from_file("data.jsonl")
        >>> data = load_from_file("data.csv")
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".txt":
        return load_text(file_path)
    elif suffix == ".json":
        return load_json(file_path)
    elif suffix == ".jsonl":
        return load_jsonl(file_path)
    elif suffix == ".csv":
        return load_csv(file_path, **kwargs)
    elif suffix == ".tsv":
        return load_csv(file_path, delimiter="\t", **kwargs)
    elif suffix == ".parquet":
        return load_parquet(file_path)
    else:
        logger.warning(f"Unknown file format: {suffix}. Treating as text.")
        return load_text(file_path)


# =============================================================================
# Data Saving
# =============================================================================

def save_text(text: str, file_path: str) -> None:
    """
    Save text to file.

    Args:
        text: Text to save
        file_path: Output path

    Example:
        >>> save_text("hello world", "output.txt")
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)

    logger.info(f"Saved text to {file_path}")


def save_texts(texts: List[str], file_path: str) -> None:
    """
    Save list of texts (one per line) to file.

    Args:
        texts: List of text strings
        file_path: Output path

    Example:
        >>> save_texts(["line1", "line2"], "output.txt")
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for text in texts:
            f.write(text + "\n")

    logger.info(f"Saved {len(texts)} texts to {file_path}")


def save_json(data: Dict[str, Any], file_path: str, pretty: bool = True) -> None:
    """
    Save data to JSON file.

    Args:
        data: Dictionary to save
        file_path: Output path
        pretty: Whether to pretty-print (default: True)

    Example:
        >>> save_json({"key": "value"}, "output.json")
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2 if pretty else None, ensure_ascii=False)

    logger.info(f"Saved JSON to {file_path}")


def save_jsonl(records: List[Dict[str, Any]], file_path: str) -> None:
    """
    Save records to JSONL file (one JSON per line).

    Args:
        records: List of dictionaries
        file_path: Output path

    Example:
        >>> save_jsonl([{"id": 1}, {"id": 2}], "output.jsonl")
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(records)} records to {file_path}")


def save_csv(
    records: List[Dict[str, Any]],
    file_path: str,
    fieldnames: Optional[List[str]] = None,
    delimiter: str = ","
) -> None:
    """
    Save records to CSV file.

    Args:
        records: List of dictionaries
        file_path: Output path
        fieldnames: Column names (auto-detected if None)
        delimiter: Field delimiter (default: comma)

    Example:
        >>> save_csv([{"name": "Alice", "age": 30}], "output.csv")
        >>> save_csv([...], "output.tsv", delimiter="\\t")
    """
    if not records:
        logger.warning("No records to save")
        return

    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    fieldnames = fieldnames or list(records[0].keys())

    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(records)

    logger.info(f"Saved {len(records)} records to {file_path}")


def save_parquet(records: List[Dict[str, Any]], file_path: str) -> None:
    """
    Save records to Parquet file.

    Args:
        records: List of dictionaries
        file_path: Output path

    Example:
        >>> save_parquet([{"id": 1}, {"id": 2}], "output.parquet")

    Requires: pip install pyarrow
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError("Install pyarrow: pip install pyarrow")

    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    table = pa.Table.from_pylist(records)
    pq.write_table(table, file_path)

    logger.info(f"Saved {len(records)} records to {file_path}")


def save_to_file(
    data: Union[str, List, Dict],
    file_path: str,
    **kwargs
) -> None:
    """
    Auto-detect and save to file.

    Supports: .txt, .json, .jsonl, .csv, .tsv, .parquet

    Args:
        data: Data to save
        file_path: Output path
        **kwargs: Additional arguments for specific savers

    Example:
        >>> save_to_file("hello", "output.txt")
        >>> save_to_file(records, "output.jsonl")
        >>> save_to_file(records, "output.csv")
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".txt":
        if isinstance(data, str):
            save_text(data, file_path)
        elif isinstance(data, list):
            save_texts(data, file_path)
        else:
            save_text(str(data), file_path)
    elif suffix == ".json":
        save_json(data, file_path, **kwargs)
    elif suffix == ".jsonl":
        save_jsonl(data, file_path)
    elif suffix == ".csv":
        save_csv(data, file_path, **kwargs)
    elif suffix == ".tsv":
        save_csv(data, file_path, delimiter="\t", **kwargs)
    elif suffix == ".parquet":
        save_parquet(data, file_path)
    else:
        logger.warning(f"Unknown file format: {suffix}. Saving as text.")
        save_text(str(data), file_path)


# =============================================================================
# Dataset Utilities
# =============================================================================

def extract_texts(records: List[Dict[str, Any]], field_name: str) -> List[str]:
    """
    Extract text field from records.

    Args:
        records: List of records
        field_name: Field name to extract (e.g., "text", "content", "prompt")

    Returns:
        List of text strings

    Example:
        >>> texts = extract_texts(records, "text")
    """
    texts = []
    for record in records:
        if field_name in record:
            text = record[field_name]
            if isinstance(text, str) and text.strip():
                texts.append(text)

    logger.info(f"Extracted {len(texts)} texts from field '{field_name}'")
    return texts


def extract_training_pairs(
    records: List[Dict[str, Any]],
    prompt_field: str = "prompt",
    response_field: str = "response"
) -> List[Dict[str, str]]:
    """
    Extract prompt-response pairs from records.

    Args:
        records: List of records
        prompt_field: Field name for prompts
        response_field: Field name for responses

    Returns:
        List of {"prompt": ..., "response": ...} dicts

    Example:
        >>> pairs = extract_training_pairs(records)
    """
    pairs = []
    for record in records:
        if prompt_field in record and response_field in record:
            pairs.append({
                "prompt": record[prompt_field],
                "response": record[response_field]
            })

    logger.info(f"Extracted {len(pairs)} training pairs")
    return pairs


def split_dataset(
    records: List[Dict[str, Any]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> tuple:
    """
    Split dataset into train/val/test.

    Args:
        records: List of records
        train_ratio: Fraction for training (default: 0.8)
        val_ratio: Fraction for validation (default: 0.1)

    Returns:
        (train_records, val_records, test_records)

    Example:
        >>> train, val, test = split_dataset(records)
    """
    import random

    total = len(records)
    shuffled = records.copy()
    random.shuffle(shuffled)

    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)

    train = shuffled[:train_size]
    val = shuffled[train_size:train_size + val_size]
    test = shuffled[train_size + val_size:]

    logger.info(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")

    return train, val, test


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "load_text",
    "load_texts",
    "load_json",
    "load_jsonl",
    "load_csv",
    "load_parquet",
    "load_from_file",
    "save_text",
    "save_texts",
    "save_json",
    "save_jsonl",
    "save_csv",
    "save_parquet",
    "save_to_file",
    "extract_texts",
    "extract_training_pairs",
    "split_dataset",
]
