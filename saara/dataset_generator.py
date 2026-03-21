"""
Dataset Generator Module
Converts labeled data into various dataset formats for training.

Supported output formats (via formats module):
- Alpaca: Domain adaptation, instruction tuning
- ChatML: Chatbot/assistant training
- ShareGPT: Multi-turn conversations
- Completion: Base model continuation
- DPO: RLHF alignment with preference pairs
- ChatML Tools: Function calling/agent training
"""

import json
import csv
import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import pandas as pd

from .labeler import LabeledDocument, LabeledChunk
from .formats import (
    FormatRegistry,
    AlpacaFormat,
    ShareGPTFormat,
    ChatMLFormat,
    CompletionFormat,
    DPOFormat,
    FormatConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class DatasetStats:
    """Statistics about a generated dataset."""
    total_samples: int = 0
    total_documents: int = 0
    categories: Dict[str, int] = None
    avg_input_length: float = 0
    avg_output_length: float = 0
    format: str = ""
    created_at: str = ""
    
    def __post_init__(self):
        if self.categories is None:
            self.categories = {}
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


class DatasetGenerator:
    """
    Generates training datasets from labeled documents.
    Supports multiple output formats: JSONL, CSV, HuggingFace Datasets.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        output_config = self.config.get('output', {})
        
        self.output_dir = Path(output_config.get('directory', './datasets'))
        self.formats = output_config.get('formats', ['jsonl'])
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_all(self, 
                     labeled_docs: List[LabeledDocument],
                     dataset_name: str = "dataset") -> Dict[str, str]:
        """
        Generate all configured dataset types.
        
        Args:
            labeled_docs: List of labeled documents
            dataset_name: Base name for output files
            
        Returns:
            Dictionary of format -> file path
        """
        output_files = {}
        
        # Generate instruction tuning dataset
        instruction_data = self._collect_instruction_data(labeled_docs)
        if instruction_data:
            files = self._save_dataset(
                instruction_data, 
                f"{dataset_name}_instruction",
                "Instruction Tuning"
            )
            output_files['instruction_tuning'] = files
        
        # Generate Q&A dataset
        qa_data = self._collect_qa_data(labeled_docs)
        if qa_data:
            files = self._save_dataset(
                qa_data, 
                f"{dataset_name}_qa",
                "Question Answering"
            )
            output_files['qa'] = files
        
        # Generate summarization dataset
        summary_data = self._collect_summary_data(labeled_docs)
        if summary_data:
            files = self._save_dataset(
                summary_data, 
                f"{dataset_name}_summarization",
                "Summarization"
            )
            output_files['summarization'] = files
        
        # Generate classification dataset
        classification_data = self._collect_classification_data(labeled_docs)
        if classification_data:
            files = self._save_dataset(
                classification_data, 
                f"{dataset_name}_classification",
                "Classification"
            )
            output_files['classification'] = files
        
        # Generate raw text dataset (for continued pretraining)
        text_data = self._collect_text_data(labeled_docs)
        if text_data:
            files = self._save_dataset(
                text_data,
                f"{dataset_name}_pretrain",
                "Pretraining"
            )
            output_files['pretrain'] = files

        # Generate ShareGPT format (multi-turn conversations)
        if 'sharegpt' in self.formats:
            sharegpt_source = qa_data + instruction_data
            if sharegpt_source:
                sharegpt_ready = ShareGPTFormatGenerator.convert(sharegpt_source)
                sharegpt_path = self.output_dir / f"{dataset_name}_sharegpt.jsonl"
                self._save_jsonl(sharegpt_ready, sharegpt_path)
                output_files['sharegpt'] = str(sharegpt_path)
                logger.info(f"Saved ShareGPT dataset: {sharegpt_path}")

        # Generate Alpaca format (domain adaptation)
        if 'alpaca' in self.formats:
            alpaca_source = instruction_data + qa_data
            if alpaca_source:
                alpaca_ready = AlpacaFormatGenerator.convert(alpaca_source)
                alpaca_path = self.output_dir / f"{dataset_name}_alpaca.jsonl"
                self._save_jsonl(alpaca_ready, alpaca_path)
                output_files['alpaca'] = str(alpaca_path)
                logger.info(f"Saved Alpaca dataset: {alpaca_path}")

        # Generate ChatML format (chatbot/assistant)
        if 'chatml' in self.formats:
            chatml_source = instruction_data + qa_data
            if chatml_source:
                chatml_ready = ChatMLFormatGenerator.convert(chatml_source)
                chatml_path = self.output_dir / f"{dataset_name}_chatml.jsonl"
                self._save_jsonl(chatml_ready, chatml_path)
                output_files['chatml'] = str(chatml_path)
                logger.info(f"Saved ChatML dataset: {chatml_path}")

        # Generate Completion format (base model continuation)
        if 'completion' in self.formats:
            if text_data:
                completion_ready = CompletionFormatGenerator.convert(text_data)
                completion_path = self.output_dir / f"{dataset_name}_completion.jsonl"
                self._save_jsonl(completion_ready, completion_path)
                output_files['completion'] = str(completion_path)
                logger.info(f"Saved Completion dataset: {completion_path}")

        # Generate combined dataset statistics
        self._save_statistics(labeled_docs, dataset_name, output_files)
        
        return output_files
    
    def _collect_instruction_data(self, docs: List[LabeledDocument]) -> List[Dict]:
        """Collect instruction-following data."""
        data = []
        
        for doc in docs:
            for chunk in doc.chunks:
                if not chunk.is_suitable:
                    continue
                    
                for instr in chunk.instruction_pairs:
                    if instr.get('instruction') and instr.get('response'):
                        data.append({
                            'instruction': instr['instruction'],
                            'input': '',  # Context if needed
                            'output': instr['response'],
                            'source': doc.title,
                            'category': doc.category,
                            'topic': doc.topics.get('main_topic', ''),
                        })
        
        logger.info(f"Collected {len(data)} instruction samples")
        return data
    
    def _collect_qa_data(self, docs: List[LabeledDocument]) -> List[Dict]:
        """Collect question-answering data."""
        data = []
        
        for doc in docs:
            for chunk in doc.chunks:
                if not chunk.is_suitable:
                    continue
                    
                for qa in chunk.qa_pairs:
                    if qa.get('question') and qa.get('answer'):
                        data.append({
                            'question': qa['question'],
                            'answer': qa['answer'],
                            'context': chunk.chunk.text[:500],  # Truncated context
                            'type': qa.get('type', 'factual'),
                            'source': doc.title,
                            'category': doc.category,
                        })
        
        logger.info(f"Collected {len(data)} Q&A samples")
        return data
    
    def _collect_summary_data(self, docs: List[LabeledDocument]) -> List[Dict]:
        """Collect summarization data."""
        data = []
        
        for doc in docs:
            for chunk in doc.chunks:
                if not chunk.is_suitable or not chunk.summary:
                    continue
                    
                data.append({
                    'text': chunk.chunk.text,
                    'summary': chunk.summary,
                    'source': doc.title,
                    'category': doc.category,
                })
        
        logger.info(f"Collected {len(data)} summarization samples")
        return data
    
    def _collect_classification_data(self, docs: List[LabeledDocument]) -> List[Dict]:
        """Collect classification data."""
        data = []
        
        for doc in docs:
            for chunk in doc.chunks:
                if not chunk.is_suitable:
                    continue
                    
                data.append({
                    'text': chunk.chunk.text,
                    'label': doc.category,
                    'confidence': doc.confidence,
                    'topics': doc.topics.get('subtopics', []),
                    'keywords': doc.topics.get('keywords', []),
                    'complexity': doc.topics.get('complexity_level', 'intermediate'),
                })
        
        logger.info(f"Collected {len(data)} classification samples")
        return data
    
    def _collect_text_data(self, docs: List[LabeledDocument]) -> List[Dict]:
        """Collect raw text for pretraining."""
        data = []
        
        for doc in docs:
            for chunk in doc.chunks:
                if not chunk.is_suitable or chunk.quality_score < 5:
                    continue
                    
                data.append({
                    'text': chunk.chunk.text,
                    'metadata': {
                        'source': doc.title,
                        'category': doc.category,
                        'quality_score': chunk.quality_score,
                    }
                })
        
        logger.info(f"Collected {len(data)} text samples for pretraining")
        return data
    
    def _save_dataset(self, 
                      data: List[Dict], 
                      name: str,
                      description: str) -> Dict[str, str]:
        """Save dataset in configured formats."""
        files = {}
        
        if not data:
            return files
        
        # JSONL format
        if 'jsonl' in self.formats:
            jsonl_path = self.output_dir / f"{name}.jsonl"
            self._save_jsonl(data, jsonl_path)
            files['jsonl'] = str(jsonl_path)
            logger.info(f"Saved JSONL: {jsonl_path}")
        
        # CSV format
        if 'csv' in self.formats:
            csv_path = self.output_dir / f"{name}.csv"
            self._save_csv(data, csv_path)
            files['csv'] = str(csv_path)
            logger.info(f"Saved CSV: {csv_path}")
        
        # HuggingFace format (as JSON with metadata)
        if 'huggingface' in self.formats:
            hf_dir = self.output_dir / f"{name}_hf"
            self._save_huggingface(data, hf_dir, description)
            files['huggingface'] = str(hf_dir)
            logger.info(f"Saved HuggingFace format: {hf_dir}")
        
        return files
    
    def _save_jsonl(self, data: List[Dict], path: Path):
        """Save as JSONL (JSON Lines)."""
        with open(path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    def _save_csv(self, data: List[Dict], path: Path):
        """Save as CSV."""
        if not data:
            return
            
        df = pd.DataFrame(data)
        
        # Handle nested dicts/lists
        for col in df.columns:
            if df[col].apply(lambda x: isinstance(x, (dict, list))).any():
                df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x)
        
        df.to_csv(path, index=False, encoding='utf-8')
    
    def _save_huggingface(self, data: List[Dict], dir_path: Path, description: str):
        """Save in HuggingFace datasets format."""
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # Split into train/val/test
        n = len(data)
        train_end = int(n * 0.8)
        val_end = int(n * 0.9)
        
        splits = {
            'train': data[:train_end],
            'validation': data[train_end:val_end],
            'test': data[val_end:]
        }
        
        for split_name, split_data in splits.items():
            if split_data:
                split_path = dir_path / f"{split_name}.jsonl"
                self._save_jsonl(split_data, split_path)
        
        # Create dataset info
        info = {
            'description': description,
            'features': self._infer_features(data[0]) if data else {},
            'num_examples': {k: len(v) for k, v in splits.items()},
            'created_at': datetime.now().isoformat()
        }
        
        with open(dir_path / 'dataset_info.json', 'w') as f:
            json.dump(info, f, indent=2)
    
    def _infer_features(self, sample: Dict) -> Dict:
        """Infer dataset features from a sample."""
        features = {}
        for key, value in sample.items():
            if isinstance(value, str):
                features[key] = {'dtype': 'string'}
            elif isinstance(value, int):
                features[key] = {'dtype': 'int64'}
            elif isinstance(value, float):
                features[key] = {'dtype': 'float64'}
            elif isinstance(value, list):
                features[key] = {'dtype': 'list'}
            elif isinstance(value, dict):
                features[key] = {'dtype': 'dict'}
        return features
    
    def _save_statistics(self, 
                         docs: List[LabeledDocument],
                         dataset_name: str,
                         output_files: Dict):
        """Save dataset statistics."""
        stats = {
            'dataset_name': dataset_name,
            'created_at': datetime.now().isoformat(),
            'total_documents': len(docs),
            'documents': []
        }
        
        category_counts = {}
        total_chunks = 0
        total_qa = 0
        total_instructions = 0
        
        for doc in docs:
            category_counts[doc.category] = category_counts.get(doc.category, 0) + 1
            total_chunks += len(doc.chunks)
            total_qa += doc.total_qa_pairs
            total_instructions += doc.total_instruction_pairs
            
            stats['documents'].append({
                'title': doc.title,
                'category': doc.category,
                'chunks': len(doc.chunks),
                'qa_pairs': doc.total_qa_pairs,
                'instruction_pairs': doc.total_instruction_pairs
            })
        
        stats['summary'] = {
            'total_chunks': total_chunks,
            'total_qa_pairs': total_qa,
            'total_instruction_pairs': total_instructions,
            'category_distribution': category_counts
        }
        
        stats['output_files'] = output_files
        
        stats_path = self.output_dir / f"{dataset_name}_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
            
        logger.info(f"Saved statistics: {stats_path}")


class AlpacaFormatGenerator:
    """
    Generate datasets in Alpaca format for instruction tuning.

    Delegates to formats.AlpacaFormat for actual conversion.
    """

    @staticmethod
    def convert(data: List[Dict]) -> List[Dict]:
        """Convert to Alpaca format."""
        converter = AlpacaFormat()
        return converter.convert(data)


class ShareGPTFormatGenerator:
    """
    Generate datasets in ShareGPT conversation format.

    Delegates to formats.ShareGPTFormat for actual conversion.
    """

    SYSTEM_PROMPT = "You are AyurGuru, an expert AI Ayurvedic doctor. Answer queries using authentic Samhita knowledge."

    @staticmethod
    def convert(data: List[Dict], system_prompt: str = None) -> List[Dict]:
        """Convert to ShareGPT format."""
        sys_prompt = system_prompt or ShareGPTFormatGenerator.SYSTEM_PROMPT
        config = FormatConfig(system_prompt=sys_prompt, include_system=True)
        converter = ShareGPTFormat(config)
        return converter.convert(data)


class ChatMLFormatGenerator:
    """
    Generate datasets in ChatML format for chatbot training.

    Delegates to formats.ChatMLFormat for actual conversion.
    """

    @staticmethod
    def convert(data: List[Dict], system_prompt: str = "") -> List[Dict]:
        """Convert to ChatML format."""
        config = FormatConfig(system_prompt=system_prompt, include_system=bool(system_prompt))
        converter = ChatMLFormat(config)
        return converter.convert(data)


class CompletionFormatGenerator:
    """
    Generate datasets in completion format for base model pretraining.

    Delegates to formats.CompletionFormat for actual conversion.
    """

    @staticmethod
    def convert(data: List[Dict]) -> List[Dict]:
        """Convert to completion format."""
        converter = CompletionFormat()
        return converter.convert(data)


class DPOFormatGenerator:
    """
    Generate datasets in DPO format for preference-based alignment.

    Delegates to formats.DPOFormat for actual conversion.
    """

    @staticmethod
    def convert(data: List[Dict]) -> List[Dict]:
        """Convert to DPO format."""
        converter = DPOFormat()
        return converter.convert(data)
