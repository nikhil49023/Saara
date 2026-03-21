"""
Data Distillation Module
Optimizes and distills the generated datasets for training a smaller model.
"""

import json
import logging
import random
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from rich.console import Console
from rich.progress import Progress

console = Console()
logger = logging.getLogger(__name__)

class DataDistiller:
    """
    Distills large datasets into high-quality training sets for student models.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.output_dir = Path(self.config.get('output', {}).get('directory', 'datasets'))
        
    def distill_batch(self, batch_name: str):
        """
        Consolidate and filter datasets from a batch run.
        
        Args:
            batch_name: Name of the batch run (prefix of files)
        """
        console.print(f"\n[bold]🧪 Distilling Data for: {batch_name}[/bold]")
        
        # 1. Consolidate all JSONL files
        data = self._consolidate_data(batch_name)
        
        if not data:
            console.print("[red]No data found to distill.[/red]")
            return
            
        # 2. Filter low quality
        filtered_data = self._filter_quality(data)
        
        # 3. Create Training Splits
        self._create_splits(filtered_data, batch_name)
        
        console.print(f"[green]Distillation Complete![/green]")
        console.print(f"Original: {len(data)} samples")
        console.print(f"Filtered: {len(filtered_data)} samples")
        
    def _consolidate_data(self, batch_name: str) -> List[Dict]:
        """Aggregate all relevant JSONL files."""
        all_data = []
        
        # We look for all formatted jsonl files in the output directory
        # The pattern seems to be {doc_name}_{type}.jsonl based on previous logs
        # or {batch_name}_{type}.jsonl?
        # Actually pipeline produces separate files per document in the target folder.
        
        # Let's verify the pattern from logs: E:\Data Set\Formatted\Suraksha\05112021..._instruction.jsonl
        # If user provides a directory to main.py batch, it puts files there.
        
        search_dir = self.output_dir
        if not search_dir.exists():
            console.print(f"[yellow]Output dir {search_dir} does not exist. Checking local datasets folder.[/yellow]")
            search_dir = Path("datasets")
            
        files = list(search_dir.glob("*_sharegpt.jsonl"))
        if not files:
            files = list(search_dir.glob("*.jsonl"))
            
        console.print(f"Found {len(files)} files to process in {search_dir}")
        
        with Progress() as progress:
            task = progress.add_task("[cyan]Loading data...", total=len(files))
            
            for f in files:
                try:
                    with open(f, 'r', encoding='utf-8') as reader:
                        for line in reader:
                            if line.strip():
                                try:
                                    item = json.loads(line)
                                    # Normalize format to ShareGPT (conversations)
                                    if 'conversations' in item:
                                        all_data.append(item)
                                    elif 'instruction' in item and 'response' in item:
                                        # Convert instruction/response to ShareGPT
                                        all_data.append({
                                            "conversations": [
                                                {"from": "system", "value": "You are a helpful assistant."},
                                                {"from": "human", "value": item['instruction']},
                                                {"from": "gpt", "value": item['response']}
                                            ],
                                            "source": f.name
                                        })
                                    elif 'question' in item and 'answer' in item:
                                        all_data.append({
                                            "conversations": [
                                                {"from": "system", "value": "You are a helpful assistant."},
                                                {"from": "human", "value": item['question']},
                                                {"from": "gpt", "value": item['answer']}
                                            ],
                                            "source": f.name
                                        })
                                except:
                                    pass
                except Exception as e:
                    logger.warning(f"Error reading {f}: {e}")
                
                progress.advance(task)
                
        return all_data

    def _filter_quality(self, data: List[Dict]) -> List[Dict]:
        """Filter out low quality samples."""
        filtered = []
        seen_prompts = set()
        
        for item in data:
            try:
                convs = item.get('conversations', [])
                if len(convs) < 2: continue
                
                user_msg = next((c['value'] for c in convs if c['from'] == 'human'), "")
                assistant_msg = next((c['value'] for c in convs if c['from'] == 'gpt'), "")
                
                # De-duplication
                if user_msg in seen_prompts:
                    continue
                seen_prompts.add(user_msg)
                
                # Length heuristics
                if len(user_msg) < 10 or len(assistant_msg) < 10:
                    continue
                    
                # JSON pollution check (if model outputted raw json when not asked)
                if assistant_msg.strip().startswith('{') and assistant_msg.strip().endswith('}'):
                    # Check if it looks like a meta-response "Here is the JSON"
                    if "confidence" in assistant_msg and "category" in assistant_msg:
                        continue
                        
                filtered.append(item)
                
            except:
                continue
                
        return filtered

    def _create_splits(self, data: List[Dict], prefix: str):
        """Create train/val splits and save."""
        random.shuffle(data)
        
        # 90/10 split
        split_idx = int(len(data) * 0.9)
        train_data = data[:split_idx]
        val_data = data[split_idx:]
        
        train_path = self.output_dir / "distilled_train.jsonl"
        val_path = self.output_dir / "distilled_val.jsonl"
        
        with open(train_path, 'w', encoding='utf-8') as f:
            for item in train_data:
                f.write(json.dumps(item) + '\n')
                
        with open(val_path, 'w', encoding='utf-8') as f:
            for item in val_data:
                f.write(json.dumps(item) + '\n')
                
        console.print(f"Saved Train: {train_path} ({len(train_data)})")
        console.print(f"Saved Val: {val_path} ({len(val_data)})")

if __name__ == "__main__":
    # Test
    distiller = DataDistiller()
    distiller.distill_batch("test")
