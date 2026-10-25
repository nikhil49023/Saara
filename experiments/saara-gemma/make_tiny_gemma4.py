"""Build a tiny random-weight model with Gemma 4's architecture for pipeline tests (no GPU, ~26M params).

It keeps the real config's structure (hybrid sliding/global attention, KV-shared layers, per-layer
embeddings, vision and audio towers) and the real processor/tokenizer, with every size shrunk.
Its outputs are meaningless: it exists only to test data formatting, loss masking, LoRA and saving.

Usage: python make_tiny_gemma4.py OUT_DIR    then    TINY_GEMMA4=OUT_DIR python -m pytest test_pipeline.py
"""
from __future__ import annotations

import sys

import torch
from transformers import AutoConfig, AutoModelForMultimodalLM, AutoProcessor

SOURCE = "google/gemma-4-E4B-it"


def main(out: str) -> None:
    cfg = AutoConfig.from_pretrained(SOURCE)
    text = cfg.text_config
    text.hidden_size, text.intermediate_size, text.num_hidden_layers = 64, 128, 4
    text.num_attention_heads, text.num_key_value_heads, text.head_dim, text.global_head_dim = 2, 1, 16, 32
    text.num_kv_shared_layers, text.hidden_size_per_layer_input = 2, 8
    text.layer_types = ["sliding_attention", "full_attention", "sliding_attention", "full_attention"]
    text.per_layer_config = {1: {"head_dim": 32}, 3: {"head_dim": 32}}  # global layers keep their wider heads
    vision = cfg.vision_config
    vision.hidden_size, vision.intermediate_size, vision.num_hidden_layers = 32, 64, 1
    vision.num_attention_heads = vision.num_key_value_heads = 2
    vision.head_dim = vision.global_head_dim = 16
    audio = cfg.audio_config
    audio.hidden_size, audio.num_hidden_layers, audio.num_attention_heads = 32, 1, 2
    torch.manual_seed(0)
    model = AutoModelForMultimodalLM.from_config(cfg, dtype=torch.float32)
    print(f"{type(model).__name__}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")
    model.save_pretrained(out)
    AutoProcessor.from_pretrained(SOURCE).save_pretrained(out)


if __name__ == "__main__":
    main(sys.argv[1])
