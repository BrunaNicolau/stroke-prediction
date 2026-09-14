"""
src/assistant/llm_backend.py

LLM backend selection for the medical assistant: tries to load the
LoRA-fine-tuned local model (produced by notebooks/04_finetuning.ipynb,
adapter downloaded from Colab into results/finetuning/lora_adapter/)
first; if the adapter is not present locally (no GPU, fine-tuning not run
yet), falls back to the Gemini wrapper already used in src/llm/client.py —
so the assistant demos are always runnable, with or without a fine-tuned
model on hand.
"""

from __future__ import annotations

import os
from typing import Callable

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_ADAPTER_DIR = os.path.join(_PROJECT_ROOT, "results", "finetuning", "lora_adapter")
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def local_adapter_available(adapter_dir: str = DEFAULT_ADAPTER_DIR) -> bool:
    """True if a fine-tuned LoRA adapter has been downloaded locally."""
    return os.path.isdir(adapter_dir) and bool(os.listdir(adapter_dir))


def load_local_pipeline(
    base_model: str = DEFAULT_BASE_MODEL,
    adapter_dir: str = DEFAULT_ADAPTER_DIR,
    max_new_tokens: int = 512,
) -> Callable[[str], str]:
    """
    Load the base model + LoRA adapter locally via transformers/peft and
    wrap it as a callable `generate(prompt) -> str`.

    Imports the heavy ML stack (torch/transformers/peft) only when called,
    so this function is only exercised when a local adapter is actually
    present — importing llm_backend itself never requires a GPU.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float32)
    model = PeftModel.from_pretrained(base, adapter_dir)
    text_gen = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=max_new_tokens)

    def _generate(prompt: str) -> str:
        output = text_gen(prompt, do_sample=False)[0]["generated_text"]
        return output[len(prompt):].strip()

    return _generate


def get_generate_fn(
    adapter_dir: str = DEFAULT_ADAPTER_DIR, base_model: str = DEFAULT_BASE_MODEL
) -> Callable[[str], str]:
    """
    Return a `generate(prompt) -> str` function: the local fine-tuned
    model if its adapter is available, otherwise the Gemini API
    (src/llm/client.generate) as a fallback.
    """
    if local_adapter_available(adapter_dir):
        return load_local_pipeline(base_model=base_model, adapter_dir=adapter_dir)

    from src.llm.client import generate as gemini_generate

    def _fallback(prompt: str) -> str:
        return gemini_generate(prompt)

    return _fallback
