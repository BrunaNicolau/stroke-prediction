"""
data/medical_corpus/preprocessing.py

Pure, dependency-light functions to clean, anonymize, curate and format
medical text examples into the instruction-tuning JSONL format consumed by
notebooks/04_finetuning.ipynb and src/finetuning/dataset.py.

None of these functions call an API or touch the network — they only
transform in-memory data, which keeps them cheap and deterministic to test
(mirrors the "pure function" style of src/llm/prompts.py and
src/llm/evaluation.py).
"""

from __future__ import annotations

import json
import random
import re
from typing import Iterable

MIN_OUTPUT_LENGTH = 20
MAX_OUTPUT_LENGTH = 4000

# Deliberately simple, deterministic regex patterns. Good enough as a
# sanity-check redaction pass over synthetic/public text — NOT a substitute
# for manual review before any real internal hospital data is ever used.
_PII_PATTERNS = [
    (re.compile(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b"), "[CPF_REMOVIDO]"),
    (re.compile(r"\b\d{2}/\d{2}/\d{4}\b"), "[DATA_REMOVIDA]"),
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "[EMAIL_REMOVIDO]"),
    (re.compile(r"\(\d{2}\)\s?\d{4,5}-\d{4}"), "[TELEFONE_REMOVIDO]"),
]


def scrub_pii(text: str) -> str:
    """
    Replace common PII patterns (CPF, dates, e-mails, phone numbers) with
    redaction placeholders.
    """
    text = text or ""
    for pattern, replacement in _PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def clean_text(text: str) -> str:
    """Collapse whitespace and strip leading/trailing space."""
    return re.sub(r"\s+", " ", text or "").strip()


def is_valid_example(instruction: str, output: str) -> bool:
    """
    Curation filter: reject empty instructions or outputs that are too
    short (likely noise) or too long (likely truncated/garbled generation).
    """
    if not instruction or not output:
        return False
    return MIN_OUTPUT_LENGTH <= len(output) <= MAX_OUTPUT_LENGTH


def dedupe_examples(examples: Iterable[dict]) -> list[dict]:
    """
    Remove exact-duplicate examples (by instruction+input+output),
    preserving first-occurrence order.
    """
    seen = set()
    result = []
    for ex in examples:
        key = (ex.get("instruction", ""), ex.get("input", ""), ex.get("output", ""))
        if key in seen:
            continue
        seen.add(key)
        result.append(ex)
    return result


def curate_examples(examples: Iterable[dict]) -> list[dict]:
    """
    Full curation pipeline for example dicts shaped like
    {"instruction": ..., "input": ..., "output": ..., ...}:
    scrub PII -> clean whitespace -> validate -> dedupe.

    Any extra keys (e.g. "source", used later for RAG citations) are kept
    as-is alongside the cleaned instruction/input/output. Returns the
    filtered/cleaned list; anything failing is_valid_example is dropped.
    """
    curated = []
    for ex in examples:
        instruction = clean_text(scrub_pii(ex.get("instruction", "")))
        input_ = clean_text(scrub_pii(ex.get("input", "")))
        output = clean_text(scrub_pii(ex.get("output", "")))
        if not is_valid_example(instruction, output):
            continue
        cleaned = dict(ex)
        cleaned.update({"instruction": instruction, "input": input_, "output": output})
        curated.append(cleaned)
    return dedupe_examples(curated)


def write_jsonl(examples: list[dict], path: str) -> None:
    """Write a list of example dicts to a JSONL file (one JSON object per line)."""
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> list[dict]:
    """Read a JSONL file into a list of dicts."""
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def split_train_eval(
    examples: list[dict], eval_ratio: float = 0.15, random_state: int = 42
) -> tuple[list[dict], list[dict]]:
    """
    Deterministic shuffled train/eval split. Uses the stdlib `random` module
    (not scikit-learn) to keep this module dependency-free.

    Returns (train_examples, eval_examples).
    """
    if not examples:
        return [], []
    rng = random.Random(random_state)
    shuffled = examples[:]
    rng.shuffle(shuffled)
    n_eval = max(1, int(len(shuffled) * eval_ratio))
    return shuffled[n_eval:], shuffled[:n_eval]
