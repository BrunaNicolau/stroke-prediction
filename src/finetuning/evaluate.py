"""
src/finetuning/evaluate.py

Deterministic, non-GPU evaluation helpers for the fine-tuned adapter.
Mirrors the spirit of src/llm/evaluation.py: a reproducible checklist
instead of an LLM-as-judge, so it runs in CI without a GPU. Reuses the same
prescription-language check as src/security/guardrails.py so both places
agree on what "sounds like a direct prescription" means.
"""

from __future__ import annotations

import re

from src.security.guardrails import contains_direct_prescription


def mentions_key_terms(generated_text: str, expected_output: str, min_overlap: int = 1) -> bool:
    """
    Lightweight grounding check: at least `min_overlap` significant words
    (length > 4) from the expected answer also appear in the generated
    text. Not semantic similarity — a cheap sanity signal only, same
    determinism trade-off as src/llm/evaluation.py's checklist.
    """
    expected_terms = {w.lower() for w in re.findall(r"\w{5,}", expected_output)}
    generated_terms = {w.lower() for w in re.findall(r"\w{5,}", generated_text)}
    return len(expected_terms & generated_terms) >= min_overlap


def avoids_direct_prescription(generated_text: str) -> bool:
    """True if the generated text does not read like a direct prescription order."""
    return not contains_direct_prescription(generated_text)


def reasonable_length(generated_text: str, min_len: int = 20, max_len: int = 4000) -> bool:
    return min_len <= len(generated_text) <= max_len


def evaluate_generation(generated_text: str, expected_output: str) -> dict:
    """
    Deterministic checklist for one (generated, expected) pair.

    Returns {"score": float in [0, 1], "checks": {rule_name: bool}}.
    """
    checks = {
        "mentions_key_terms": mentions_key_terms(generated_text, expected_output),
        "avoids_direct_prescription": avoids_direct_prescription(generated_text),
        "reasonable_length": reasonable_length(generated_text),
    }
    score = sum(checks.values()) / len(checks)
    return {"score": score, "checks": checks}


def evaluate_eval_set(pairs: list[tuple[str, str]]) -> dict:
    """
    Aggregate evaluate_generation over a list of (generated_text,
    expected_output) pairs, e.g. produced by running the fine-tuned model
    over data/medical_corpus/eval.jsonl.

    Returns {"mean_score": float, "results": [per-pair dicts]}.
    """
    results = [evaluate_generation(gen, exp) for gen, exp in pairs]
    mean_score = sum(r["score"] for r in results) / len(results) if results else 0.0
    return {"mean_score": mean_score, "results": results}
