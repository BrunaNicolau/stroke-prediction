"""
src/security/guardrails.py

Pure, deterministic safety checks for assistant responses. The Fase 3
challenge requires the assistant to never prescribe directly without human
validation — these functions detect that pattern and attach the required
disclaimer/flag, instead of letting a raw LLM response reach a clinician
unfiltered.
"""

from __future__ import annotations

import re

_PRESCRIPTION_PATTERNS = (
    re.compile(r"\btome\b", re.IGNORECASE),
    re.compile(r"\bprescrevo\b", re.IGNORECASE),
    re.compile(r"\bprescreva\b", re.IGNORECASE),
    re.compile(r"\baplique\s+\d", re.IGNORECASE),
    re.compile(r"\badministre\s+\d", re.IGNORECASE),
    re.compile(r"\b\d+\s?mg\b", re.IGNORECASE),
)

HUMAN_VALIDATION_DISCLAIMER = (
    "\n\n⚠️ Esta sugestão foi gerada por um assistente de IA e NÃO substitui "
    "o julgamento clínico. Requer validação humana por um médico responsável "
    "antes de qualquer conduta ou prescrição."
)


def contains_direct_prescription(text: str) -> bool:
    """True if `text` reads like a direct prescription order (dosage/imperative verb)."""
    return any(pattern.search(text) for pattern in _PRESCRIPTION_PATTERNS)


def enforce_human_validation(response: str) -> dict:
    """
    Apply the "never prescribe directly without human validation" guardrail
    to one assistant response.

    Returns:
        {
            "response": str,                      # original text (+ disclaimer if flagged)
            "requires_human_validation": bool,
            "blocked_patterns": list[str],         # matched regex patterns, if any
        }
    """
    matched = [p.pattern for p in _PRESCRIPTION_PATTERNS if p.search(response)]
    requires_validation = bool(matched)
    annotated = response + HUMAN_VALIDATION_DISCLAIMER if requires_validation else response
    return {
        "response": annotated,
        "requires_human_validation": requires_validation,
        "blocked_patterns": matched,
    }
