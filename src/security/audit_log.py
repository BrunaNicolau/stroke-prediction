"""
src/security/audit_log.py

Structured audit logging for every assistant interaction, using Python's
stdlib `logging` module writing JSON lines. Required by the Fase 3
challenge ("implementar logging detalhado para rastreamento e auditoria").
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_LOG_PATH = os.path.join(_PROJECT_ROOT, "results", "audit_log.jsonl")


def pseudonymize_patient_id(patient_id) -> str:
    """
    One-way pseudonymization of a patient identifier for the audit log, so
    raw patient IDs are never written to a log file that could end up
    outside the trusted boundary.
    """
    return hashlib.sha256(str(patient_id).encode("utf-8")).hexdigest()[:16]


def _get_file_logger(log_path: str) -> logging.Logger:
    """
    Return a logger with exactly one FileHandler pointed at log_path. Keyed
    by the absolute path so tests using different tmp_path files (or a
    long-running process switching paths) each get their own handler
    without duplicating log lines.
    """
    abs_path = os.path.abspath(log_path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    logger = logging.getLogger(f"assistant.audit.{abs_path}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.FileHandler(abs_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    return logger


def log_interaction(
    patient_id,
    question: str,
    response: str,
    sources: list[str] | None = None,
    requires_human_validation: bool = False,
    guardrail_flags: list[str] | None = None,
    log_path: str = DEFAULT_LOG_PATH,
) -> dict:
    """
    Append one structured audit entry (as a JSON line) to log_path.

    Returns the entry dict that was written, e.g.:
        {
            "timestamp": "2026-09-13T12:00:00+00:00",
            "patient_id": "a1b2c3...",  # pseudonymized
            "question": "...",
            "response": "...",
            "sources": ["MedQuAD:...", "protocolo:..."],
            "requires_human_validation": False,
            "guardrail_flags": [],
        }
    """
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "patient_id": pseudonymize_patient_id(patient_id),
        "question": question,
        "response": response,
        "sources": sources or [],
        "requires_human_validation": requires_human_validation,
        "guardrail_flags": guardrail_flags or [],
    }
    logger = _get_file_logger(log_path)
    logger.info(json.dumps(entry, ensure_ascii=False))
    return entry


def read_audit_log(log_path: str = DEFAULT_LOG_PATH) -> list[dict]:
    """Read back all entries from an audit log JSONL file (empty list if it doesn't exist yet)."""
    if not os.path.exists(log_path):
        return []
    with open(log_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
