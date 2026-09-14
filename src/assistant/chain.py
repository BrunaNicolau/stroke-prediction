"""
src/assistant/chain.py

Orchestrates the medical assistant Q&A pipeline: patient record
(patient_db) + retrieved protocol/FAQ snippets (retriever) + pending exams
(tools) -> grounded prompt (prompts.build_assistant_prompt) -> LLM backend
(llm_backend) -> security guardrail (src/security/guardrails) -> answer +
sources (explainability).

This is a deterministic pipeline, not an LLM-driven agent loop: every step
runs in a fixed order, which keeps it reliable and easy to test end to end.
For the full clinical decision flow (stroke risk prediction + alerting +
audit logging), see src/assistant/graph.py, which reuses these same
building blocks as LangGraph nodes.
"""

from __future__ import annotations

from src.assistant.patient_db import DEFAULT_DB_PATH, get_patient_record
from src.assistant.prompts import build_assistant_prompt
from src.assistant.retriever import retrieve as retrieve_documents
from src.assistant.tools import check_pending_exams
from src.security.guardrails import enforce_human_validation


def answer_question(
    patient_id: int,
    question: str,
    vectorstore,
    generate_fn,
    db_path: str = DEFAULT_DB_PATH,
    k: int = 3,
) -> dict:
    """
    Run the full assistant pipeline for one clinician question about one
    patient.

    Args:
        patient_id:  id in the "prontuarios" table (patient_db.py).
        question:    the clinician's free-text question.
        vectorstore: a FAISS vectorstore from src.assistant.retriever.
        generate_fn: a callable prompt -> str (see llm_backend.get_generate_fn).
        db_path:     patient records DB path (defaults to patient_db.DEFAULT_DB_PATH).
        k:           number of retrieved protocol/FAQ snippets.

    Returns:
        {
            "response": str,
            "sources": list[str],
            "requires_human_validation": bool,
            "guardrail_flags": list[str],
            "patient_record": dict | None,
            "pending_exams": list[str],
        }
    """
    record = get_patient_record(patient_id, db_path=db_path)
    pending_exams = check_pending_exams(record) if record else []
    retrieved = retrieve_documents(vectorstore, question, k=k)

    prompt = build_assistant_prompt(
        question=question,
        patient_record=record,
        retrieved_sources=retrieved,
        pending_exams=pending_exams,
    )
    raw_response = generate_fn(prompt)
    guarded = enforce_human_validation(raw_response)

    return {
        "response": guarded["response"],
        "sources": [s["source"] for s in retrieved],
        "requires_human_validation": guarded["requires_human_validation"],
        "guardrail_flags": guarded["blocked_patterns"],
        "patient_record": record,
        "pending_exams": pending_exams,
    }
