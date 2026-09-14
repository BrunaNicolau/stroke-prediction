"""
src/assistant/prompts.py

Pure prompt-building functions for the LangChain medical assistant. No API
calls here — only text assembly, kept cheap and deterministic to test
(same pattern as src/llm/prompts.py).
"""

from __future__ import annotations


def _format_record_block(record: dict | None) -> str:
    if not record:
        return "Nenhum registro de prontuário encontrado para este paciente."
    return "\n".join(f"- {k}: {v}" for k, v in record.items())


def _format_sources_block(sources: list[dict]) -> str:
    if not sources:
        return "Nenhuma fonte de protocolo/FAQ relevante encontrada."
    blocks = [
        f"[Fonte {i} — {s.get('source', 'desconhecida')}]\n{s.get('text', '')}"
        for i, s in enumerate(sources, start=1)
    ]
    return "\n\n".join(blocks)


def build_assistant_prompt(
    question: str,
    patient_record: dict | None,
    retrieved_sources: list[dict],
    pending_exams: list[str] | None = None,
) -> str:
    """
    Build the grounded prompt for the medical assistant: patient record +
    retrieved protocol/FAQ snippets + pending exams + the clinician's
    question. Instructs the model to ground its answer only in the
    provided context, cite sources, and never issue a direct prescription.
    """
    record_block = _format_record_block(patient_record)
    sources_block = _format_sources_block(retrieved_sources)
    exams_block = ", ".join(pending_exams) if pending_exams else "Nenhum exame pendente identificado."

    return (
        "Você é um assistente clínico de apoio à decisão em um hospital, "
        "treinado com protocolos internos fictícios sobre AVC (Acidente "
        "Vascular Cerebral). Você NUNCA prescreve medicação ou conduta "
        "diretamente — apenas sugere possibilidades para avaliação de um "
        "médico responsável, que deve validar antes de qualquer ação.\n\n"
        "Dados do prontuário do paciente:\n"
        f"{record_block}\n\n"
        f"Exames pendentes: {exams_block}\n\n"
        "Trechos de protocolos/FAQs internos relevantes:\n"
        f"{sources_block}\n\n"
        f"Pergunta do médico: {question}\n\n"
        "Responda em português, citando explicitamente qual fonte (ex.: "
        "'Fonte 1') embasou cada parte da sua resposta. Use apenas as "
        "informações fornecidas acima — não invente exames, históricos ou "
        "protocolos que não foram passados. Use linguagem de sugestão "
        "('considere', 'pode ser avaliado'), nunca de prescrição direta."
    )
