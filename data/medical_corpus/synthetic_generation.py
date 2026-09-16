"""
data/medical_corpus/synthetic_generation.py

Generates synthetic hospital-style medical examples (internal protocols,
frequently-asked clinician questions, laudo/receita templates) for the
stroke (AVC) domain, reusing the same Gemini wrapper as src/llm/client.py.

All content produced here is FICTIONAL/SYNTHETIC — there is no real internal
hospital dataset available, so this stands in for "protocolos médicos do
hospital" and "modelos de laudos, receitas e procedimentos internos" as
training material. Raw output must still be run through
data/medical_corpus/preprocessing.curate_examples before being used for
fine-tuning (see build_corpus.py).

Usage:
    python -m data.medical_corpus.synthetic_generation
"""

from __future__ import annotations

import json
import os
import re
import time

from src.llm.client import generate

MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 15

TOPICS = [
    "protocolo de atendimento a AVC isquêmico agudo (janela de trombólise)",
    "protocolo de atendimento a AVC hemorrágico",
    "escala NIHSS: como aplicar e interpretar",
    "critérios de elegibilidade para trombólise endovenosa",
    "cuidados de enfermagem pós-AVC nas primeiras 24 horas",
    "reabilitação precoce após AVC isquêmico",
    "prevenção secundária de AVC (antiagregantes/anticoagulantes)",
    "manejo da pressão arterial na fase aguda do AVC",
    "sinais de alerta (escala FAST) para AVC e orientação a familiares",
    "critérios de transferência para UTI após AVC",
]

_PROMPT_TEMPLATE = """\
Você é um especialista em neurologia vascular ajudando a criar material de \
treinamento SINTÉTICO e FICTÍCIO para um assistente de IA hospitalar sobre \
o tema: "{topic}".

Gere exatamente {n} exemplos, cada um um objeto JSON com as chaves:
- "instruction": uma pergunta que um médico faria sobre o tema, ou um pedido \
de um modelo de documento (ex.: "gere um modelo de laudo para...").
- "input": contexto opcional do paciente (pode ser uma string vazia).
- "output": a resposta correspondente, escrita como um protocolo interno de \
hospital, um laudo-modelo, uma receita-modelo ou uma resposta de FAQ, em \
português.

Não use nomes reais de pacientes, médicos ou hospitais — se precisar de um \
identificador, use "Paciente Fictício A". Deixe claro que é material \
sintético de treinamento, não uma recomendação clínica validada.

Responda APENAS com uma lista JSON válida de {n} objetos, sem texto, \
comentários ou markdown adicional antes ou depois da lista.
"""


def build_generation_prompt(topic: str, n: int = 5) -> str:
    """Pure prompt builder for one topic (no API call)."""
    return _PROMPT_TEMPLATE.format(topic=topic, n=n)


def _parse_examples(raw_text: str) -> list[dict]:
    """
    Parse the LLM's JSON-list response into example dicts. Tolerates the
    model wrapping the list in a ```json ... ``` fenced block or adding
    stray text around it.
    """
    match = re.search(r"\[.*\]", raw_text, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return [ex for ex in data if isinstance(ex, dict)]


def generate_examples_for_topic(
    topic: str, n: int = 5, client=None, model: str | None = None
) -> list[dict]:
    """
    Call the LLM once for one topic and return parsed example dicts
    (unvalidated — run through preprocessing.curate_examples afterwards).

    Retries with a fixed backoff on transient API errors (e.g. 503 model
    overload) so a single flaky call does not abort the whole corpus build;
    returns an empty list (with a warning) if every attempt fails.
    """
    prompt = build_generation_prompt(topic, n=n)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw_text = generate(prompt, client=client, model=model)
            return _parse_examples(raw_text)
        except Exception as exc:  # pragma: no cover - network failure path
            print(f"  [warn] generation failed for topic {topic!r} (attempt {attempt}/{MAX_RETRIES}): {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS)
    print(f"  [warn] giving up on topic {topic!r} after {MAX_RETRIES} attempts")
    return []


def generate_synthetic_corpus(
    topics: list[str] | None = None,
    n_per_topic: int = 5,
    client=None,
    model: str | None = None,
) -> list[dict]:
    """
    Generate raw synthetic examples across all topics (one LLM call per
    topic). Does not curate — call
    data/medical_corpus/preprocessing.curate_examples on the result before
    writing to disk.
    """
    topics = topics or TOPICS
    examples = []
    for topic in topics:
        examples.extend(
            generate_examples_for_topic(topic, n=n_per_topic, client=client, model=model)
        )
    return examples


def _default_output_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "synthetic_raw.jsonl")


if __name__ == "__main__":
    from data.medical_corpus.preprocessing import write_jsonl

    raw_examples = generate_synthetic_corpus()
    out_path = _default_output_path()
    write_jsonl(raw_examples, out_path)
    print(
        f"Generated {len(raw_examples)} raw synthetic examples across "
        f"{len(TOPICS)} topics -> {out_path}"
    )
