"""
data/medical_corpus/build_corpus.py

Assembles the medical fine-tuning / RAG corpus for the stroke (AVC) domain:

1. Pulls stroke-related Q&A pairs from the two public datasets suggested by
   the challenge (MedQuAD, PubMedQA), via the Hugging Face `datasets-server`
   full-text search REST API — no local dataset download needed, only the
   stdlib `urllib` (keeps this script dependency-free to run).
2. Combines them with the synthetic hospital-style examples produced by
   synthetic_generation.py (protocols, FAQs, laudo/receita models).
3. Runs everything through preprocessing.curate_examples (PII scrub,
   cleanup, validation, dedupe).
4. Splits into train/eval and writes data/medical_corpus/train.jsonl and
   data/medical_corpus/eval.jsonl.

Usage:
    python -m data.medical_corpus.build_corpus
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request

from data.medical_corpus.preprocessing import (
    curate_examples,
    read_jsonl,
    split_train_eval,
    write_jsonl,
)

_SEARCH_URL = "https://datasets-server.huggingface.co/search"
_CORPUS_DIR = os.path.dirname(os.path.abspath(__file__))

MEDQUAD_QUERIES = ["stroke"]
PUBMEDQA_QUERIES = ["stroke"]


def _search(dataset: str, config: str, query: str, length: int = 100, max_rows: int = 1000) -> list[dict]:
    """
    Page through the datasets-server /search endpoint for one dataset
    config and query string. Returns raw row dicts.

    Network/availability errors are caught and logged rather than raised,
    so a temporarily unreachable public dataset does not crash the whole
    corpus build (the synthetic examples alone are still usable).
    """
    rows: list[dict] = []
    offset = 0
    while offset < max_rows:
        params = urllib.parse.urlencode(
            {
                "dataset": dataset,
                "config": config,
                "split": "train",
                "query": query,
                "offset": offset,
                "length": length,
            }
        )
        url = f"{_SEARCH_URL}?{params}"
        try:
            with urllib.request.urlopen(url, timeout=40) as resp:
                data = json.load(resp)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            print(f"  [warn] search failed for {dataset}/{config} query={query!r}: {exc}")
            break
        batch = [r["row"] for r in data.get("rows", [])]
        rows.extend(batch)
        total = data.get("num_rows_total") or 0
        offset += length
        if offset >= total or not batch:
            break
    return rows


def fetch_medquad_examples() -> list[dict]:
    """Fetch stroke-related rows from lavita/MedQuAD and reshape them."""
    seen_ids = set()
    examples = []
    for query in MEDQUAD_QUERIES:
        for row in _search("lavita/MedQuAD", "default", query):
            qid = row.get("question_id")
            if qid in seen_ids:
                continue
            seen_ids.add(qid)
            examples.append(
                {
                    "instruction": row.get("question", ""),
                    "input": "",
                    "output": row.get("answer", ""),
                    "source": f"MedQuAD:{row.get('document_source', '')}:{qid}",
                }
            )
    print(f"MedQuAD: {len(examples)} stroke-related examples")
    return examples


def fetch_pubmedqa_examples() -> list[dict]:
    """Fetch stroke-related rows from qiaojin/PubMedQA (pqa_labeled config)."""
    seen_ids = set()
    examples = []
    for query in PUBMEDQA_QUERIES:
        for row in _search("qiaojin/PubMedQA", "pqa_labeled", query):
            pubid = row.get("pubid")
            if pubid in seen_ids:
                continue
            seen_ids.add(pubid)
            examples.append(
                {
                    "instruction": row.get("question", ""),
                    "input": "",
                    "output": row.get("long_answer", ""),
                    "source": f"PubMedQA:{pubid}",
                }
            )
    print(f"PubMedQA: {len(examples)} stroke-related examples")
    return examples


def load_synthetic_examples() -> list[dict]:
    """Load the raw synthetic examples produced by synthetic_generation.py, if present."""
    path = os.path.join(_CORPUS_DIR, "synthetic_raw.jsonl")
    if not os.path.exists(path):
        print(
            f"  [info] {path} not found — run "
            "'python -m data.medical_corpus.synthetic_generation' first "
            "for synthetic protocol/laudo examples."
        )
        return []
    examples = read_jsonl(path)
    for ex in examples:
        ex.setdefault("source", "synthetic")
    print(f"Synthetic: {len(examples)} raw examples")
    return examples


def build_corpus(
    include_public: bool = True, include_synthetic: bool = True
) -> tuple[list[dict], list[dict]]:
    """
    Assemble, curate and split the full corpus.

    Returns (train_examples, eval_examples), each a list of
    {"instruction", "input", "output"} dicts ready for
    src/finetuning/dataset.py.
    """
    raw_examples: list[dict] = []
    if include_public:
        raw_examples.extend(fetch_medquad_examples())
        raw_examples.extend(fetch_pubmedqa_examples())
    if include_synthetic:
        raw_examples.extend(load_synthetic_examples())

    curated = curate_examples(raw_examples)
    print(f"Curated: {len(curated)} examples (from {len(raw_examples)} raw)")

    train, eval_ = split_train_eval(curated)
    print(f"Split: {len(train)} train / {len(eval_)} eval")
    return train, eval_


if __name__ == "__main__":
    train_examples, eval_examples = build_corpus()
    train_path = os.path.join(_CORPUS_DIR, "train.jsonl")
    eval_path = os.path.join(_CORPUS_DIR, "eval.jsonl")
    write_jsonl(train_examples, train_path)
    write_jsonl(eval_examples, eval_path)
    print(f"Wrote {train_path} and {eval_path}")
