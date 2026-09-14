"""
src/finetuning/dataset.py

Pure functions to turn the curated corpus (data/medical_corpus/train.jsonl,
eval.jsonl) into the prompt format used to fine-tune the base LLM. No
torch/transformers/datasets import happens at module load time — only
inside build_hf_dataset(), so this module stays importable (and testable)
without the heavy ML stack installed.
"""

from __future__ import annotations

from data.medical_corpus.preprocessing import read_jsonl

PROMPT_TEMPLATE = "### Instrução:\n{instruction}\n\n{input_block}### Resposta:\n{output}"

PROMPT_TEMPLATE_NO_OUTPUT = "### Instrução:\n{instruction}\n\n{input_block}### Resposta:\n"


def _input_block(example: dict) -> str:
    context = example.get("input")
    return f"### Contexto:\n{context}\n\n" if context else ""


def format_example(example: dict) -> str:
    """
    Format one {"instruction", "input", "output"} example into the plain
    text used for supervised fine-tuning (instruction + context + answer).
    """
    return PROMPT_TEMPLATE.format(
        instruction=example["instruction"],
        input_block=_input_block(example),
        output=example["output"],
    )


def format_prompt_only(example: dict) -> str:
    """
    Same shape as format_example but WITHOUT the answer — the prompt the
    fine-tuned model should complete at inference/evaluation time.
    """
    return PROMPT_TEMPLATE_NO_OUTPUT.format(
        instruction=example["instruction"],
        input_block=_input_block(example),
    )


def load_split(path: str) -> list[dict]:
    """Load a JSONL split (train.jsonl or eval.jsonl) as example dicts."""
    return read_jsonl(path)


def build_hf_dataset(examples: list[dict]):
    """
    Convert example dicts into a Hugging Face `datasets.Dataset` with a
    single "text" column (the formatted training string), ready for
    `trl.SFTTrainer` / `transformers.Trainer` in notebooks/04_finetuning.ipynb.

    Imports `datasets` lazily so importing this module does not require the
    heavy ML stack to be installed.
    """
    from datasets import Dataset

    texts = [format_example(ex) for ex in examples]
    return Dataset.from_dict({"text": texts})
