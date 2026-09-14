from src.finetuning.dataset import format_example, format_prompt_only, load_split

EXAMPLE = {"instruction": "O que é AVC?", "input": "", "output": "É um acidente vascular cerebral."}
EXAMPLE_WITH_CONTEXT = {
    "instruction": "Qual a conduta?",
    "input": "Paciente com NIHSS 12.",
    "output": "Considere avaliar trombolise.",
}


def test_format_example_includes_instruction_and_output():
    text = format_example(EXAMPLE)
    assert EXAMPLE["instruction"] in text
    assert EXAMPLE["output"] in text


def test_format_example_omits_context_block_when_no_input():
    text = format_example(EXAMPLE)
    assert "### Contexto:" not in text


def test_format_example_includes_context_block_when_input_present():
    text = format_example(EXAMPLE_WITH_CONTEXT)
    assert "### Contexto:" in text
    assert "NIHSS 12" in text


def test_format_prompt_only_excludes_output():
    text = format_prompt_only(EXAMPLE)
    assert EXAMPLE["instruction"] in text
    assert EXAMPLE["output"] not in text


def test_load_split_reads_jsonl(tmp_path):
    from data.medical_corpus.preprocessing import write_jsonl

    path = tmp_path / "split.jsonl"
    write_jsonl([EXAMPLE], str(path))

    loaded = load_split(str(path))
    assert loaded == [EXAMPLE]
