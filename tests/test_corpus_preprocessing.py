from data.medical_corpus.preprocessing import (
    clean_text,
    curate_examples,
    dedupe_examples,
    is_valid_example,
    read_jsonl,
    scrub_pii,
    split_train_eval,
    write_jsonl,
)


def test_scrub_pii_redacts_cpf():
    text = "Paciente CPF 123.456.789-09 foi atendido."
    assert "123.456.789-09" not in scrub_pii(text)
    assert "[CPF_REMOVIDO]" in scrub_pii(text)


def test_scrub_pii_redacts_email():
    text = "Contato: medico@hospital.com para duvidas."
    result = scrub_pii(text)
    assert "medico@hospital.com" not in result
    assert "[EMAIL_REMOVIDO]" in result


def test_scrub_pii_redacts_date():
    text = "Internado em 05/03/2024 as 10h."
    result = scrub_pii(text)
    assert "05/03/2024" not in result
    assert "[DATA_REMOVIDA]" in result


def test_scrub_pii_handles_none():
    assert scrub_pii(None) == ""


def test_clean_text_collapses_whitespace():
    assert clean_text("  algo   com\n\nespacos  ") == "algo com espacos"


def test_clean_text_handles_none():
    assert clean_text(None) == ""


def test_is_valid_example_rejects_empty_instruction():
    assert not is_valid_example("", "resposta valida com tamanho suficiente")


def test_is_valid_example_rejects_short_output():
    assert not is_valid_example("pergunta", "curto")


def test_is_valid_example_rejects_too_long_output():
    assert not is_valid_example("pergunta", "x" * 5000)


def test_is_valid_example_accepts_reasonable_pair():
    assert is_valid_example("pergunta valida", "resposta com tamanho razoavel para ser aceita")


def test_dedupe_examples_removes_exact_duplicates():
    examples = [
        {"instruction": "a", "input": "", "output": "b"},
        {"instruction": "a", "input": "", "output": "b"},
        {"instruction": "c", "input": "", "output": "d"},
    ]
    result = dedupe_examples(examples)
    assert len(result) == 2


def test_curate_examples_scrubs_and_filters():
    raw = [
        {
            "instruction": "Qual o protocolo?",
            "input": "",
            "output": "Contato do medico: doc@hospital.com para mais detalhes sobre o protocolo interno.",
        },
        {"instruction": "", "input": "", "output": "sem instrucao, deve ser descartado"},
        {"instruction": "curto", "input": "", "output": "no"},
    ]
    curated = curate_examples(raw)
    assert len(curated) == 1
    assert "doc@hospital.com" not in curated[0]["output"]
    assert "[EMAIL_REMOVIDO]" in curated[0]["output"]


def test_curate_examples_preserves_extra_fields():
    raw = [{"instruction": "q", "input": "", "output": "resposta com tamanho suficiente aqui", "source": "MedQuAD:1"}]
    curated = curate_examples(raw)
    assert curated[0]["source"] == "MedQuAD:1"


def test_write_and_read_jsonl_roundtrip(tmp_path):
    examples = [{"instruction": "a", "input": "", "output": "b"}]
    path = tmp_path / "out.jsonl"
    write_jsonl(examples, str(path))
    assert read_jsonl(str(path)) == examples


def test_split_train_eval_is_deterministic():
    examples = [{"instruction": str(i), "input": "", "output": str(i)} for i in range(20)]
    train_a, eval_a = split_train_eval(examples, eval_ratio=0.2, random_state=42)
    train_b, eval_b = split_train_eval(examples, eval_ratio=0.2, random_state=42)
    assert train_a == train_b
    assert eval_a == eval_b


def test_split_train_eval_respects_ratio():
    examples = [{"instruction": str(i), "input": "", "output": str(i)} for i in range(20)]
    train, eval_ = split_train_eval(examples, eval_ratio=0.2)
    assert len(eval_) == 4
    assert len(train) == 16


def test_split_train_eval_handles_empty_list():
    assert split_train_eval([]) == ([], [])
