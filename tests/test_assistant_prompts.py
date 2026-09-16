from src.assistant.prompts import build_assistant_prompt

PATIENT_RECORD = {"id": 1, "age": 67, "hypertension": 0, "bmi": 36.6}
SOURCES = [{"text": "Trecho do protocolo interno.", "instruction": "q", "source": "synthetic:protocolo-1"}]


def test_build_assistant_prompt_includes_question():
    prompt = build_assistant_prompt("Qual a conduta?", PATIENT_RECORD, SOURCES)
    assert "Qual a conduta?" in prompt


def test_build_assistant_prompt_includes_patient_record_fields():
    prompt = build_assistant_prompt("pergunta", PATIENT_RECORD, SOURCES)
    for key, value in PATIENT_RECORD.items():
        assert key in prompt
        assert str(value) in prompt


def test_build_assistant_prompt_includes_source_text_and_label():
    prompt = build_assistant_prompt("pergunta", PATIENT_RECORD, SOURCES)
    assert "Trecho do protocolo interno." in prompt
    assert "synthetic:protocolo-1" in prompt
    assert "Fonte 1" in prompt


def test_build_assistant_prompt_handles_missing_record():
    prompt = build_assistant_prompt("pergunta", None, [])
    assert "Nenhum registro" in prompt
    assert "Nenhuma fonte" in prompt


def test_build_assistant_prompt_includes_pending_exams():
    prompt = build_assistant_prompt("pergunta", PATIENT_RECORD, SOURCES, pending_exams=["Perfil metabólico / IMC"])
    assert "Perfil metabólico / IMC" in prompt


def test_build_assistant_prompt_instructs_against_direct_prescription():
    prompt = build_assistant_prompt("pergunta", PATIENT_RECORD, SOURCES)
    assert "NUNCA prescreve" in prompt
