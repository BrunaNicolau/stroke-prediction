from src.security.guardrails import (
    HUMAN_VALIDATION_DISCLAIMER,
    contains_direct_prescription,
    enforce_human_validation,
)


def test_contains_direct_prescription_detects_imperative_verb():
    assert contains_direct_prescription("Tome 500mg de dipirona a cada 8 horas.")


def test_contains_direct_prescription_detects_dosage_pattern():
    assert contains_direct_prescription("Administre 10mg de enalapril.")


def test_contains_direct_prescription_false_for_safe_suggestion():
    text = "Considere avaliar o paciente para trombolise, sujeito ao protocolo interno."
    assert not contains_direct_prescription(text)


def test_enforce_human_validation_flags_and_appends_disclaimer():
    result = enforce_human_validation("Tome 500mg de AAS agora.")
    assert result["requires_human_validation"] is True
    assert HUMAN_VALIDATION_DISCLAIMER in result["response"]
    assert result["blocked_patterns"]


def test_enforce_human_validation_leaves_safe_text_untouched():
    safe_text = "Considere avaliar trombolise conforme protocolo interno."
    result = enforce_human_validation(safe_text)
    assert result["requires_human_validation"] is False
    assert result["response"] == safe_text
    assert result["blocked_patterns"] == []
