import json

from src.security.audit_log import log_interaction, pseudonymize_patient_id, read_audit_log


def test_pseudonymize_patient_id_is_deterministic_and_hides_raw_id():
    hashed = pseudonymize_patient_id(9046)
    assert hashed != "9046"
    assert hashed == pseudonymize_patient_id(9046)
    assert hashed != pseudonymize_patient_id(9047)


def test_log_interaction_writes_jsonl_entry(tmp_path):
    log_path = tmp_path / "audit.jsonl"

    entry = log_interaction(
        patient_id=9046,
        question="Qual a conduta?",
        response="Considere avaliar.",
        sources=["MedQuAD:1"],
        requires_human_validation=False,
        guardrail_flags=[],
        log_path=str(log_path),
    )

    assert log_path.exists()
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    persisted = json.loads(lines[0])
    assert persisted == entry
    assert persisted["patient_id"] == pseudonymize_patient_id(9046)
    assert persisted["sources"] == ["MedQuAD:1"]


def test_log_interaction_appends_multiple_entries(tmp_path):
    log_path = tmp_path / "audit.jsonl"

    log_interaction(1, "q1", "r1", log_path=str(log_path))
    log_interaction(2, "q2", "r2", log_path=str(log_path))

    entries = read_audit_log(str(log_path))
    assert len(entries) == 2
    assert entries[0]["question"] == "q1"
    assert entries[1]["question"] == "q2"


def test_read_audit_log_returns_empty_list_when_missing(tmp_path):
    missing_path = tmp_path / "does_not_exist.jsonl"
    assert read_audit_log(str(missing_path)) == []
