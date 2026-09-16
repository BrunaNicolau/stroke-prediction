from unittest.mock import patch

import pytest

from src.assistant.graph import receive_patient_data, run_flow

PATIENT_RECORD = {"id": 1, "age": 67, "hypertension": 0, "bmi": 36.6, "smoking_status": "formerly smoked"}


def _safe_generate(prompt: str) -> str:
    return "Considere avaliar trombolise conforme protocolo interno."


def _unsafe_generate(prompt: str) -> str:
    return "Administre 10mg de enalapril agora."


@pytest.fixture
def graph_mocks():
    """
    Patch every external dependency the graph touches (DB, ML model,
    retriever, audit log) so tests exercise only the routing logic — no
    real SQLite file, trained model, or embeddings needed.
    """
    with (
        patch("src.assistant.graph.get_patient_record") as mock_get_record,
        patch("src.assistant.graph.predict_stroke_risk") as mock_predict,
        patch("src.assistant.graph.retrieve_documents") as mock_retrieve,
        patch("src.assistant.graph.log_interaction") as mock_log,
    ):
        mock_get_record.return_value = PATIENT_RECORD
        mock_retrieve.return_value = [{"text": "trecho", "instruction": "q", "source": "synthetic:1"}]
        mock_log.return_value = {"logged": True}
        yield {"get_record": mock_get_record, "predict": mock_predict, "retrieve": mock_retrieve, "log": mock_log}


def test_receive_patient_data_requires_patient_id_and_question():
    with pytest.raises(ValueError):
        receive_patient_data({"question": "q"})
    with pytest.raises(ValueError):
        receive_patient_data({"patient_id": 1})


def test_high_risk_prediction_triggers_alert(graph_mocks):
    graph_mocks["predict"].return_value = {"prediction": 1, "probability": 0.8}

    result = run_flow(patient_id=1, question="Quais os proximos passos?", vectorstore=None, generate_fn=_safe_generate)

    assert result["alert"] is True
    assert "risco de AVC alto" in result["alert_reason"]
    assert result["requires_human_validation"] is False
    graph_mocks["log"].assert_called_once()


def test_low_risk_and_safe_response_does_not_alert(graph_mocks):
    graph_mocks["predict"].return_value = {"prediction": 0, "probability": 0.1}

    result = run_flow(patient_id=1, question="Este paciente precisa de trombolise?", vectorstore=None, generate_fn=_safe_generate)

    assert not result.get("alert")
    assert result["requires_human_validation"] is False


def test_guardrail_triggers_alert_even_with_low_risk(graph_mocks):
    graph_mocks["predict"].return_value = {"prediction": 0, "probability": 0.1}

    result = run_flow(patient_id=1, question="O que fazer?", vectorstore=None, generate_fn=_unsafe_generate)

    assert result["requires_human_validation"] is True
    assert result["alert"] is True
    assert "guardrail" in result["alert_reason"]


def test_missing_patient_record_still_completes_flow(graph_mocks):
    graph_mocks["get_record"].return_value = None

    result = run_flow(patient_id=999, question="pergunta", vectorstore=None, generate_fn=_safe_generate)

    assert result["patient_record"] is None
    assert result["pending_exams"] == []
    assert result["stroke_risk"] == {"prediction": None, "probability": None}
    assert not result.get("alert")
    graph_mocks["log"].assert_called_once()


def test_audit_log_receives_sources_and_flags(graph_mocks):
    graph_mocks["predict"].return_value = {"prediction": 0, "probability": 0.1}

    run_flow(patient_id=1, question="pergunta", vectorstore=None, generate_fn=_unsafe_generate)

    _, kwargs = graph_mocks["log"].call_args
    assert kwargs["sources"] == ["synthetic:1"]
    assert kwargs["requires_human_validation"] is True
    assert kwargs["guardrail_flags"]
