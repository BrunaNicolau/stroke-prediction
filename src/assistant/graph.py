"""
src/assistant/graph.py

LangGraph implementation of the automated clinical decision flow required
by the Fase 3 challenge:

    receive_patient_data -> query_patient_record -> check_pending_exams
    -> run_stroke_prediction (reuses the Fase 1/2 model, src/models.py)
    -> suggest_conduct (LLM + RAG, src/assistant/chain.py building blocks)
    -> apply_guardrails (src/security/guardrails.py)
    -> [conditional] emit_alert (high risk OR guardrail triggered) -> audit_log
                      \\_______________________________________________/
                       (no alert needed) ------------------------------> audit_log

audit_log always runs, regardless of which branch was taken.

Each node is a plain function over a TypedDict state, so it is easy to
unit test node-by-node (see tests/test_graph_flow.py) with a fake
generate_fn/vectorstore injected through the state — no real LLM/GPU
needed to test the routing logic.
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, StateGraph

from src.assistant.patient_db import DEFAULT_DB_PATH, get_patient_record
from src.assistant.prompts import build_assistant_prompt
from src.assistant.retriever import retrieve as retrieve_documents
from src.assistant.tools import check_pending_exams, predict_stroke_risk
from src.security.audit_log import log_interaction
from src.security.guardrails import enforce_human_validation

HIGH_RISK_THRESHOLD = 0.5


class AssistantState(TypedDict, total=False):
    # Inputs
    patient_id: int
    question: str
    db_path: str
    vectorstore: object
    generate_fn: object
    model_name: str
    k: int

    # Populated as the flow runs
    patient_record: dict | None
    pending_exams: list
    stroke_risk: dict
    retrieved_sources: list
    response: str
    requires_human_validation: bool
    guardrail_flags: list
    alert: bool
    alert_reason: str
    audit_entry: dict


def receive_patient_data(state: AssistantState) -> AssistantState:
    """Entry node: validates the minimal required inputs are present."""
    if not state.get("patient_id") or not state.get("question"):
        raise ValueError("patient_id and question are required to run the flow.")
    return state


def query_patient_record(state: AssistantState) -> AssistantState:
    record = get_patient_record(state["patient_id"], db_path=state.get("db_path", DEFAULT_DB_PATH))
    return {**state, "patient_record": record}


def check_pending_exams_node(state: AssistantState) -> AssistantState:
    record = state.get("patient_record")
    return {**state, "pending_exams": check_pending_exams(record) if record else []}


def run_stroke_prediction_node(state: AssistantState) -> AssistantState:
    record = state.get("patient_record")
    if not record:
        return {**state, "stroke_risk": {"prediction": None, "probability": None}}
    risk = predict_stroke_risk(record, model_name=state.get("model_name", "logistic_regression"))
    return {**state, "stroke_risk": risk}


def suggest_conduct(state: AssistantState) -> AssistantState:
    retrieved = retrieve_documents(state["vectorstore"], state["question"], k=state.get("k", 3))
    prompt = build_assistant_prompt(
        question=state["question"],
        patient_record=state.get("patient_record"),
        retrieved_sources=retrieved,
        pending_exams=state.get("pending_exams"),
    )
    raw_response = state["generate_fn"](prompt)
    return {**state, "response": raw_response, "retrieved_sources": retrieved}


def apply_guardrails(state: AssistantState) -> AssistantState:
    guarded = enforce_human_validation(state["response"])
    return {
        **state,
        "response": guarded["response"],
        "requires_human_validation": guarded["requires_human_validation"],
        "guardrail_flags": guarded["blocked_patterns"],
    }


def _route_after_guardrails(state: AssistantState) -> str:
    """Conditional edge: only visit emit_alert when there's something to alert on."""
    risk = state.get("stroke_risk") or {}
    high_risk = (risk.get("probability") or 0) >= HIGH_RISK_THRESHOLD
    guardrail_triggered = bool(state.get("requires_human_validation"))
    return "emit_alert" if (high_risk or guardrail_triggered) else "audit_log"


def emit_alert(state: AssistantState) -> AssistantState:
    risk = state.get("stroke_risk") or {}
    high_risk = (risk.get("probability") or 0) >= HIGH_RISK_THRESHOLD
    guardrail_triggered = bool(state.get("requires_human_validation"))

    reasons = []
    if high_risk:
        reasons.append(f"risco de AVC alto (probabilidade={risk.get('probability'):.2f})")
    if guardrail_triggered:
        reasons.append("guardrail de prescrição direta acionado")

    return {**state, "alert": True, "alert_reason": "; ".join(reasons)}


def write_audit_log(state: AssistantState) -> AssistantState:
    entry = log_interaction(
        patient_id=state["patient_id"],
        question=state["question"],
        response=state.get("response", ""),
        sources=[s["source"] for s in state.get("retrieved_sources", [])],
        requires_human_validation=state.get("requires_human_validation", False),
        guardrail_flags=state.get("guardrail_flags", []),
    )
    return {**state, "audit_entry": entry, "alert": state.get("alert", False)}


def build_flow():
    """Compile the LangGraph StateGraph implementing the clinical decision flow."""
    graph = StateGraph(AssistantState)

    graph.add_node("receive_patient_data", receive_patient_data)
    graph.add_node("query_patient_record", query_patient_record)
    graph.add_node("check_pending_exams", check_pending_exams_node)
    graph.add_node("run_stroke_prediction", run_stroke_prediction_node)
    graph.add_node("suggest_conduct", suggest_conduct)
    graph.add_node("apply_guardrails", apply_guardrails)
    graph.add_node("emit_alert", emit_alert)
    graph.add_node("audit_log", write_audit_log)

    graph.set_entry_point("receive_patient_data")
    graph.add_edge("receive_patient_data", "query_patient_record")
    graph.add_edge("query_patient_record", "check_pending_exams")
    graph.add_edge("check_pending_exams", "run_stroke_prediction")
    graph.add_edge("run_stroke_prediction", "suggest_conduct")
    graph.add_edge("suggest_conduct", "apply_guardrails")
    graph.add_conditional_edges(
        "apply_guardrails",
        _route_after_guardrails,
        {"emit_alert": "emit_alert", "audit_log": "audit_log"},
    )
    graph.add_edge("emit_alert", "audit_log")
    graph.add_edge("audit_log", END)

    return graph.compile()


def run_flow(
    patient_id: int,
    question: str,
    vectorstore,
    generate_fn,
    db_path: str = DEFAULT_DB_PATH,
    model_name: str = "logistic_regression",
    k: int = 3,
) -> AssistantState:
    """Convenience entry point: build and run the compiled flow once."""
    flow = build_flow()
    initial_state: AssistantState = {
        "patient_id": patient_id,
        "question": question,
        "vectorstore": vectorstore,
        "generate_fn": generate_fn,
        "db_path": db_path,
        "model_name": model_name,
        "k": k,
    }
    return flow.invoke(initial_state)
