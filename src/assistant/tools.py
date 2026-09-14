"""
src/assistant/tools.py

Plain Python functions (not LLM-invoked "agent tools") that wrap: (1) the
stroke risk prediction model already trained in Fase 1/2 (src/models.py),
and (2) a deterministic "pending exams" check over a patient record.

These are called directly, in a fixed order, by src/assistant/chain.py and
src/assistant/graph.py — a deterministic pipeline rather than an
LLM-driven tool-calling agent, which keeps the flow reliable and easy to
unit test (no risk of the LLM deciding not to call a tool, or calling it
with malformed arguments).
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from src.models import load_model, predict
from src.preprocessing import clean_data, encode_features, load_data


@lru_cache(maxsize=1)
def _reference_frame() -> pd.DataFrame:
    """
    Cached copy of the full cleaned dataset (minus the target column),
    used only to supply the complete category vocabulary that
    pd.get_dummies needs. One-hot encoding a single row in isolation is
    ambiguous: with only one category present per field, drop_first drops
    it unconditionally, silently zeroing out a dummy column that should
    have been 1. Concatenating the row onto the full reference frame
    before encoding avoids that.
    """
    return clean_data(load_data()).drop(columns=["stroke"])


def predict_stroke_risk(patient_record: dict, model_name: str = "logistic_regression") -> dict:
    """
    Run the existing stroke risk model (src/models.py) on one patient
    record from the patient DB, applying the same one-hot encoding used at
    training time (src/preprocessing.encode_features).

    Returns {"prediction": int, "probability": float}.

    Raises FileNotFoundError if the model has not been trained/saved yet
    (run notebooks/01_baseline.ipynb first — same requirement as
    src.models.load_model).
    """
    row = {k: v for k, v in patient_record.items() if k not in ("id", "stroke")}
    combined = pd.concat([_reference_frame(), pd.DataFrame([row])], ignore_index=True)
    encoded_row = encode_features(combined).iloc[[-1]]

    model = load_model(model_name)
    trained_columns = list(getattr(model, "feature_names_in_", encoded_row.columns))
    for col in trained_columns:
        if col not in encoded_row.columns:
            encoded_row[col] = 0
    encoded_row = encoded_row[trained_columns]

    predictions, probabilities = predict(model, encoded_row)
    return {"prediction": int(predictions[0]), "probability": float(probabilities[0])}


def check_pending_exams(patient_record: dict) -> list[str]:
    """
    Deterministic mock: flags fields missing/unknown in the patient record
    as "pending exams" the assistant should mention, since there is no
    real hospital exam-ordering system to query.
    """
    pending = []
    bmi = patient_record.get("bmi")
    if bmi is None or (isinstance(bmi, float) and pd.isna(bmi)):
        pending.append("Perfil metabólico / IMC")
    if patient_record.get("smoking_status") in (None, "Unknown"):
        pending.append("Histórico de tabagismo (anamnese)")
    return pending
