"""
src/assistant/tools.py

Plain Python functions (not LLM-invoked "agent tools") that wrap:
(1) the stroke risk prediction model already trained in Fase 1/2
(src/models.py), and
(2) a deterministic "pending exams" check over a patient record.

These are called directly, in a fixed order, by src/assistant/chain.py
and src/assistant/graph.py — a deterministic pipeline rather than an
LLM-driven tool-calling agent.
"""

from __future__ import annotations

import os
from functools import lru_cache

import pandas as pd
from sklearn.model_selection import train_test_split

from src.models import load_model, predict
from src.preprocessing import clean_data, encode_features, load_data


@lru_cache(maxsize=1)
def _reference_frame() -> pd.DataFrame:
    """
    Cached copy of the full cleaned dataset (minus the target column).

    This provides the complete category vocabulary required by
    pd.get_dummies(), avoiding the problem of one-hot encoding a
    single patient row in isolation.
    """
    return clean_data(load_data()).drop(columns=["stroke"])


@lru_cache(maxsize=1)
def _training_bmi_mean() -> float:
    """
    Reproduce exactly the BMI mean used during model training.

    The original preprocessing calculates the BMI mean after the
    stratified train/test split and uses that mean to fill missing
    BMI values in both sets.

    We reproduce that calculation here so inference uses the same
    value instead of an arbitrary fill value such as 0.
    """
    df = load_data()
    df = clean_data(df)
    df = encode_features(df)

    X = df.drop(columns=["stroke"])
    y = df["stroke"]

    X_train, _, _, _ = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    return float(X_train["bmi"].mean())


def predict_stroke_risk(
    patient_record: dict,
    model_name: str = "logistic_regression",
) -> dict:
    """
    Run the existing stroke risk model on one patient record.

    The inference preprocessing reproduces the same transformations
    used during training:

    1. Remove id/stroke.
    2. Use the full dataset as the categorical vocabulary.
    3. Apply the same one-hot encoding.
    4. Fill missing BMI with the training-set BMI mean.
    5. Align columns with the features expected by the trained model.
    6. Run prediction and probability estimation.

    Returns:
        {
            "prediction": int,
            "probability": float
        }

    Raises:
        FileNotFoundError:
            If the trained model has not been saved yet.
    """
    # ---------------------------------------------------------------
    # 1. Load trained model
    # ---------------------------------------------------------------
    model = load_model(model_name)

    # ---------------------------------------------------------------
    # 2. Prepare patient row
    # ---------------------------------------------------------------
    row = {
        k: v
        for k, v in patient_record.items()
        if k not in ("id", "stroke")
    }

    reference = _reference_frame()

    # Keep only columns that exist in the training dataset.
    row = {
        k: v
        for k, v in row.items()
        if k in reference.columns
    }

    patient_df = pd.DataFrame([row])

    # Make sure the patient row has the same raw columns as the
    # reference dataset. Missing fields become NaN and are handled
    # explicitly below.
    patient_df = patient_df.reindex(columns=reference.columns)

    # ---------------------------------------------------------------
    # 3. Fill BMI using the exact training-set mean
    # ---------------------------------------------------------------
    bmi_mean = _training_bmi_mean()

    if "bmi" in patient_df.columns:
        patient_df["bmi"] = pd.to_numeric(
            patient_df["bmi"],
            errors="coerce",
        )

        patient_df["bmi"] = patient_df["bmi"].fillna(bmi_mean)

    # ---------------------------------------------------------------
    # 4. One-hot encode using the full reference dataset
    # ---------------------------------------------------------------
    combined = pd.concat(
        [reference, patient_df],
        ignore_index=True,
    )

    encoded = encode_features(combined)
    encoded_row = encoded.iloc[[-1]].copy()

    # ---------------------------------------------------------------
    # 5. Align with the exact columns expected by the model
    # ---------------------------------------------------------------
    trained_columns = list(
        getattr(model, "feature_names_in_", encoded_row.columns)
    )

    # Add any missing columns expected by the model.
    for col in trained_columns:
        if col not in encoded_row.columns:
            encoded_row[col] = 0

    # Remove any columns not used during training and preserve
    # the exact training column order.
    encoded_row = encoded_row[trained_columns]

    # ---------------------------------------------------------------
    # 6. Final safety check
    # ---------------------------------------------------------------
    if encoded_row.isna().any().any():
        nan_columns = encoded_row.columns[
            encoded_row.isna().any()
        ].tolist()

        raise ValueError(
            "NaN encontrado nas features antes da previsão. "
            f"Colunas afetadas: {nan_columns}"
        )

    # ---------------------------------------------------------------
    # 7. Prediction
    # ---------------------------------------------------------------
    predictions, probabilities = predict(
        model,
        encoded_row,
    )

    return {
        "prediction": int(predictions[0]),
        "probability": float(probabilities[0]),
    }


def check_pending_exams(patient_record: dict) -> list[str]:
    """
    Deterministic mock: flags fields missing/unknown in the patient
    record as "pending exams" the assistant should mention.

    There is no real hospital exam-ordering system to query.
    """
    pending = []

    bmi = patient_record.get("bmi")

    if bmi is None or (
        isinstance(bmi, float) and pd.isna(bmi)
    ):
        pending.append("Perfil metabólico / IMC")

    if patient_record.get("smoking_status") in (
        None,
        "Unknown",
    ):
        pending.append("Histórico de tabagismo (anamnese)")

    return pending