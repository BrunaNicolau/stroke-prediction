"""
src/assistant/patient_db.py

Loads the stroke dataset as a mock "prontuário" (patient record) table in
SQLite, used by the LangChain assistant to answer questions with
up-to-date patient context (challenge item 2: "realizar consultas em base
de dados estruturadas... contextualizar as respostas com informações
atualizadas do paciente").

There is no real hospital EHR available, so
data/healthcare-dataset-stroke-data.csv (already used for the stroke risk
model in src/models.py) is reused as the structured record source — each
row's `id` becomes the patient_id.
"""

from __future__ import annotations

import os
import sqlite3

import pandas as pd

from src.preprocessing import CSV_NAME

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_CSV_PATH = os.path.join(_PROJECT_ROOT, "data", CSV_NAME)
DEFAULT_DB_PATH = os.path.join(_PROJECT_ROOT, "results", "patient_records.db")
TABLE_NAME = "prontuarios"


def build_patient_db(csv_path: str = DEFAULT_CSV_PATH, db_path: str = DEFAULT_DB_PATH) -> str:
    """
    Load the stroke CSV into a SQLite database with one table
    (`prontuarios`), replacing any existing file at db_path.

    Returns db_path.
    """
    df = pd.read_csv(csv_path)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    if os.path.exists(db_path):
        os.remove(db_path)
    conn = sqlite3.connect(db_path)
    try:
        df.to_sql(TABLE_NAME, conn, index=False)
    finally:
        conn.close()
    return db_path


def get_patient_record(patient_id: int, db_path: str = DEFAULT_DB_PATH) -> dict | None:
    """
    Fetch one patient's record by id.

    Returns None if the id does not exist. Raises FileNotFoundError if the
    DB has not been built yet (run build_patient_db() first).
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(
            f"Patient records DB not found at '{db_path}'.\n"
            "Run src.assistant.patient_db.build_patient_db() first."
        )
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute(f"SELECT * FROM {TABLE_NAME} WHERE id = ?", (patient_id,))
        row = cursor.fetchone()
    finally:
        conn.close()
    return dict(row) if row else None


def run_sql_query(query: str, db_path: str = DEFAULT_DB_PATH) -> list[dict]:
    """
    Run a read-only SQL query against the patient records DB (backing
    function for the LangChain SQL tool in tools.py). Only SELECT
    statements are allowed — anything else raises ValueError.
    """
    normalized = query.strip().lower()
    if not normalized.startswith("select"):
        raise ValueError("Only SELECT queries are allowed against the patient records DB.")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute(query)
        rows = cursor.fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]
