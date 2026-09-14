import pandas as pd
import pytest

from src.assistant.patient_db import build_patient_db, get_patient_record, run_sql_query


@pytest.fixture
def small_csv(tmp_path):
    df = pd.DataFrame(
        [
            {"id": 1, "gender": "Male", "age": 67.0, "hypertension": 0, "heart_disease": 1,
             "ever_married": "Yes", "work_type": "Private", "Residence_type": "Urban",
             "avg_glucose_level": 228.69, "bmi": 36.6, "smoking_status": "formerly smoked", "stroke": 1},
            {"id": 2, "gender": "Female", "age": 30.0, "hypertension": 0, "heart_disease": 0,
             "ever_married": "No", "work_type": "Private", "Residence_type": "Rural",
             "avg_glucose_level": 90.0, "bmi": None, "smoking_status": "Unknown", "stroke": 0},
        ]
    )
    path = tmp_path / "mini_stroke.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_build_patient_db_creates_file(tmp_path, small_csv):
    db_path = str(tmp_path / "patients.db")
    result_path = build_patient_db(csv_path=small_csv, db_path=db_path)
    assert result_path == db_path
    import os
    assert os.path.exists(db_path)


def test_get_patient_record_returns_matching_row(tmp_path, small_csv):
    db_path = str(tmp_path / "patients.db")
    build_patient_db(csv_path=small_csv, db_path=db_path)

    record = get_patient_record(1, db_path=db_path)
    assert record["id"] == 1
    assert record["gender"] == "Male"
    assert record["stroke"] == 1


def test_get_patient_record_returns_none_for_missing_id(tmp_path, small_csv):
    db_path = str(tmp_path / "patients.db")
    build_patient_db(csv_path=small_csv, db_path=db_path)

    assert get_patient_record(999, db_path=db_path) is None


def test_get_patient_record_raises_if_db_missing(tmp_path):
    db_path = str(tmp_path / "does_not_exist.db")
    with pytest.raises(FileNotFoundError):
        get_patient_record(1, db_path=db_path)


def test_run_sql_query_executes_select(tmp_path, small_csv):
    db_path = str(tmp_path / "patients.db")
    build_patient_db(csv_path=small_csv, db_path=db_path)

    rows = run_sql_query("SELECT id, age FROM prontuarios WHERE stroke = 1", db_path=db_path)
    assert rows == [{"id": 1, "age": 67.0}]


def test_run_sql_query_rejects_non_select(tmp_path, small_csv):
    db_path = str(tmp_path / "patients.db")
    build_patient_db(csv_path=small_csv, db_path=db_path)

    with pytest.raises(ValueError):
        run_sql_query("DROP TABLE prontuarios", db_path=db_path)
