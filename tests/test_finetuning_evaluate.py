from src.finetuning.evaluate import (
    avoids_direct_prescription,
    evaluate_eval_set,
    evaluate_generation,
    mentions_key_terms,
    reasonable_length,
)


def test_mentions_key_terms_true_when_overlap_present():
    assert mentions_key_terms("Considere trombolise conforme protocolo", "trombolise e protocolo interno")


def test_mentions_key_terms_false_when_no_overlap():
    assert not mentions_key_terms("Resposta generica sem nada em comum", "termos completamente diferentes")


def test_avoids_direct_prescription_true_for_safe_text():
    assert avoids_direct_prescription("Considere avaliar o paciente.")


def test_avoids_direct_prescription_false_for_prescription_text():
    assert not avoids_direct_prescription("Tome 500mg agora.")


def test_reasonable_length_bounds():
    assert not reasonable_length("curto")
    assert reasonable_length("x" * 100)
    assert not reasonable_length("x" * 5000)


def test_evaluate_generation_returns_score_and_checks():
    result = evaluate_generation("Considere avaliar trombolise conforme protocolo interno.", "trombolise protocolo")
    assert 0.0 <= result["score"] <= 1.0
    assert set(result["checks"]) == {"mentions_key_terms", "avoids_direct_prescription", "reasonable_length"}


def test_evaluate_eval_set_aggregates_mean_score():
    pairs = [
        ("Considere avaliar trombolise conforme protocolo interno.", "trombolise protocolo"),
        ("Tome 500mg agora.", "trombolise protocolo"),
    ]
    result = evaluate_eval_set(pairs)
    assert len(result["results"]) == 2
    assert 0.0 <= result["mean_score"] <= 1.0


def test_evaluate_eval_set_handles_empty_list():
    result = evaluate_eval_set([])
    assert result == {"mean_score": 0.0, "results": []}
