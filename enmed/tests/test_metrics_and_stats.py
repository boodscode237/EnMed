"""Unit tests for the metric and statistical functions.

Run with:  pytest -q
These tests use small synthetic inputs and require no GPU, models, or network.
"""
import numpy as np

from enmed.evaluation.metrics import (
    exact_match,
    hamming_accuracy,
    mcqa_item_correct,
    normalize_fr,
    token_f1,
)
from enmed.stats.corrections import benjamini_hochberg, bonferroni
from enmed.stats.critical_difference import critical_difference
from enmed.stats.paired_tests import paired_test


# --------------------------- metrics --------------------------------------- #
def test_normalize_fr_strips_articles_and_accents():
    assert normalize_fr("Le Patient") == "patient"
    assert normalize_fr("l'œdème") == normalize_fr("oedeme") or "deme" in normalize_fr("l'œdème")


def test_mcqa_exact_and_multi():
    assert mcqa_item_correct("B", "B") == 1.0
    assert mcqa_item_correct("A", "B") == 0.0
    assert mcqa_item_correct("A, C", "C,A") == 1.0  # order-independent


def test_hamming_partial_credit():
    assert hamming_accuracy("A,B", "A,B,C") == 2 / 3
    assert hamming_accuracy("A", "A") == 1.0


def test_token_f1_french_morphology():
    # "le patient" vs "patient" should score perfectly after normalisation
    assert token_f1("le patient", "patient") == 1.0
    assert token_f1("", "patient") == 0.0
    assert 0.0 < token_f1("douleur thoracique aigue", "douleur thoracique") < 1.0


def test_exact_match():
    assert exact_match("Le patient", "patient") == 1.0
    assert exact_match("fièvre", "toux") == 0.0


# --------------------------- paired tests ---------------------------------- #
def test_paired_test_identical_is_null():
    x = np.array([1.0, 0.0, 1.0, 1.0])
    res = paired_test(x, x)
    assert res["delta"] == 0.0
    assert res["p_value"] == 1.0
    assert res["cohen_d"] == 0.0


def test_paired_test_positive_delta():
    ref = np.zeros(30)
    cand = np.ones(30)
    res = paired_test(cand, cand * 0 + 1)  # identical → null
    assert res["delta"] == 0.0
    res2 = paired_test(cand, ref)
    assert res2["delta"] == 1.0
    assert res2["ci_low"] <= res2["delta"] <= res2["ci_high"]


# --------------------------- corrections ----------------------------------- #
def test_benjamini_hochberg_monotone():
    p = [0.001, 0.01, 0.2, 0.8]
    mask = benjamini_hochberg(p, q=0.05)
    # smallest p must be rejected; largest must not
    assert mask[0] and not mask[-1]


def test_bonferroni_threshold():
    p = [0.01, 0.04, 0.5]
    mask = bonferroni(p, alpha=0.05)  # threshold 0.05/3 ≈ 0.0167
    assert mask[0] and not mask[1] and not mask[2]


# --------------------------- critical difference --------------------------- #
def test_critical_difference_positive():
    cd = critical_difference(n_models=8, n_datasets=3)
    assert cd > 0
    # more models → larger CD
    assert critical_difference(9, 3) > critical_difference(4, 3)
