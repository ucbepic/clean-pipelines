import pytest
from prap_core.eval import binary_prf, confusion_matrix, prf


def test_prf_perfect():
    r = prf(["a", "b", "c"], ["a", "b", "c"])
    assert r.precision == 1.0
    assert r.recall == 1.0
    assert r.f1 == 1.0


def test_prf_mixed():
    r = prf(["a", "b", "x"], ["a", "b", "c"])
    assert r.true_positive == 2
    assert r.false_positive == 1
    assert r.false_negative == 1
    assert r.precision == pytest.approx(2 / 3)
    assert r.recall == pytest.approx(2 / 3)
    assert r.f1 == pytest.approx(2 / 3)


def test_prf_empty():
    r = prf([], [])
    assert r.precision == 0.0
    assert r.recall == 0.0
    assert r.f1 == 0.0


def test_confusion_matrix():
    cm = confusion_matrix(["a", "a", "b"], ["a", "b", "b"])
    assert cm[("a", "a")] == 1
    assert cm[("b", "a")] == 1
    assert cm[("b", "b")] == 1


def test_confusion_matrix_length_mismatch():
    with pytest.raises(ValueError):
        confusion_matrix(["a"], ["a", "b"])


def test_binary_prf_mixed():
    # gold: T T F F   pred: T F F T  → TP=1, FN=1, TN=1, FP=1
    r = binary_prf([True, False, False, True], [True, True, False, False])
    assert (r.true_positive, r.false_positive, r.true_negative, r.false_negative) == (1, 1, 1, 1)
    assert r.precision == pytest.approx(0.5)
    assert r.recall == pytest.approx(0.5)
    assert r.f1 == pytest.approx(0.5)
    assert r.accuracy == pytest.approx(0.5)
    assert r.n == 4


def test_binary_prf_perfect():
    r = binary_prf([True, False, True], [True, False, True])
    assert r.precision == 1.0 and r.recall == 1.0 and r.f1 == 1.0 and r.accuracy == 1.0


def test_binary_prf_all_negative():
    # avoids divide-by-zero when there are no positives anywhere
    r = binary_prf([False, False], [False, False])
    assert r.precision == 0.0 and r.recall == 0.0 and r.f1 == 0.0
    assert r.accuracy == 1.0


def test_binary_prf_length_mismatch():
    with pytest.raises(ValueError):
        binary_prf([True], [True, False])
