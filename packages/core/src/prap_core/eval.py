from collections.abc import Hashable, Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class PRF:
    precision: float
    recall: float
    f1: float
    true_positive: int
    false_positive: int
    false_negative: int


@dataclass(frozen=True)
class BinaryPRF:
    precision: float
    recall: float
    f1: float
    accuracy: float
    n: int
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def binary_prf(predicted: Iterable[bool], gold: Iterable[bool]) -> BinaryPRF:
    """Record-level precision / recall / F1 for positional boolean labels.

    Use this for per-record binary classification (one True/False per item,
    aligned by position). For set-membership metrics over hashable items
    use `prf` instead.
    """
    pred = [bool(p) for p in predicted]
    gold_list = [bool(g) for g in gold]
    if len(pred) != len(gold_list):
        raise ValueError(f"length mismatch: predicted={len(pred)}, gold={len(gold_list)}")
    tp = sum(1 for p, g in zip(pred, gold_list, strict=True) if p and g)
    fp = sum(1 for p, g in zip(pred, gold_list, strict=True) if p and not g)
    fn = sum(1 for p, g in zip(pred, gold_list, strict=True) if not p and g)
    tn = sum(1 for p, g in zip(pred, gold_list, strict=True) if not p and not g)
    n = tp + fp + fn + tn
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    accuracy = _safe_div(tp + tn, n)
    return BinaryPRF(
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=accuracy,
        n=n,
        true_positive=tp,
        false_positive=fp,
        true_negative=tn,
        false_negative=fn,
    )


def prf(predicted: Iterable[Hashable], gold: Iterable[Hashable]) -> PRF:
    """Set-level precision / recall / F1 over hashable items."""
    pred_set = set(predicted)
    gold_set = set(gold)
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return PRF(
        precision=precision,
        recall=recall,
        f1=f1,
        true_positive=tp,
        false_positive=fp,
        false_negative=fn,
    )


def confusion_matrix(
    predicted: Iterable[Hashable], gold: Iterable[Hashable]
) -> dict[tuple[Hashable, Hashable], int]:
    """Counts of (gold_label, predicted_label) pairs, aligned by position."""
    pred = list(predicted)
    g = list(gold)
    if len(pred) != len(g):
        raise ValueError(f"length mismatch: predicted={len(pred)}, gold={len(g)}")
    out: dict[tuple[Hashable, Hashable], int] = {}
    for gi, pi in zip(g, pred, strict=True):
        out[(gi, pi)] = out.get((gi, pi), 0) + 1
    return out
