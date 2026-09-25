"""Episodic metrics, interval estimates and paired significance tests.

Every headline number this project reports is a mean over episodes accompanied
by a 95% confidence interval. Single-run point estimates on thirty episodes --
what the previous results consisted of -- cannot distinguish a real effect from
sampling noise, and a reviewer is entitled to assume the worst when no interval
is given.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy import stats


@dataclass
class Interval:
    mean: float
    lo: float
    hi: float
    n: int

    def __str__(self) -> str:
        return f"{self.mean:.4f} [{self.lo:.4f}, {self.hi:.4f}]"

    def as_dict(self) -> dict:
        return asdict(self)


def mean_ci(values: np.ndarray | list[float], confidence: float = 0.95) -> Interval:
    """Mean with a Student-t confidence interval over episodes.

    The episode is the unit of analysis, not the query instance: query
    predictions within an episode share a support set and are therefore
    correlated, so an interval computed over pooled instances would be too
    narrow.
    """
    v = np.asarray(values, dtype=float)
    n = len(v)
    mean = float(v.mean())
    if n < 2:
        return Interval(mean, mean, mean, n)
    sem = float(v.std(ddof=1) / np.sqrt(n))
    half = sem * stats.t.ppf(0.5 + confidence / 2, df=n - 1)
    return Interval(mean, mean - half, mean + half, n)


def wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> Interval:
    """Wilson score interval for a binomial proportion.

    Used wherever a per-family score is computed on a small query set. A score
    of exactly 1.0 on 30 instances has a lower Wilson bound near 0.88, which is
    the honest way to report it.
    """
    if trials == 0:
        return Interval(float("nan"), float("nan"), float("nan"), 0)
    z = stats.norm.ppf(0.5 + confidence / 2)
    p = successes / trials
    denom = 1 + z**2 / trials
    centre = (p + z**2 / (2 * trials)) / denom
    half = z * np.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denom
    return Interval(float(p), float(centre - half), float(centre + half), trials)


def episode_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_way: int) -> dict[str, float]:
    """Accuracy and macro-averaged precision, recall and F1 for one episode."""
    acc = float((y_true == y_pred).mean())
    precisions, recalls, f1s = [], [], []
    for c in range(n_way):
        tp = float(((y_pred == c) & (y_true == c)).sum())
        fp = float(((y_pred == c) & (y_true != c)).sum())
        fn = float(((y_pred != c) & (y_true == c)).sum())
        prec = tp / (tp + fp) if tp + fp > 0 else 0.0
        rec = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
    return {
        "accuracy": acc,
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
    }


def binary_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray | None = None
) -> dict[str, float]:
    """Binary metrics with the attack family as the positive class (label 1)."""
    tp = float(((y_pred == 1) & (y_true == 1)).sum())
    fp = float(((y_pred == 1) & (y_true == 0)).sum())
    fn = float(((y_pred == 0) & (y_true == 1)).sum())
    tn = float(((y_pred == 0) & (y_true == 0)).sum())
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec = tp / (tp + fn) if tp + fn > 0 else 0.0
    out = {
        "accuracy": (tp + tn) / max(len(y_true), 1),
        "precision": prec,
        "recall": rec,
        "f1": 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0,
        "fpr": fp / (fp + tn) if fp + tn > 0 else 0.0,
    }
    if probs is not None and len(np.unique(y_true)) == 2:
        out["auroc"] = float(_auroc(y_true, probs[:, 1]))
    return out


def _auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based AUROC, ties handled by average ranks."""
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average ranks within tied groups.
    sorted_scores = scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def expected_calibration_error(
    y_true: np.ndarray, probs: np.ndarray, n_bins: int = 15
) -> float:
    """ECE over equal-width confidence bins.

    An adaptive detector whose confidence is miscalibrated cannot be given a
    usable alerting threshold, so this is reported alongside accuracy.
    """
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if mask.sum() == 0:
            continue
        ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def paired_test(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Compare two methods episode-by-episode on the same episode stream.

    Reports both the paired t-test and Wilcoxon signed-rank, plus Cohen's d_z.
    Pairing is only valid when both methods were evaluated on the identical
    episode list -- which ``EpisodeSampler.fixed_set`` guarantees.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired test needs equal lengths, got {a.shape} and {b.shape}")
    diff = a - b
    result = {
        "mean_difference": float(diff.mean()),
        "n": int(len(diff)),
    }
    if np.allclose(diff, 0):
        result.update({"t_statistic": 0.0, "t_p_value": 1.0,
                       "wilcoxon_p_value": 1.0, "cohens_dz": 0.0})
        return result
    t_stat, t_p = stats.ttest_rel(a, b)
    try:
        _, w_p = stats.wilcoxon(a, b)
    except ValueError:  # all differences zero after ties are dropped
        w_p = 1.0
    result.update({
        "t_statistic": float(t_stat),
        "t_p_value": float(t_p),
        "wilcoxon_p_value": float(w_p),
        "cohens_dz": float(diff.mean() / diff.std(ddof=1)),
    })
    return result


def holm_bonferroni(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Holm-Bonferroni correction across a family of comparisons.

    The baseline table compares the proposed method against six alternatives on
    the same episodes; reporting six uncorrected p-values would inflate the
    family-wise error rate.
    """
    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(ordered)
    out: dict[str, dict] = {}
    still_rejecting = True
    for rank, (name, p) in enumerate(ordered):
        threshold = alpha / (m - rank)
        if p > threshold:
            still_rejecting = False
        out[name] = {
            "p_value": p,
            "threshold": threshold,
            "significant": still_rejecting,
        }
    return out
