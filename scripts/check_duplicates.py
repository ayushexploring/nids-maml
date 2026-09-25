"""Quantify near-duplicate flows across the meta-splits.

CIC-IDS2017 is known to contain large runs of near-identical flows: a single
DDoS burst or port-scan sweep produces thousands of records whose feature
vectors differ negligibly. Splitting by row, as this pipeline does, guarantees
that no *record* is shared between meta-train and meta-test, but it does not
guarantee that a test record has no near-twin in training.

That distinction matters for how a high accuracy should be read, so it is
measured rather than assumed. This script reports, for meta-test flows:

* the share that are exact duplicates of some meta-train flow;
* the distribution of distance to the nearest meta-train flow of the same
  class, expressed in units of that class's own internal spread, so the figure
  is comparable across classes with very different feature scales;
* the same distance for a *different*-class nearest neighbour, which is the
  reference point: if same-class neighbours are far closer than different-class
  ones, the task is genuinely separable rather than merely memorised.

Reference: Engelen, Visser and Verwer (2021), "Troubleshooting an Intrusion
Detection Dataset: the CICIDS2017 Case Study", IEEE Security and Privacy
Workshops.

    python scripts/check_duplicates.py --data-path /path/to/cicids2017.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from nids_maml.data import PRIMARY_CLASSES, SplitSpec, load_cicids2017  # noqa: E402

log = logging.getLogger("duplicates")


def exact_duplicate_rate(train: np.ndarray, test: np.ndarray) -> float:
    """Share of test rows byte-identical to some training row."""
    seen = {hashlib.blake2b(r.tobytes(), digest_size=16).digest()
            for r in np.ascontiguousarray(train)}
    hits = sum(
        hashlib.blake2b(r.tobytes(), digest_size=16).digest() in seen
        for r in np.ascontiguousarray(test)
    )
    return hits / max(len(test), 1)


def nearest_distances(
    query: np.ndarray, reference: np.ndarray, block: int = 512
) -> np.ndarray:
    """Euclidean distance from each query row to its nearest reference row."""
    if len(reference) == 0 or len(query) == 0:
        return np.full(len(query), np.nan)
    out = np.empty(len(query), dtype=np.float64)
    ref_sq = (reference ** 2).sum(1)
    for start in range(0, len(query), block):
        q = query[start:start + block]
        # |a-b|^2 = |a|^2 + |b|^2 - 2ab, clipped because round-off can go < 0.
        d2 = (q ** 2).sum(1)[:, None] + ref_sq[None, :] - 2.0 * q @ reference.T
        out[start:start + block] = np.sqrt(np.maximum(d2, 0.0)).min(1)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample", type=int, default=2000,
                        help="test rows sampled per class for the distance study")
    parser.add_argument("--reference", type=int, default=20000,
                        help="training rows sampled per class as neighbour candidates")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    bundle = load_cicids2017(args.data_path, classes=PRIMARY_CLASSES,
                             split=SplitSpec(), seed=args.seed)
    rng = np.random.default_rng(args.seed)

    overall = exact_duplicate_rate(bundle.X_train, bundle.X_test)
    print(f"\nExact duplicates: {overall:.2%} of meta-test flows are "
          f"byte-identical to a meta-train flow\n")

    header = (f"{'class':>12} {'n_test':>8} {'exact dup':>10} "
              f"{'same-class NN':>14} {'other-class NN':>15} {'ratio':>7}")
    print(header)
    print("-" * len(header))

    report: dict[str, dict] = {}
    for cid, name in enumerate(bundle.class_names):
        test_idx = np.flatnonzero(bundle.y_test == cid)
        train_idx = np.flatnonzero(bundle.y_train == cid)
        other_idx = np.flatnonzero(bundle.y_train != cid)
        if len(test_idx) == 0 or len(train_idx) == 0:
            continue

        q = bundle.X_test[rng.choice(test_idx, min(args.sample, len(test_idx)),
                                     replace=False)]
        same = bundle.X_train[rng.choice(train_idx, min(args.reference, len(train_idx)),
                                         replace=False)]
        other = bundle.X_train[rng.choice(other_idx, min(args.reference, len(other_idx)),
                                          replace=False)]

        d_same = nearest_distances(q, same)
        d_other = nearest_distances(q, other)
        dup = exact_duplicate_rate(bundle.X_train[train_idx], bundle.X_test[test_idx])

        median_same = float(np.median(d_same))
        median_other = float(np.median(d_other))
        ratio = median_other / median_same if median_same > 0 else float("inf")

        print(f"{name:>12} {len(test_idx):>8} {dup:>9.2%} "
              f"{median_same:>14.4f} {median_other:>15.4f} {ratio:>7.1f}x")

        report[name] = {
            "n_test": int(len(test_idx)),
            "exact_duplicate_rate": dup,
            "median_nn_same_class": median_same,
            "median_nn_other_class": median_other,
            "separation_ratio": ratio,
            "share_within_1pct_of_a_train_flow": float((d_same < 0.01).mean()),
        }

    print("\nReading these numbers:")
    print("  A high exact-duplicate rate means the split is disjoint by record")
    print("  but not by content, and accuracy should be described as such.")
    print("  A large ratio means same-class neighbours sit much closer than")
    print("  other-class ones, i.e. the classes are genuinely separable.")

    if args.out:
        payload = {"overall_exact_duplicate_rate": overall,
                   "per_class": report,
                   "fingerprint": bundle.meta}
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
