"""Summarise the tuning sweep and report the configuration it favours.

    python scripts/pick_config.py --results /path/to/results

Reads the runs tagged ``experiment: tuning`` and ranks them by meta-test
accuracy. Runs that never left chance level are called out separately: those
indicate an inner learning rate below the threshold at which the base learner
meta-learns at all, which is a qualitative failure rather than a weaker score.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    rows = []
    for path in sorted(Path(args.results).glob("*.json")):
        try:
            r = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if r["config"].get("experiment") != "tuning":
            continue
        acc = r["test"]["summary"]["accuracy"]
        n_way = r["config"]["episodes"]["n_way"]
        history = r.get("history") or {}
        rows.append({
            "inner_lr": r["config"]["algorithm"]["inner_lr"],
            "meta_lr": r["config"]["algorithm"]["meta_lr"],
            "accuracy": acc["mean"],
            "lo": acc["lo"],
            "hi": acc["hi"],
            "chance": 1.0 / n_way,
            "best_val": max(history.get("val_accuracy") or [0.0]),
            "steps_run": (history.get("steps") or [0])[-1],
        })

    if not rows:
        print(f"No tuning runs found in {args.results}.")
        print("Run: python scripts/experiments.py --run --only tuning --data-path ...")
        return 1

    rows.sort(key=lambda r: r["accuracy"], reverse=True)
    print(f"{'inner_lr':>9} {'meta_lr':>9} {'meta-test acc':>26} {'best val':>9} {'steps':>6}")
    print("-" * 66)
    for r in rows:
        # Within one standard-error-ish band of chance means the run never
        # bootstrapped; flag it rather than let it look like a weak score.
        stalled = r["accuracy"] < r["chance"] * 1.15
        note = "  <- never left chance" if stalled else ""
        print(f"{r['inner_lr']:>9} {r['meta_lr']:>9} "
              f"{r['accuracy']:>9.4f} [{r['lo']:.4f}, {r['hi']:.4f}] "
              f"{r['best_val']:>9.4f} {r['steps_run']:>6}{note}")

    best = rows[0]
    print(f"\nBest: inner_lr={best['inner_lr']}, meta_lr={best['meta_lr']} "
          f"-> {best['accuracy']:.4f} meta-test accuracy")
    print("\nIf this is not already what configs/primary.yaml sets, update")
    print("algorithm.inner_lr and algorithm.meta_lr there before the matrix.")

    stalled = [r for r in rows if r["accuracy"] < r["chance"] * 1.15]
    if stalled:
        print(f"\n{len(stalled)} of {len(rows)} configurations never left chance "
              f"({best['chance']:.2f}); the lowest inner_lr values are expected to.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
