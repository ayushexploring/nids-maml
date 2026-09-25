"""Run the whole campaign with one command, resumably.

    python scripts/run_all.py --data-path DATA --output-dir OUT --max-hours 11

Phases run in order of how much the manuscript depends on them, so an
interrupted session still leaves the most important results finished:

    1. duplicates   near-duplicate diagnostic (minutes, no training)
    2. headline     the primary configuration under both protocols
    3. baselines    every comparison method, five seeds
    4. ablations    ten factors, two seeds
    5. analyse      tables and figures from whatever exists

Everything is skippable and resumable: a phase whose outputs are already
present is not repeated, so re-issuing the identical command after a dropped
Colab session continues where it stopped. ``--max-hours`` stops cleanly before
a hosted runtime is reclaimed, rather than being killed mid-run.

Progress is written to ``campaign_status.json`` in the output directory after
every run, so the state of a long campaign can be read without watching a log.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

log = logging.getLogger("campaign")

PHASES = ("duplicates", "headline", "baselines", "ablations", "analyse")


class Budget:
    """Wall-clock guard so a phase is not started that cannot finish."""

    def __init__(self, max_hours: float | None) -> None:
        self.start = time.time()
        self.limit = max_hours * 3600 if max_hours else None

    @property
    def elapsed(self) -> float:
        return time.time() - self.start

    @property
    def remaining(self) -> float:
        return float("inf") if self.limit is None else self.limit - self.elapsed

    def exhausted(self, needed_seconds: float = 0.0) -> bool:
        return self.remaining <= needed_seconds

    def __str__(self) -> str:
        if self.limit is None:
            return f"{self.elapsed / 3600:.1f}h elapsed, no limit"
        return (f"{self.elapsed / 3600:.1f}h elapsed, "
                f"{self.remaining / 3600:.1f}h remaining")


def run(cmd: list[str], budget: Budget) -> int:
    """Run a subprocess, streaming its output, and return its exit code."""
    log.info("$ %s", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=REPO_ROOT)
    if proc.returncode:
        log.error("command failed with exit %d: %s", proc.returncode, " ".join(cmd))
    return proc.returncode


def write_status(out_dir: Path, status: dict) -> None:
    status["updated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (out_dir / "campaign_status.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )


def count_results(out_dir: Path) -> dict[str, int]:
    """Summarise what has already been produced."""
    counts: dict[str, int] = {}
    for path in out_dir.glob("*.json"):
        if path.name in ("campaign_status.json", "duplicates.json", "failed_runs.json"):
            continue
        try:
            cfg = json.loads(path.read_text(encoding="utf-8")).get("config", {})
        except (json.JSONDecodeError, KeyError):
            continue
        counts[cfg.get("experiment", "unknown")] = counts.get(
            cfg.get("experiment", "unknown"), 0) + 1
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-hours", type=float, default=None,
                        help="stop cleanly before a hosted runtime is reclaimed")
    parser.add_argument("--phases", nargs="*", default=list(PHASES), choices=PHASES)
    parser.add_argument("--seeds", type=int, default=5,
                        help="seeds for the baseline table")
    parser.add_argument("--skip-dedup-headline", action="store_true",
                        help="run only the record-level headline configuration")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    budget = Budget(args.max_hours)
    python = sys.executable

    status: dict = {"phases": {}, "data_path": args.data_path,
                    "output_dir": str(out_dir)}
    log.info("campaign starting | %s | phases: %s", budget, ", ".join(args.phases))

    # --- 1. duplicates -------------------------------------------------------
    if "duplicates" in args.phases:
        target = out_dir / "duplicates.json"
        if target.exists():
            log.info("[duplicates] already present, skipping")
            status["phases"]["duplicates"] = "skipped"
        else:
            rc = run([python, "scripts/check_duplicates.py",
                      "--data-path", args.data_path, "--out", str(target)], budget)
            status["phases"]["duplicates"] = "ok" if rc == 0 else "failed"
        write_status(out_dir, status)

    # --- 2. headline ---------------------------------------------------------
    # Both protocols for the primary configuration. These two runs carry the
    # paper's central comparison, so they come before the wider matrix.
    if "headline" in args.phases:
        jobs = [("configs/primary.yaml", "primary_fomaml_seed0")]
        if not args.skip_dedup_headline:
            jobs.append(("configs/dedup.yaml", "dedup_fomaml_seed0"))
        results = {}
        for config, tag in jobs:
            if (out_dir / f"{tag}.json").exists():
                log.info("[headline] %s already present, skipping", tag)
                results[tag] = "skipped"
                continue
            if budget.exhausted(45 * 60):
                log.warning("[headline] not enough time left for %s; stopping", tag)
                results[tag] = "deferred"
                break
            rc = run([python, "-m", "nids_maml.run", "--config", config,
                      "--data-path", args.data_path, "--output-dir", str(out_dir),
                      "--tag", tag, "--seed", "0"], budget)
            results[tag] = "ok" if rc == 0 else "failed"
        status["phases"]["headline"] = results
        write_status(out_dir, status)
        log.info("[headline] %s | %s", results, budget)

    # --- 3 & 4. the matrix ---------------------------------------------------
    for phase in ("baselines", "ablations"):
        if phase not in args.phases:
            continue
        if budget.exhausted(30 * 60):
            log.warning("[%s] not enough time left; stopping cleanly", phase)
            status["phases"][phase] = "deferred"
            write_status(out_dir, status)
            break
        rc = run([python, "scripts/experiments.py", "--run", "--only", phase,
                  "--data-path", args.data_path, "--output-dir", str(out_dir)],
                 budget)
        # A non-zero code here means some individual runs failed; the driver
        # records them in failed_runs.json and continues, so the campaign does
        # not abandon the remaining phases over one bad configuration.
        status["phases"][phase] = "ok" if rc == 0 else "partial"
        status["counts"] = count_results(out_dir)
        write_status(out_dir, status)
        log.info("[%s] finished (%s) | %s", phase, status["phases"][phase], budget)

    # --- 5. analyse ----------------------------------------------------------
    if "analyse" in args.phases:
        rc = run([python, "-m", "nids_maml.analyse",
                  "--results", str(out_dir),
                  "--out", str(out_dir / "paper_assets"), "--figures"], budget)
        status["phases"]["analyse"] = "ok" if rc == 0 else "failed"

    status["counts"] = count_results(out_dir)
    status["elapsed_hours"] = round(budget.elapsed / 3600, 2)
    write_status(out_dir, status)

    log.info("campaign finished | %s", budget)
    log.info("results by experiment: %s", status["counts"])
    deferred = [k for k, v in status["phases"].items()
                if v == "deferred" or (isinstance(v, dict) and "deferred" in v.values())]
    if deferred:
        log.info("deferred: %s -- re-run the identical command to continue",
                 ", ".join(deferred))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
