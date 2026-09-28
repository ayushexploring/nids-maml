"""The UNSW-NB15 experiment set, resumable, for a single GPU session.

    python scripts/run_unsw_campaign.py --data-path DATA --output-dir OUT \
        --cache-dir CACHE --max-hours 8

Ordered so that a session cut short still leaves something reportable:

    1. FOMAML seeds 3 and 4      establishes whether the one-in-three failure
                                 seen on seeds 0-2 is a rate or an accident;
                                 the paper's motivating claim depends on it
    2. robust prototype variants  medoid, trimmed and attention against the
                                 mean, seed 0 -- the method contribution
    3. the same variants, seeds 1-2   intervals for whichever variant wins

Runs whose result file already exists are skipped, so re-issuing the identical
command after a dropped session continues rather than restarting.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

log = logging.getLogger("unsw")

# (config stem, result tag, seed)
def build_plan() -> list[tuple[str, str, int]]:
    plan: list[tuple[str, str, int]] = []

    # 1. Is the FOMAML failure a rate? Two more seeds, cheap and decisive.
    for seed in (3, 4):
        plan.append(("unsw_fomaml", f"unsw_fomaml_seed{seed}", seed))

    # 2. Robust estimators at seed 0, against the mean already measured.
    for est in ("medoid", "trimmed", "attention"):
        plan.append((f"unsw_proto_{est}", f"unsw_proto-{est}_seed0", 0))

    # 3. Intervals for the robust estimators.
    for seed in (1, 2):
        for est in ("medoid", "trimmed", "attention"):
            plan.append((f"unsw_proto_{est}", f"unsw_proto-{est}_seed{seed}", seed))

    return plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--max-hours", type=float, default=None)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plan = build_plan()

    if args.list:
        for cfg, tag, seed in plan:
            state = "done" if (out / f"{tag}.json").exists() else "pending"
            print(f"  {tag:<34} {cfg:<22} seed {seed}  [{state}]")
        return 0

    started = time.time()
    limit = args.max_hours * 3600 if args.max_hours else None
    completed, skipped, failed = 0, 0, []

    for index, (cfg, tag, seed) in enumerate(plan, 1):
        if (out / f"{tag}.json").exists():
            skipped += 1
            log.info("[%d/%d] %s -- present, skipping", index, len(plan), tag)
            continue
        if limit is not None and time.time() - started > limit - 1800:
            log.warning("stopping cleanly: under 30 min of the budget remains")
            break

        log.info("[%d/%d] %s", index, len(plan), tag)
        cmd = [sys.executable, "-u", "-m", "nids_maml.run",
               "--config", f"configs/{cfg}.yaml",
               "--data-path", args.data_path,
               "--output-dir", str(out),
               "--tag", tag, "--seed", str(seed)]
        if args.cache_dir:
            cmd += ["--cache-dir", args.cache_dir]

        run_log = out / "run_logs" / f"{tag}.log"
        run_log.parent.mkdir(parents=True, exist_ok=True)
        with open(run_log, "w", encoding="utf-8") as handle:
            proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=handle,
                                  stderr=subprocess.STDOUT)

        if proc.returncode == 0 and (out / f"{tag}.json").exists():
            accuracy = json.loads((out / f"{tag}.json").read_text())
            value = accuracy["test"]["summary"]["accuracy"]
            completed += 1
            log.info("      %.4f [%.4f, %.4f]", value["mean"], value["lo"], value["hi"])
        else:
            tail = "".join(run_log.read_text(encoding="utf-8", errors="replace")
                           .splitlines(True)[-12:])
            log.error("      failed (exit %s)\n%s", proc.returncode, tail)
            failed.append(tag)

        elapsed = (time.time() - started) / 60
        log.info("      %.0f min elapsed, %d done / %d skipped / %d failed",
                 elapsed, completed, skipped, len(failed))

    log.info("finished: %d completed, %d skipped, %d failed",
             completed, skipped, len(failed))
    if failed:
        log.error("failed: %s", ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
