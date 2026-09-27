"""Define and drive the full experiment matrix.

The matrix is declared here in one place so that the set of runs behind the
manuscript is itself a reviewable artefact rather than a shell history.

    python scripts/experiments.py --list                  # show the matrix
    python scripts/experiments.py --run --data-path ...   # execute it
    python scripts/experiments.py --run --only baselines  # one group

Runs are skipped when their result file already exists, so an interrupted
Colab session resumes by re-issuing the same command.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402

from nids_maml.run import DEFAULTS, deep_merge  # noqa: E402

log = logging.getLogger("experiments")

# Three seeds, not five. Three is enough for a confidence interval across
# initialisations, and the matrix is resumable, so seeds 3 and 4 can be added
# later by widening this list and re-issuing the same command -- completed runs
# are skipped.
SEEDS = [0, 1, 2]

# --- Group 1: baselines ----------------------------------------------------
# Every method is run on identical splits and identical episode streams, which
# is what makes the paired tests in analyse.py valid. The Transformer encoder
# is held fixed across the meta-learners so that the comparison isolates the
# meta-learning algorithm and nothing else.
# Ordered by how much the baseline table depends on each row, because a
# campaign can be cut short by a session or quota limit at any point. The
# classical baselines sit early despite being less interesting: they need no
# meta-training, so they cost a minute or two each and fill table rows almost
# for free.
# Protocol B (benign-vs-family binary episodes) and the adaptation curve cost
# roughly 40% of a run's wall clock, and the manuscript needs them only for the
# proposed method. Every other row of the baseline table is a 5-way accuracy,
# so those runs skip both. LIGHT below applies that.
LIGHT = {"evaluation": {"protocol_b": False, "adaptation_curve_steps": 0}}

BASELINES = [
    {"experiment": "primary", "algorithm": {"name": "fomaml"}},
    # The control that matters most: same encoder, same label budget, ordinary
    # supervised pre-training instead of episodic meta-training.
    {"experiment": "primary", "algorithm": {"name": "supervised"}},
    {"experiment": "primary", "algorithm": {"name": "classical:random_forest"}},
    {"experiment": "primary", "algorithm": {"name": "classical:gradient_boosting"}},
    {"experiment": "primary", "algorithm": {"name": "classical:logistic"}},
    {"experiment": "primary", "algorithm": {"name": "protonet"}},
    # Isolates the contribution of attention while holding the meta-learner
    # fixed; belongs in this table rather than the ablations.
    {"experiment": "primary", "algorithm": {"name": "fomaml"},
     "model": {"name": "mlp"}, "tag_suffix": "mlp"},
    {"experiment": "primary", "algorithm": {"name": "maml"}},
    {"experiment": "primary", "algorithm": {"name": "reptile"}},
]

# --- Group 2: ablations ----------------------------------------------------
# Each factor is varied with all others held at the primary configuration.
ABLATIONS: dict[str, list[dict]] = {
    "k_shot": [{"episodes": {"k_shot": k}} for k in (1, 5, 10, 20)],
    "n_way": [{"episodes": {"n_way": n}} for n in (2, 3, 5)],
    "inner_steps": [{"algorithm": {"inner_steps": s}} for s in (1, 3, 5, 10)],
    # The decisive factor: at 0.01 a Transformer base learner does not
    # meta-learn at all. Included at the value used by the previous work.
    "inner_lr": [{"algorithm": {"inner_lr": lr}} for lr in (0.01, 0.05, 0.1, 0.3)],
    "n_blocks": [{"model": {"n_blocks": b}} for b in (1, 2, 3, 4)],
    "d_model": [{"model": {"d_model": d}} for d in (64, 128, 256)],
    "n_heads": [{"model": {"n_heads": h}} for h in (2, 4, 8)],
    "slot_encoding": [{"model": {"slot_encoding": s}}
                      for s in ("sinusoidal", "learned", "none")],
    "learn_lr": [{"algorithm": {"learn_lr": v}} for v in (False, True)],
    "order": [{"algorithm": {"name": a}} for a in ("fomaml", "maml")],
}

# Ablation factors are expensive; by default they use fewer seeds than the
# baseline table, which needs the tighter intervals.
ABLATION_SEEDS = [0, 1]

# Ablations answer a relative question -- does this factor move the result --
# so every cell only has to share one budget, not the largest one. Measured
# convergence on CIC-IDS2017 plateaus by roughly step 1200, so 1500 steps costs
# about 0.3 accuracy points against the full budget while halving the runtime
# of the largest group in the matrix. The baseline table keeps the full budget,
# because its numbers are the ones that get reported as headline results.
ABLATION_META_STEPS = 1500

# --- Group 0: tuning -------------------------------------------------------
# A short sweep over the two learning rates, run before the matrix so the
# operating point is chosen on the actual data rather than assumed. The inner
# learning rate is the decisive one: below a threshold the Transformer base
# learner does not meta-learn at all, and the threshold is dataset dependent.
#
# These runs use a reduced step budget and a single seed, and carry the
# experiment name "tuning" so that analyse.py keeps them out of the baseline
# table. They are for choosing a configuration, not for reporting.
TUNING_META_STEPS = 800
TUNING_SEED = 0
TUNING = [
    {"experiment": "tuning",
     "algorithm": {"inner_lr": inner_lr, "meta_lr": meta_lr},
     "train": {"meta_steps": TUNING_META_STEPS, "eval_every": 100,
               "eval_episodes": 100, "patience": 8},
     "evaluation": {"test_episodes": 200, "adaptation_curve_steps": 0,
                    "protocol_b": False}}
    for inner_lr in (0.01, 0.03, 0.1, 0.3)
    for meta_lr in (0.001, 0.0003)
]


def build_matrix(groups: set[str]) -> list[dict]:
    """Expand the declared matrix into concrete run specifications."""
    runs: list[dict] = []

    if "tuning" in groups:
        for spec in TUNING:
            spec = copy.deepcopy(spec)
            inner_lr = spec["algorithm"]["inner_lr"]
            meta_lr = spec["algorithm"]["meta_lr"]
            tag = _safe(f"tuning_inner{inner_lr}_meta{meta_lr}_seed{TUNING_SEED}")
            runs.append({"overrides": spec, "seed": TUNING_SEED, "tag": tag})

    if "baselines" in groups:
        # Seed-major, not method-major. A campaign stopped early then leaves a
        # complete table at fewer seeds, which is reportable, rather than a few
        # methods at five seeds and the rest missing, which is not.
        for seed, spec in itertools.product(SEEDS, BASELINES):
            spec = copy.deepcopy(spec)
            suffix = spec.pop("tag_suffix", None)
            algo = spec["algorithm"]["name"]
            is_proposed = algo == "fomaml" and "model" not in spec
            if not is_proposed:
                spec = deep_merge(spec, copy.deepcopy(LIGHT))
            tag = f"primary_{algo.replace(':', '-')}"
            if suffix:
                tag += f"_{suffix}"
            runs.append({"overrides": spec, "seed": seed, "tag": f"{tag}_seed{seed}"})

    if "ablations" in groups:
        for factor, levels in ABLATIONS.items():
            for seed, level in itertools.product(ABLATION_SEEDS, levels):
                spec = copy.deepcopy(level)
                spec["experiment"] = f"ablation_{factor}"
                spec.setdefault("train", {})["meta_steps"] = ABLATION_META_STEPS
                spec = deep_merge(spec, copy.deepcopy(LIGHT))
                value = _level_value(level)
                tag = f"ablation_{factor}_{value}_seed{seed}"
                runs.append({"overrides": spec, "seed": seed, "tag": _safe(tag)})

    return runs


def _level_value(level: dict) -> str:
    for section in level.values():
        if isinstance(section, dict):
            return str(next(iter(section.values())))
    return "x"


def _safe(text: str) -> str:
    return text.replace("/", "-").replace(" ", "").replace(":", "-").replace(".", "p")


def estimate(runs: list[dict], seconds_per_run: float) -> str:
    total = len(runs) * seconds_per_run
    return f"{len(runs)} runs, ~{total / 3600:.1f} GPU-hours at {seconds_per_run / 60:.0f} min/run"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="print the matrix and exit")
    parser.add_argument("--run", action="store_true", help="execute the matrix")
    parser.add_argument("--only", nargs="*", default=["baselines", "ablations"],
                        choices=["tuning", "baselines", "ablations"])
    parser.add_argument("--data-path", type=str, required=False)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--base-config", type=Path,
                        default=REPO_ROOT / "configs" / "primary.yaml")
    parser.add_argument("--meta-steps", type=int, help="override for a quick pass")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="directory for cached preprocessed splits")
    parser.add_argument("--workers", type=int, default=2,
                        help="configurations to run concurrently on one GPU")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")

    runs = build_matrix(set(args.only))
    out_dir = Path(args.output_dir)

    if args.list or not args.run:
        for r in runs:
            print(f"  {r['tag']:<52} {json.dumps(r['overrides'])}")
        print(f"\n{estimate(runs, 600)}")
        print("(estimate assumes ~10 min/run; measure one run first)")
        return 0

    base = yaml.safe_load(args.base_config.read_text()) if args.base_config.exists() else {}
    config_dir = out_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    completed, skipped, failed = 0, 0, []
    started = time.time()

    # Build the work list first, so the executor below only sees real work.
    pending: list[tuple[dict, Path]] = []
    for spec in runs:
        if (out_dir / f"{spec['tag']}.json").exists():
            skipped += 1
            continue
        cfg = deep_merge(deep_merge(copy.deepcopy(DEFAULTS), base), spec["overrides"])
        cfg["seed"] = spec["seed"]
        if args.data_path:
            cfg["data"]["path"] = args.data_path
        if args.meta_steps is not None:
            cfg["train"]["meta_steps"] = args.meta_steps
        cfg["output_dir"] = str(out_dir)
        cfg["device"] = args.device
        if args.cache_dir:
            cfg["data"]["cache_dir"] = args.cache_dir
        config_path = config_dir / f"{spec['tag']}.yaml"
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        pending.append((spec, config_path))

    log.info("%d runs to execute, %d already present, %d worker(s)",
             len(pending), skipped, args.workers)
    if args.dry_run:
        return 0

    # A single run leaves a T4 almost idle: the model is small, so each process
    # holds a few hundred MiB and the card sits far below capacity. Running
    # several configurations at once converts that headroom into throughput
    # without altering any experiment, since the runs are independent and each
    # writes its own result file. Workers are capped by CPU rather than GPU
    # memory -- the hosted runtime has two cores, and each process needs one
    # for data loading, the Python driver and the classical baselines.
    running: list[tuple[dict, subprocess.Popen, Path]] = []
    queue = list(pending)
    launched = 0

    env = dict(os.environ)
    # Each worker keeps to one thread; oversubscribing two cores across several
    # processes makes every one of them slower.
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"

    def launch(spec: dict, config_path: Path) -> None:
        nonlocal launched
        launched += 1
        log_path = out_dir / "run_logs" / f"{spec['tag']}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-u", "-m", "nids_maml.run",
               "--config", str(config_path), "--seed", str(spec["seed"]),
               "--tag", spec["tag"], "--output-dir", str(out_dir)]
        handle = open(log_path, "w", encoding="utf-8")
        proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env,
                                stdout=handle, stderr=subprocess.STDOUT)
        running.append((spec, proc, log_path))
        log.info("[%d/%d] started %s (pid %d)",
                 launched, len(pending), spec["tag"], proc.pid)

    while queue or running:
        while queue and len(running) < args.workers:
            spec, config_path = queue.pop(0)
            launch(spec, config_path)

        time.sleep(5)

        for entry in list(running):
            spec, proc, log_path = entry
            if proc.poll() is None:
                continue
            running.remove(entry)
            if proc.returncode == 0 and (out_dir / f"{spec['tag']}.json").exists():
                completed += 1
                log.info("     done %s  (%d/%d complete)",
                         spec["tag"], completed, len(pending))
            else:
                tail = ""
                try:
                    tail = "".join(log_path.read_text(
                        encoding="utf-8", errors="replace").splitlines(True)[-15:])
                except OSError:
                    pass
                log.error("run failed: %s (exit %s)\n%s",
                          spec["tag"], proc.returncode, tail)
                failed.append({"tag": spec["tag"], "exit_code": proc.returncode,
                               "log": str(log_path), "tail": tail})

            elapsed = time.time() - started
            if completed:
                per_run = elapsed / completed
                left = (len(pending) - completed - len(failed)) * per_run / args.workers
                log.info("     elapsed %.1f min | %.1f min/run | ~%.1f min remaining",
                         elapsed / 60, per_run / 60, left / 60)

    log.info("matrix finished: %d completed, %d skipped, %d failed",
             completed, skipped, len(failed))
    if failed:
        (out_dir / "failed_runs.json").write_text(json.dumps(failed, indent=2),
                                                  encoding="utf-8")
        for f in failed:
            log.error("  %s: %s", f["tag"], f["error"])
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
