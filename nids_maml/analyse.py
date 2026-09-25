"""Turn result files into the manuscript's tables and figures.

Nothing here recomputes a model result. It reads the JSON files written by
``nids_maml.run`` and aggregates them, so every number in the manuscript is
traceable to a recorded run and its data fingerprint.

    python -m nids_maml.analyse --results results --out paper_assets
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

from .metrics import Interval, holm_bonferroni, mean_ci, paired_test

log = logging.getLogger(__name__)

# Display names for the methods, in the order the baseline table should read.
METHOD_ORDER = [
    "fomaml", "maml", "reptile", "protonet",
    "supervised", "classical:random_forest", "classical:gradient_boosting",
    "classical:logistic",
]
METHOD_LABELS = {
    "fomaml": "Transformer + FOMAML (proposed)",
    "maml": "Transformer + second-order MAML",
    "reptile": "Transformer + Reptile",
    "protonet": "Prototypical Networks",
    "supervised": "Transformer, supervised pre-training + fine-tune",
    "classical:random_forest": "Random Forest (support set only)",
    "classical:gradient_boosting": "Gradient Boosting (support set only)",
    "classical:logistic": "Logistic Regression (support set only)",
}


def load_results(results_dir: Path) -> list[dict]:
    """Read every result JSON in a directory."""
    files = sorted(Path(results_dir).glob("*.json"))
    out = []
    for f in files:
        try:
            out.append(json.loads(f.read_text(encoding="utf-8")))
        except json.JSONDecodeError as exc:
            log.warning("skipping unreadable %s: %s", f.name, exc)
    log.info("loaded %d result files from %s", len(out), results_dir)
    return out


def _key(result: dict) -> tuple[str, str]:
    return result["config"]["experiment"], result["config"]["algorithm"]["name"].lower()


def check_comparability(results: list[dict]) -> list[str]:
    """Report any reason a set of runs cannot be compared directly.

    Runs are only comparable if they used the same data split and the same
    episode configuration. A paired test across runs that disagree on either is
    meaningless, so the condition is checked rather than assumed.
    """
    problems = []
    fingerprints = defaultdict(set)
    episodes = defaultdict(set)
    for r in results:
        seed = r["config"]["seed"]
        fingerprints[seed].add(r["data"]["fingerprint"])
        ep = r["config"]["episodes"]
        episodes[seed].add((ep["n_way"], ep["k_shot"], ep["n_query"]))
    for seed, fps in fingerprints.items():
        if len(fps) > 1:
            problems.append(
                f"seed {seed}: runs used {len(fps)} different data splits {sorted(fps)}"
            )
    for seed, eps in episodes.items():
        if len(eps) > 1:
            problems.append(f"seed {seed}: runs used different episode shapes {sorted(eps)}")
    return problems


def aggregate_over_seeds(results: list[dict], metric: str = "accuracy") -> dict:
    """Group runs by method and pool their meta-test episodes across seeds.

    Two intervals are produced. The within-seed interval treats episodes as the
    unit of analysis and answers "how precisely is this seed's model measured".
    The across-seed interval treats the seed as the unit and answers "how much
    does the result depend on initialisation and split" -- the quantity a
    reader actually needs, and the one the previous single-run results could
    not provide.
    """
    by_method: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_method[r["config"]["algorithm"]["name"].lower()].append(r)

    table = {}
    for method, runs in by_method.items():
        runs = sorted(runs, key=lambda r: r["config"]["seed"])
        per_seed_means = []
        pooled: list[float] = []
        for r in runs:
            values = r["test"].get(f"per_episode_{metric}")
            if values is None:
                continue
            per_seed_means.append(float(np.mean(values)))
            pooled.extend(values)
        if not per_seed_means:
            continue
        table[method] = {
            "label": METHOD_LABELS.get(method, method),
            "n_seeds": len(per_seed_means),
            "seeds": [r["config"]["seed"] for r in runs],
            "per_seed_means": per_seed_means,
            "across_seeds": mean_ci(per_seed_means).as_dict(),
            "pooled_episodes": mean_ci(pooled).as_dict(),
            "n_episodes_total": len(pooled),
        }
    return table


def baseline_comparison(
    results: list[dict], reference: str = "fomaml", metric: str = "accuracy"
) -> dict:
    """Paired comparison of every method against the reference, per seed.

    Pairing is done within a seed, episode by episode: both methods saw the
    identical episode list because ``fixed_set`` derives it from the seed. The
    per-seed p-values are combined across the family of comparisons with a
    Holm-Bonferroni correction.
    """
    by_seed: dict[int, dict[str, dict]] = defaultdict(dict)
    for r in results:
        by_seed[r["config"]["seed"]][r["config"]["algorithm"]["name"].lower()] = r

    comparisons: dict[str, dict] = {}
    for method in {m for seed in by_seed.values() for m in seed}:
        if method == reference:
            continue
        per_seed = []
        for seed, methods in sorted(by_seed.items()):
            if reference not in methods or method not in methods:
                continue
            a = np.asarray(methods[reference]["test"][f"per_episode_{metric}"])
            b = np.asarray(methods[method]["test"][f"per_episode_{metric}"])
            if a.shape != b.shape:
                log.warning("seed %s: %s and %s have different episode counts; skipped",
                            seed, reference, method)
                continue
            per_seed.append({"seed": seed, **paired_test(a, b)})
        if per_seed:
            comparisons[method] = {
                "label": METHOD_LABELS.get(method, method),
                "per_seed": per_seed,
                "mean_difference": float(np.mean([p["mean_difference"] for p in per_seed])),
                # Combine across seeds by taking the least favourable p-value:
                # a claim should hold on every seed, not on the luckiest one.
                "max_p_value": float(max(p["t_p_value"] for p in per_seed)),
            }

    corrected = holm_bonferroni({m: c["max_p_value"] for m, c in comparisons.items()})
    for method, c in comparisons.items():
        c["holm"] = corrected[method]
    return comparisons


def format_interval(d: dict, pct: bool = True) -> str:
    scale = 100.0 if pct else 1.0
    fmt = "{:.2f}" if pct else "{:.4f}"
    return (f"{fmt.format(d['mean'] * scale)} "
            f"[{fmt.format(d['lo'] * scale)}, {fmt.format(d['hi'] * scale)}]")


def latex_baseline_table(table: dict, comparisons: dict, metric: str = "accuracy") -> str:
    """Render the baseline table as LaTeX (booktabs)."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Meta-test %s under the 5-way, 5-shot protocol. "
        r"Intervals are 95\%% confidence intervals across seeds. "
        r"$\Delta$ is the mean per-episode difference from the proposed method; "
        r"$p$ is the least favourable paired $t$-test across seeds after "
        r"Holm--Bonferroni correction.}" % metric,
        r"\label{tab:baselines}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Method & %s (\%%) & $\Delta$ (pp) & $p$ \\" % metric.replace("_", " ").title(),
        r"\midrule",
    ]
    for method in METHOD_ORDER:
        if method not in table:
            continue
        row = table[method]
        cells = [row["label"], format_interval(row["across_seeds"])]
        if method in comparisons:
            c = comparisons[method]
            cells.append(f"{-c['mean_difference'] * 100:+.2f}")
            p = c["holm"]["p_value"]
            star = r"$^{*}$" if c["holm"]["significant"] else ""
            cells.append((r"$<$0.001" if p < 0.001 else f"{p:.3f}") + star)
        else:
            cells += [r"---", r"---"]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def markdown_baseline_table(table: dict, comparisons: dict) -> str:
    rows = ["| Method | Accuracy (%) | Δ (pp) | p |", "|---|---|---|---|"]
    for method in METHOD_ORDER:
        if method not in table:
            continue
        row = table[method]
        cells = [row["label"], format_interval(row["across_seeds"])]
        if method in comparisons:
            c = comparisons[method]
            cells.append(f"{-c['mean_difference'] * 100:+.2f}")
            p = c["holm"]["p_value"]
            cells.append(("<0.001" if p < 0.001 else f"{p:.3f}")
                         + ("*" if c["holm"]["significant"] else ""))
        else:
            cells += ["—", "—"]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def ablation_table(results: list[dict]) -> dict:
    """Group ablation runs by the factor they varied.

    A run participates in an ablation when its ``experiment`` name begins with
    ``ablation_``; the remainder names the factor, e.g. ``ablation_k_shot``.
    """
    factors: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        name = r["config"]["experiment"]
        if not name.startswith("ablation_"):
            continue
        factor = name[len("ablation_"):]
        level = _level_of(r, factor)
        values = r["test"].get("per_episode_accuracy")
        if values is not None:
            factors[factor][str(level)].append(float(np.mean(values)))

    return {
        factor: {
            level: {**mean_ci(v).as_dict(), "n_seeds": len(v)}
            for level, v in sorted(levels.items(), key=lambda kv: _sortable(kv[0]))
        }
        for factor, levels in factors.items()
    }


def _level_of(result: dict, factor: str):
    """Read the varied factor's value out of a run's config."""
    cfg = result["config"]
    lookup = {
        "k_shot": cfg["episodes"]["k_shot"],
        "n_way": cfg["episodes"]["n_way"],
        "inner_steps": cfg["algorithm"]["inner_steps"],
        "inner_lr": cfg["algorithm"]["inner_lr"],
        "n_blocks": cfg["model"]["n_blocks"],
        "d_model": cfg["model"]["d_model"],
        "n_heads": cfg["model"]["n_heads"],
        "slot_encoding": cfg["model"].get("slot_encoding"),
        "model": cfg["model"]["name"],
        "order": cfg["algorithm"]["name"],
        "learn_lr": cfg["algorithm"].get("learn_lr"),
    }
    if factor not in lookup:
        raise KeyError(
            f"ablation factor '{factor}' is not mapped to a config field; "
            f"add it to _level_of (known: {sorted(lookup)})"
        )
    return lookup[factor]


def _sortable(level: str):
    try:
        return (0, float(level))
    except ValueError:
        return (1, level)


def adaptation_curves(results: list[dict]) -> dict:
    """Collect accuracy-versus-inner-step curves, averaged over seeds."""
    by_method: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        curve = r.get("adaptation_curve") or {}
        method = r["config"]["algorithm"]["name"].lower()
        for steps, summary in curve.items():
            by_method[method][int(steps)].append(summary["accuracy"]["mean"])
    return {
        method: {
            "steps": sorted(steps_map),
            "accuracy": [float(np.mean(steps_map[s])) for s in sorted(steps_map)],
            "n_seeds": [len(steps_map[s]) for s in sorted(steps_map)],
        }
        for method, steps_map in by_method.items()
    }


def protocol_b_table(results: list[dict], method: str = "fomaml") -> dict:
    """Per-family binary results, averaged over seeds, with Wilson bounds."""
    families: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    wilson: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        if r["config"]["algorithm"]["name"].lower() != method:
            continue
        for family, payload in (r.get("protocol_b") or {}).items():
            for metric, interval in payload["per_episode"].items():
                families[family][metric].append(interval["mean"])
            wilson[family].append(payload["pooled_accuracy_wilson"])
    return {
        family: {
            **{m: mean_ci(v).as_dict() for m, v in metrics.items()},
            "pooled_accuracy_wilson_lo": float(np.mean([w["lo"] for w in wilson[family]])),
            "n_seeds": len(wilson[family]),
        }
        for family, metrics in families.items()
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("paper_assets"))
    parser.add_argument("--reference", default="fomaml")
    parser.add_argument("--figures", action="store_true", help="also render figures")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    results = load_results(args.results)
    if not results:
        log.error("no results found in %s", args.results)
        return 1

    problems = check_comparability(results)
    for p in problems:
        log.warning("COMPARABILITY: %s", p)

    args.out.mkdir(parents=True, exist_ok=True)
    table = aggregate_over_seeds(results)
    comparisons = baseline_comparison(results, args.reference)
    ablations = ablation_table(results)
    curves = adaptation_curves(results)
    protocol_b = protocol_b_table(results, args.reference)

    summary = {
        "n_results": len(results),
        "comparability_warnings": problems,
        "baselines": table,
        "comparisons": comparisons,
        "ablations": ablations,
        "adaptation_curves": curves,
        "protocol_b": protocol_b,
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (args.out / "table_baselines.tex").write_text(
        latex_baseline_table(table, comparisons), encoding="utf-8"
    )
    (args.out / "table_baselines.md").write_text(
        markdown_baseline_table(table, comparisons), encoding="utf-8"
    )

    print("\n" + markdown_baseline_table(table, comparisons) + "\n")
    for factor, levels in ablations.items():
        print(f"Ablation: {factor}")
        for level, stats in levels.items():
            print(f"   {level:>12} : {format_interval(stats)}  (n={stats['n_seeds']} seeds)")
    if protocol_b:
        print("\nProtocol B (binary, per family):")
        for family, stats in protocol_b.items():
            print(f"   {family:>12} : F1 {format_interval(stats['f1'], pct=False)}")

    if args.figures:
        from .figures import render_all
        render_all(results, summary, args.out)

    log.info("wrote analysis to %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
