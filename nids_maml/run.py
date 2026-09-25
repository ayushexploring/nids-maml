"""Command-line entry point: one config file, one experiment, one results file.

    python -m nids_maml.run --config configs/primary.yaml --seed 0

Every run writes a single JSON file containing the resolved configuration, the
data fingerprint, the training history, and meta-test results with confidence
intervals. Analysis and figures are produced from those files alone, so no
number in the manuscript can originate anywhere but a recorded run.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml

from . import __version__
from .baselines import ClassicalBaseline, SupervisedFinetune
from .data import (
    DatasetBundle,
    PRIMARY_CLASSES,
    SplitSpec,
    fingerprint,
    load_cicids2017,
    make_synthetic,
)
from .episodes import BinaryEpisodeSampler, EpisodeSampler
from .meta import InnerConfig, build_meta_learner
from .metrics import binary_metrics, mean_ci, wilson_interval
from .models import build_model
from .train import TrainConfig, adaptation_curve, evaluate, meta_train, save_json

log = logging.getLogger("nids_maml")

DEFAULTS: dict = {
    "experiment": "primary",
    "data": {
        "path": None,              # None -> synthetic, for smoke tests
        "classes": list(PRIMARY_CLASSES),
        "max_per_class": 60000,
        "split": {"train": 0.70, "val": 0.15, "test": 0.15},
    },
    "episodes": {"n_way": 5, "k_shot": 5, "n_query": 15},
    "model": {
        "name": "transformer",
        "d_model": 128,
        "n_heads": 4,
        "n_blocks": 3,
        "dropout": 0.1,
        "slot_encoding": "sinusoidal",
    },
    "algorithm": {
        "name": "fomaml",
        "inner_steps": 5,
        # 0.1, not the 0.01 used previously. A Transformer base learner does
        # not meta-learn at all at 0.01: the inner loop moves the parameters
        # too little for the meta-gradient to carry usable signal, and
        # meta-validation accuracy stays at chance indefinitely. The earlier
        # implementation tolerated 0.01 only because its encoder had collapsed
        # to a linear stack. Measured as ablation F3.
        "inner_lr": 0.1,
        "learn_lr": False,
        "meta_lr": 0.001,
        "weight_decay": 1e-5,
        "grad_clip": 1.0,
    },
    "train": {
        "meta_steps": 3000,
        "meta_batch_size": 16,
        "eval_every": 100,
        "eval_episodes": 200,
        "patience": 15,
    },
    "evaluation": {
        "test_episodes": 600,
        "adaptation_curve_steps": 10,
        "adaptation_curve_episodes": 200,
        "protocol_b": True,
    },
    "seed": 0,
    "device": "auto",
    "output_dir": "results",
}


def deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def load_data(cfg: dict) -> DatasetBundle:
    data_cfg = cfg["data"]
    if not data_cfg.get("path"):
        log.warning("no data.path configured -- using synthetic data (smoke test only)")
        return make_synthetic(
            n_classes=max(cfg["episodes"]["n_way"], 5), seed=cfg["seed"]
        )
    return load_cicids2017(
        data_cfg["path"],
        classes=tuple(data_cfg["classes"]),
        split=SplitSpec(**data_cfg["split"]),
        seed=cfg["seed"],
        max_per_class=data_cfg.get("max_per_class"),
    )


def build_learner(cfg: dict, bundle: DatasetBundle, device: torch.device):
    algo = cfg["algorithm"]["name"].lower()
    n_way = cfg["episodes"]["n_way"]

    if algo.startswith("classical:"):
        return ClassicalBaseline(kind=algo.split(":", 1)[1], seed=cfg["seed"])

    inner = InnerConfig(
        steps=cfg["algorithm"]["inner_steps"],
        lr=cfg["algorithm"]["inner_lr"],
        learn_lr=cfg["algorithm"].get("learn_lr", False),
    )
    shared = {
        "meta_lr": cfg["algorithm"]["meta_lr"],
        "weight_decay": cfg["algorithm"]["weight_decay"],
        "grad_clip": cfg["algorithm"]["grad_clip"],
        "device": device,
    }

    model_cfg = {k: v for k, v in cfg["model"].items() if k != "name"}
    model_name = cfg["model"]["name"]

    if algo == "supervised":
        # The head must span the pool's global classes for pre-training, but
        # episodes are n_way; the head is reinitialised per episode anyway, so
        # it is sized to the larger of the two.
        n_out = max(n_way, len(bundle.class_names))
        model = build_model(model_name, bundle.n_features, n_out, **model_cfg)
        learner = SupervisedFinetune(model, inner, **shared)
        learner.set_pool(bundle.X_train, bundle.y_train)
        return learner

    model = build_model(model_name, bundle.n_features, n_way, **model_cfg)
    return build_meta_learner(algo, model, inner, **shared)


def run_protocol_b(learner, bundle: DatasetBundle, cfg: dict) -> dict:
    """Benign-vs-family binary evaluation on the meta-test pool.

    Reported per family as a mean over episodes with a confidence interval, and
    additionally as a Wilson interval on the pooled query instances, so a
    perfect score is never presented as a zero-error rate.
    """
    names = bundle.class_names
    if "Benign" not in names:
        log.warning("no Benign class present; skipping Protocol B")
        return {}
    benign = names.index("Benign")
    results: dict = {}

    for attack_id, attack_name in enumerate(names):
        if attack_id == benign:
            continue
        try:
            sampler = BinaryEpisodeSampler(
                bundle.X_test, bundle.y_test,
                benign_class=benign, attack_class=attack_id,
                k_shot=cfg["episodes"]["k_shot"],
                n_query=cfg["episodes"]["n_query"],
                seed=cfg["seed"] + 777,
            )
        except ValueError as exc:
            log.warning("skipping Protocol B for %s: %s", attack_name, exc)
            continue

        episodes = sampler.fixed_set(cfg["evaluation"]["test_episodes"], seed=cfg["seed"] + 777)
        per_episode: dict[str, list[float]] = {}
        pooled_correct = 0
        pooled_total = 0
        if hasattr(learner, "evaluate_batch"):
            outputs = learner.evaluate_batch(episodes)
        else:
            outputs = [learner.evaluate_episode(ep) for ep in episodes]
        for out in outputs:
            m = binary_metrics(out["y_true"], out["y_pred"], out["probs"])
            for key, value in m.items():
                per_episode.setdefault(key, []).append(value)
            pooled_correct += int((out["y_pred"] == out["y_true"]).sum())
            pooled_total += len(out["y_true"])

        results[attack_name] = {
            "per_episode": {k: mean_ci(v).as_dict() for k, v in per_episode.items()},
            "pooled_accuracy_wilson": wilson_interval(pooled_correct, pooled_total).as_dict(),
            "n_episodes": len(episodes),
        }
        log.info("Protocol B %-12s acc %s  f1 %s",
                 attack_name,
                 mean_ci(per_episode["accuracy"]),
                 mean_ci(per_episode["f1"]))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="YAML config file")
    parser.add_argument("--seed", type=int, help="override config seed")
    parser.add_argument("--data-path", type=str, help="override data.path")
    parser.add_argument("--output-dir", type=str, help="override output_dir")
    parser.add_argument("--tag", type=str, default="", help="suffix for the result filename")
    parser.add_argument("--meta-steps", type=int, help="override train.meta_steps")
    parser.add_argument("--device", type=str, help="cpu, cuda or auto")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = dict(DEFAULTS)
    if args.config:
        cfg = deep_merge(cfg, yaml.safe_load(args.config.read_text()) or {})
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.data_path:
        cfg["data"]["path"] = args.data_path
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.meta_steps is not None:
        cfg["train"]["meta_steps"] = args.meta_steps
    if args.device:
        cfg["device"] = args.device

    device = resolve_device(cfg["device"])
    log.info("experiment=%s algorithm=%s model=%s seed=%d device=%s",
             cfg["experiment"], cfg["algorithm"]["name"],
             cfg["model"]["name"], cfg["seed"], device)

    bundle = load_data(cfg)
    fp = fingerprint(bundle)
    log.info("data fingerprint %s | %d features | classes %s",
             fp, bundle.n_features, bundle.class_names)

    ep = cfg["episodes"]
    train_sampler = EpisodeSampler(
        bundle.X_train, bundle.y_train, ep["n_way"], ep["k_shot"], ep["n_query"],
        seed=cfg["seed"],
    )
    val_sampler = EpisodeSampler(
        bundle.X_val, bundle.y_val, ep["n_way"], ep["k_shot"], ep["n_query"],
        seed=cfg["seed"] + 1,
    )
    test_sampler = EpisodeSampler(
        bundle.X_test, bundle.y_test, ep["n_way"], ep["k_shot"], ep["n_query"],
        seed=cfg["seed"] + 2,
    )
    if train_sampler.is_degenerate:
        log.warning(
            "class pool size equals n_way (%d): every episode spans the same "
            "classes, so no class is ever novel at meta-test time",
            ep["n_way"],
        )

    learner = build_learner(cfg, bundle, device)

    out_dir = Path(cfg["output_dir"])
    tag = args.tag or f"{cfg['experiment']}_{cfg['algorithm']['name']}_seed{cfg['seed']}"
    checkpoint = out_dir / "checkpoints" / f"{tag}.pt"

    is_classical = cfg["algorithm"]["name"].lower().startswith("classical:")
    train_cfg = TrainConfig(seed=cfg["seed"], **cfg["train"])
    if is_classical:
        log.info("classical baseline: no meta-training")
        history, best_val = None, {}
    else:
        history, best_val = meta_train(
            learner, train_sampler, val_sampler, train_cfg, checkpoint
        )

    # --- meta-test -----------------------------------------------------------
    # Every method is scored on the episode list produced by this seed, which is
    # what makes the paired tests in analyse.py valid.
    test_episodes = test_sampler.fixed_set(
        cfg["evaluation"]["test_episodes"], seed=cfg["seed"] + 20_000
    )
    test_result = evaluate(learner, test_episodes)
    log.info("META-TEST accuracy %s over %d episodes",
             mean_ci(test_result["per_episode"]["accuracy"]),
             len(test_episodes))

    curve = {}
    if cfg["evaluation"]["adaptation_curve_steps"] and not is_classical:
        curve_episodes = test_sampler.fixed_set(
            cfg["evaluation"]["adaptation_curve_episodes"], seed=cfg["seed"] + 30_000
        )
        curve = adaptation_curve(
            learner, curve_episodes, cfg["evaluation"]["adaptation_curve_steps"]
        )

    protocol_b = {}
    if cfg["evaluation"].get("protocol_b"):
        protocol_b = run_protocol_b(learner, bundle, cfg)

    payload = {
        "config": cfg,
        "version": __version__,
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "platform": platform.platform(),
            "device": str(device),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "data": {
            "fingerprint": fp,
            "meta": bundle.meta,
            "n_features": bundle.n_features,
            "degenerate_class_pool": train_sampler.is_degenerate,
        },
        "history": history.as_dict() if history else None,
        "validation_best": best_val,
        "test": {
            "summary": test_result["summary"],
            "per_class_accuracy": test_result["per_class_accuracy"],
            "n_episodes": test_result["n_episodes"],
            # Retained so analyse.py can run paired tests across runs.
            "per_episode_accuracy": test_result["per_episode"]["accuracy"],
            "per_episode_macro_f1": test_result["per_episode"]["macro_f1"],
        },
        "adaptation_curve": curve,
        "protocol_b": protocol_b,
    }
    save_json(payload, out_dir / f"{tag}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
