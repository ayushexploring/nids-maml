"""Meta-training loop and episodic evaluation.

Differences from the procedure being replaced that affect reported numbers:

* Meta-training episodes are resampled at every step. Previously 140 episodes
  were materialised once and cycled for 400 epochs, which is a small fixed
  training set by any other name and is the most likely source of the reported
  meta-validation divergence.
* Meta-validation uses a *fixed, seeded* episode list, so the validation signal
  is not itself noisy across checkpoints and model selection is meaningful.
* Evaluation runs with dropout disabled and applies the full number of inner
  steps requested.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .episodes import Episode, EpisodeSampler
from .metrics import (
    Interval,
    episode_metrics,
    expected_calibration_error,
    mean_ci,
)

log = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    meta_steps: int = 3000
    meta_batch_size: int = 16
    eval_every: int = 100
    eval_episodes: int = 200
    patience: int = 15           # in evaluations, not steps
    warmup_steps: int = 100
    cosine_schedule: bool = True
    min_lr_factor: float = 0.05
    seed: int = 0


@dataclass
class TrainingHistory:
    steps: list[int] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    train_accuracy: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_accuracy: list[float] = field(default_factory=list)
    val_accuracy_lo: list[float] = field(default_factory=list)
    val_accuracy_hi: list[float] = field(default_factory=list)
    wall_clock: list[float] = field(default_factory=list)

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(
    learner,
    episodes: list[Episode],
    inner_steps: int | None = None,
) -> dict:
    """Score a meta-learner on a fixed episode list.

    Returns per-episode metric arrays alongside their interval estimates. The
    per-episode arrays are kept because the paired significance tests operate
    on them.
    """
    per_episode: dict[str, list[float]] = {
        "accuracy": [], "macro_f1": [], "macro_precision": [],
        "macro_recall": [], "loss": [], "ece": [],
    }
    # Accumulated per global class id, to recover per-family performance.
    class_correct: dict[int, list[float]] = {}

    for episode in episodes:
        out = learner.evaluate_episode(episode, steps=inner_steps)
        m = episode_metrics(out["y_true"], out["y_pred"], episode.n_way)
        for key in ("accuracy", "macro_f1", "macro_precision", "macro_recall"):
            per_episode[key].append(m[key])
        per_episode["loss"].append(out["loss"])
        per_episode["ece"].append(
            expected_calibration_error(out["y_true"], out["probs"])
        )
        for local_label, global_id in enumerate(episode.classes):
            mask = out["y_true"] == local_label
            if mask.any():
                acc = float((out["y_pred"][mask] == local_label).mean())
                class_correct.setdefault(int(global_id), []).append(acc)

    summary = {k: mean_ci(v).as_dict() for k, v in per_episode.items()}
    return {
        "summary": summary,
        "per_episode": {k: np.asarray(v) for k, v in per_episode.items()},
        "per_class_accuracy": {
            cid: mean_ci(v).as_dict() for cid, v in sorted(class_correct.items())
        },
        "n_episodes": len(episodes),
        "inner_steps": inner_steps if inner_steps is not None else learner.inner.steps,
    }


def meta_train(
    learner,
    train_sampler: EpisodeSampler,
    val_sampler: EpisodeSampler,
    cfg: TrainConfig,
    checkpoint_path: Path | str | None = None,
) -> tuple[TrainingHistory, dict]:
    """Meta-train with early stopping on meta-validation accuracy.

    Returns the training history and the best checkpoint's validation result.
    The best checkpoint is restored into ``learner`` before returning, so the
    caller can evaluate on meta-test immediately.
    """
    set_seed(cfg.seed)
    history = TrainingHistory()

    # One fixed validation episode list for the whole run: comparing
    # checkpoints against different episodes would confound model quality with
    # task difficulty.
    val_episodes = val_sampler.fixed_set(cfg.eval_episodes, seed=cfg.seed + 10_000)
    log.info("meta-validation: %d fixed episodes (degenerate class pool: %s)",
             len(val_episodes), val_sampler.is_degenerate)

    base_lr = learner.optimizer.param_groups[0]["lr"]
    best_acc = -np.inf
    best_state = None
    best_step = 0
    best_val: dict = {}
    since_improvement = 0
    start = time.time()

    for step in range(1, cfg.meta_steps + 1):
        lr = _schedule(step, base_lr, cfg)
        for group in learner.optimizer.param_groups:
            group["lr"] = lr

        stats = learner.meta_step(train_sampler.batch(cfg.meta_batch_size))

        if step % cfg.eval_every == 0 or step == cfg.meta_steps:
            val = evaluate(learner, val_episodes)
            acc: dict = val["summary"]["accuracy"]
            history.steps.append(step)
            history.train_loss.append(stats["loss"])
            history.train_accuracy.append(stats["accuracy"])
            history.val_loss.append(val["summary"]["loss"]["mean"])
            history.val_accuracy.append(acc["mean"])
            history.val_accuracy_lo.append(acc["lo"])
            history.val_accuracy_hi.append(acc["hi"])
            history.wall_clock.append(time.time() - start)

            log.info(
                "step %5d | lr %.2e | train loss %.4f acc %.4f | "
                "val loss %.4f acc %.4f [%.4f, %.4f]",
                step, lr, stats["loss"], stats["accuracy"],
                val["summary"]["loss"]["mean"], acc["mean"], acc["lo"], acc["hi"],
            )

            if acc["mean"] > best_acc:
                best_acc = acc["mean"]
                best_step = step
                best_state = {
                    k: {kk: vv.detach().cpu().clone() for kk, vv in v.items()}
                    for k, v in learner.state_dict().items()
                }
                best_val = {
                    "summary": val["summary"],
                    "per_class_accuracy": val["per_class_accuracy"],
                    "n_episodes": val["n_episodes"],
                }
                since_improvement = 0
                if checkpoint_path is not None:
                    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                    torch.save(best_state, checkpoint_path)
            else:
                since_improvement += 1
                if since_improvement >= cfg.patience:
                    log.info("early stopping at step %d (best step %d, acc %.4f)",
                             step, best_step, best_acc)
                    break

    if best_state is not None:
        learner.load_state_dict(best_state)
    best_val["best_step"] = best_step
    best_val["total_steps"] = history.steps[-1] if history.steps else 0
    best_val["wall_clock_seconds"] = time.time() - start
    return history, best_val


def _schedule(step: int, base_lr: float, cfg: TrainConfig) -> float:
    """Linear warmup followed by optional cosine decay."""
    if step <= cfg.warmup_steps:
        return base_lr * step / max(cfg.warmup_steps, 1)
    if not cfg.cosine_schedule:
        return base_lr
    progress = (step - cfg.warmup_steps) / max(cfg.meta_steps - cfg.warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1 + np.cos(np.pi * progress))
    return base_lr * (cfg.min_lr_factor + (1 - cfg.min_lr_factor) * cosine)


def adaptation_curve(
    learner,
    episodes: list[Episode],
    max_steps: int = 10,
) -> dict[int, dict]:
    """Accuracy as a function of inner-loop steps, evaluated at each step count.

    Step 0 is included. At zero steps an N-way head has not yet been told which
    logit corresponds to which episode-local class, so chance-level performance
    there is expected and is not evidence of a defect.
    """
    return {
        steps: {
            k: v for k, v in evaluate(learner, episodes, inner_steps=steps)["summary"].items()
        }
        for steps in range(max_steps + 1)
    }


def save_json(obj, path: Path | str) -> None:
    """Write results as JSON, converting NumPy scalars and arrays."""
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Interval):
            return o.as_dict()
        if isinstance(o, Path):
            return str(o)
        raise TypeError(f"not JSON serialisable: {type(o)}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=default), encoding="utf-8")
    log.info("wrote %s", path)
