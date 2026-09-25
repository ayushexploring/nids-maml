"""Non-meta baselines evaluated under the identical episodic protocol.

The comparison a reviewer will insist on is not "our method versus published
numbers from other papers". It is: given the *same* N*K labelled instances, the
*same* splits and the *same* episode stream, does episodic meta-learning beat
simply training on those instances? These baselines answer that question.

Two families are provided:

* ``SupervisedFinetune`` -- a network pre-trained on the meta-training pool with
  ordinary supervised learning, then fine-tuned on each episode's support set.
  This isolates the contribution of episodic meta-training, holding the
  architecture and the label budget fixed.
* ``ClassicalBaseline`` -- Random Forest / gradient boosting / logistic
  regression fitted from scratch on each episode's support set. No pre-training,
  no shared representation: the honest floor.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from .episodes import Episode
from .meta import InnerConfig, adapt, clone_params, forward_with

log = logging.getLogger(__name__)


class SupervisedFinetune:
    """Conventionally pre-trained encoder, fine-tuned per episode.

    Pre-training uses the meta-training pool's *global* labels. Because episode
    labels are local and permuted, the pre-trained classification head cannot
    transfer; it is reinitialised for each episode and the encoder is fine-tuned
    with it. This is the standard transfer-learning control for few-shot claims.
    """

    name = "supervised"

    def __init__(
        self,
        model: torch.nn.Module,
        inner: InnerConfig,
        meta_lr: float = 1e-3,
        weight_decay: float = 1e-5,
        grad_clip: float | None = 1.0,
        device: torch.device | str = "cpu",
        **_ignored,
    ) -> None:
        self.model = model.to(device)
        self.device = torch.device(device)
        self.inner = inner
        self.grad_clip = grad_clip
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=meta_lr, weight_decay=weight_decay
        )
        self._pool: tuple[np.ndarray, np.ndarray] | None = None

    def set_pool(self, X: np.ndarray, y: np.ndarray) -> None:
        """Supply the meta-training flow pool used for supervised pre-training."""
        self._pool = (X, y)

    def meta_step(self, episodes: list[Episode]) -> dict[str, float]:
        """One ordinary supervised minibatch step over the training pool.

        Episodes are accepted so the training loop is interchangeable with the
        meta-learners', but only their count sets the batch size; the labels
        used are the pool's global labels.
        """
        if self._pool is None:
            raise RuntimeError("call set_pool() before training the supervised baseline")
        X, y = self._pool
        batch = len(episodes) * 32
        idx = np.random.choice(len(y), size=min(batch, len(y)), replace=False)
        xb = torch.as_tensor(X[idx], device=self.device)
        yb = torch.as_tensor(y[idx], device=self.device)

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(xb)
        loss = F.cross_entropy(logits, yb)
        loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        return {
            "loss": float(loss.item()),
            "accuracy": float((logits.argmax(1) == yb).float().mean().item()),
        }

    @torch.enable_grad()
    def evaluate_episode(self, episode: Episode, steps: int | None = None) -> dict:
        self.model.eval()
        n_steps = self.inner.steps if steps is None else steps
        sx = torch.as_tensor(episode.support_x, device=self.device)
        sy = torch.as_tensor(episode.support_y, device=self.device)
        qx = torch.as_tensor(episode.query_x, device=self.device)
        qy = torch.as_tensor(episode.query_y, device=self.device)

        params = {n: p.detach().clone().requires_grad_(True)
                  for n, p in self.model.named_parameters()}
        # Reinitialise the head: its output slots carry no meaning for this
        # episode's local label assignment.
        for name in params:
            if name.startswith("head."):
                params[name] = torch.zeros_like(params[name]).requires_grad_(True)

        cfg = InnerConfig(steps=n_steps, lr=self.inner.lr, first_order=True)
        adapted = adapt(self.model, params, sx, sy, cfg, create_graph=False)
        with torch.no_grad():
            logits = forward_with(self.model, adapted, qx)
        return {
            "y_true": episode.query_y,
            "y_pred": logits.argmax(-1).cpu().numpy(),
            "probs": logits.softmax(-1).cpu().numpy(),
            "loss": float(F.cross_entropy(logits, qy).item()),
        }

    def state_dict(self) -> dict:
        return {"model": self.model.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self.model.load_state_dict(state["model"])


class ClassicalBaseline:
    """Fit a classical classifier from scratch on each episode's support set.

    No meta-training happens, so ``meta_step`` is a no-op and the training loop
    should be run for zero steps. With K=5 and N=5 the model sees 25 labelled
    instances, exactly the budget available to the meta-learners at test time.
    """

    name = "classical"

    def __init__(self, kind: str = "random_forest", seed: int = 0, **_ignored) -> None:
        self.kind = kind
        self.seed = seed
        self.inner = InnerConfig(steps=0, lr=0.0)

    def _fresh(self):
        if self.kind == "random_forest":
            return RandomForestClassifier(
                n_estimators=300, random_state=self.seed, n_jobs=-1
            )
        if self.kind == "gradient_boosting":
            return HistGradientBoostingClassifier(random_state=self.seed)
        if self.kind == "logistic":
            return LogisticRegression(max_iter=2000, random_state=self.seed)
        raise ValueError(f"unknown classical baseline '{self.kind}'")

    def meta_step(self, episodes: list[Episode]) -> dict[str, float]:
        return {"loss": float("nan"), "accuracy": float("nan")}

    def evaluate_episode(self, episode: Episode, steps: int | None = None) -> dict:
        clf = self._fresh()
        clf.fit(episode.support_x, episode.support_y)
        y_pred = clf.predict(episode.query_x)
        probs = np.zeros((len(y_pred), episode.n_way), dtype=float)
        if hasattr(clf, "predict_proba"):
            raw = clf.predict_proba(episode.query_x)
            for col, cls in enumerate(clf.classes_):
                probs[:, int(cls)] = raw[:, col]
        else:
            probs[np.arange(len(y_pred)), y_pred] = 1.0
        eps = 1e-12
        loss = float(
            -np.log(np.clip(probs[np.arange(len(y_pred)), episode.query_y], eps, 1.0)).mean()
        )
        return {
            "y_true": episode.query_y,
            "y_pred": y_pred.astype(np.int64),
            "probs": probs,
            "loss": loss,
        }

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:
        return None
