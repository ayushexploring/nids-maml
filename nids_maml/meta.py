"""Meta-learning algorithms: MAML (first and second order), Reptile, ProtoNet.

The inner loop is *functional*. Adapted parameters are produced by
``torch.autograd.grad`` with ``create_graph=True`` and applied through
``torch.func.functional_call``, so the adaptation trajectory stays on the
autograd graph and the true second-order meta-gradient is available. The
implementation being replaced adapted weights by calling ``set_weights`` with
NumPy arrays, which severs the graph; the meta-gradient it computed was
therefore a first-order approximation by accident rather than by choice, and
second-order MAML could not be run at all.

A second defect corrected here: in the previous ``evaluate`` routine the weight
assignment sat outside the inner-step loop, so every step recomputed the
gradient at the unchanged initialisation and exactly one update was ever
applied. Evaluations labelled "five inner steps" were one-step evaluations. The
``adapt`` function below takes the loop structure as its single source of truth
for both meta-training and evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, grad, grad_and_value, vmap

from .episodes import Episode

ParamDict = dict[str, torch.Tensor]


def clone_params(model: nn.Module) -> ParamDict:
    """Trainable parameters as a name -> tensor mapping (no copy of storage)."""
    return {name: p for name, p in model.named_parameters()}


def buffers_of(model: nn.Module) -> ParamDict:
    return {name: b for name, b in model.named_buffers()}


def forward_with(
    model: nn.Module, params: ParamDict, x: torch.Tensor
) -> torch.Tensor:
    """Run ``model`` using ``params`` instead of its own attached parameters."""
    return functional_call(model, {**params, **buffers_of(model)}, (x,))


@dataclass
class InnerConfig:
    steps: int = 5
    lr: float = 0.01
    first_order: bool = True
    # Per-parameter learned inner learning rates (MAML++ / Meta-SGD style).
    learn_lr: bool = False


def adapt(
    model: nn.Module,
    params: ParamDict,
    support_x: torch.Tensor,
    support_y: torch.Tensor,
    cfg: InnerConfig,
    lrs: ParamDict | None = None,
    create_graph: bool | None = None,
    return_trajectory: bool = False,
) -> ParamDict | list[ParamDict]:
    """Run the inner loop and return the adapted parameters.

    Args:
        model: the module supplying the architecture (its own parameters are
            not read or modified).
        params: the initialisation to adapt from.
        support_x, support_y: the support set.
        cfg: inner-loop configuration.
        lrs: per-parameter learning rates when ``cfg.learn_lr`` is set.
        create_graph: retain the graph through the update, enabling the
            second-order meta-gradient. Defaults to ``not cfg.first_order``.
        return_trajectory: return the parameters after every step, including
            step 0, which is what the adaptation-curve figure needs.

    Every step recomputes the gradient at the *current* adapted parameters.
    """
    if create_graph is None:
        create_graph = not cfg.first_order

    names = list(params)
    current = dict(params)
    trajectory = [dict(current)] if return_trajectory else None

    for _ in range(cfg.steps):
        logits = forward_with(model, current, support_x)
        loss = F.cross_entropy(logits, support_y)
        grads = torch.autograd.grad(
            loss,
            [current[n] for n in names],
            create_graph=create_graph,
            allow_unused=True,
        )
        updated = {}
        # Named ``g`` rather than ``grad``: the module-level ``grad`` imported
        # from torch.func would otherwise be shadowed inside this function.
        for name, g in zip(names, grads):
            if g is None:
                updated[name] = current[name]
                continue
            step = lrs[name] if (cfg.learn_lr and lrs is not None) else cfg.lr
            updated[name] = current[name] - step * g
        current = updated
        if trajectory is not None:
            trajectory.append(dict(current))

    return trajectory if return_trajectory else current


class MAML:
    """Model-agnostic meta-learning over a batch of episodes.

    ``first_order=True`` reproduces FOMAML; ``first_order=False`` retains the
    second-order term. Both are exercised by the same code path, so the
    comparison between them isolates exactly the approximation and nothing else.
    """

    name = "maml"

    def __init__(
        self,
        model: nn.Module,
        inner: InnerConfig,
        meta_lr: float = 1e-3,
        weight_decay: float = 1e-5,
        grad_clip: float | None = 1.0,
        device: torch.device | str = "cpu",
        vectorized: bool = True,
    ) -> None:
        self.model = model.to(device)
        self.inner = inner
        self.device = torch.device(device)
        self.grad_clip = grad_clip
        self.vectorized = vectorized

        self.inner_lrs: ParamDict | None = None
        trainable: list[torch.Tensor] = list(self.model.parameters())
        if inner.learn_lr:
            # One learning rate per parameter tensor, meta-learned alongside
            # the initialisation. Stored as log-values to keep them positive.
            self._log_lrs = nn.ParameterDict(
                {
                    name.replace(".", "__"): nn.Parameter(
                        torch.full((), float(np.log(inner.lr)))
                    )
                    for name, _ in self.model.named_parameters()
                }
            ).to(device)
            trainable += list(self._log_lrs.parameters())

        self.optimizer = torch.optim.AdamW(
            trainable, lr=meta_lr, weight_decay=weight_decay
        )

    def _current_lrs(self) -> ParamDict | None:
        if not self.inner.learn_lr:
            return None
        return {
            name: self._log_lrs[name.replace(".", "__")].exp()
            for name, _ in self.model.named_parameters()
        }

    def _to_torch(self, episode: Episode) -> tuple[torch.Tensor, ...]:
        return (
            torch.as_tensor(episode.support_x, device=self.device),
            torch.as_tensor(episode.support_y, device=self.device),
            torch.as_tensor(episode.query_x, device=self.device),
            torch.as_tensor(episode.query_y, device=self.device),
        )

    def _stack(self, episodes: list[Episode]) -> tuple[torch.Tensor, ...]:
        """Stack a meta-batch along a leading task axis for vmap."""
        as_t = lambda arrays: torch.as_tensor(np.stack(arrays), device=self.device)
        return (
            as_t([e.support_x for e in episodes]),
            as_t([e.support_y for e in episodes]),
            as_t([e.query_x for e in episodes]),
            as_t([e.query_y for e in episodes]),
        )

    def _meta_step_vectorized(self, episodes: list[Episode]) -> dict[str, float]:
        """Outer update with all tasks in the meta-batch adapted in parallel.

        Mathematically identical to the sequential path; the tasks in a
        meta-batch are independent, so mapping over them changes only the
        order of floating-point reductions. The gain is that the inner loop
        issues one batched kernel per step instead of one per task, which is
        what makes the full baseline and ablation matrix affordable.
        """
        buffers = {n: b.detach() for n, b in self.model.named_buffers()}
        model = self.model
        first_order = self.inner.first_order
        steps, base_lr = self.inner.steps, self.inner.lr

        def support_loss(p, x, y):
            return F.cross_entropy(functional_call(model, {**p, **buffers}, (x,)), y)

        def per_task(p, lrs, sx, sy, qx, qy):
            for _ in range(steps):
                g = grad(support_loss)(p, sx, sy)
                p = {
                    k: p[k] - (lrs[k] if lrs is not None else base_lr)
                    * (g[k].detach() if first_order else g[k])
                    for k in p
                }
            logits = functional_call(model, {**p, **buffers}, (qx,))
            correct = (logits.argmax(-1) == qy).float().mean()
            return F.cross_entropy(logits, qy), correct

        sx, sy, qx, qy = self._stack(episodes)
        lrs = self._current_lrs()

        def meta_objective(p, lrs):
            losses, accs = vmap(
                per_task, in_dims=(None, None, 0, 0, 0, 0), randomness="different"
            )(p, lrs, sx, sy, qx, qy)
            return losses.mean(), accs.mean()

        params = {n: p for n, p in self.model.named_parameters()}
        argnums = (0, 1) if lrs is not None else 0
        grads, (_, acc) = grad_and_value(meta_objective, argnums=argnums, has_aux=True)(
            params, lrs
        )
        param_grads = grads[0] if lrs is not None else grads

        self.optimizer.zero_grad(set_to_none=True)
        for name, p in self.model.named_parameters():
            p.grad = param_grads[name]
        if lrs is not None:
            for name, _ in self.model.named_parameters():
                key = name.replace(".", "__")
                # d/d(log lr) = lr * d/d(lr), by the chain rule through exp().
                self._log_lrs[key].grad = grads[1][name] * lrs[name].detach()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()

        with torch.no_grad():
            loss_value = meta_objective(
                {n: p.detach() for n, p in self.model.named_parameters()},
                None if lrs is None else {k: v.detach() for k, v in lrs.items()},
            )[0]
        return {"loss": float(loss_value), "accuracy": float(acc)}

    def meta_step(self, episodes: list[Episode]) -> dict[str, float]:
        """One outer update over a meta-batch. Returns loss and accuracy."""
        self.model.train()
        if self.vectorized:
            return self._meta_step_vectorized(episodes)

        self.optimizer.zero_grad(set_to_none=True)

        params = clone_params(self.model)
        lrs = self._current_lrs()
        total_loss = 0.0
        total_acc = 0.0

        for episode in episodes:
            sx, sy, qx, qy = self._to_torch(episode)
            adapted = adapt(self.model, params, sx, sy, self.inner, lrs)
            logits = forward_with(self.model, adapted, qx)
            loss = F.cross_entropy(logits, qy) / len(episodes)
            # Backward per episode rather than accumulating the graph across
            # the whole meta-batch: identical gradient, far less peak memory,
            # which is what allows second-order runs at this meta-batch size.
            loss.backward()
            total_loss += loss.item()
            total_acc += (logits.argmax(1) == qy).float().mean().item() / len(episodes)

        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        return {"loss": total_loss, "accuracy": total_acc}

    @torch.enable_grad()
    def evaluate_episode(
        self, episode: Episode, steps: int | None = None
    ) -> dict[str, np.ndarray]:
        """Adapt to one episode and return query predictions and probabilities.

        The model is placed in eval mode, so dropout is off during both
        adaptation and prediction.
        """
        self.model.eval()
        cfg = InnerConfig(
            steps=self.inner.steps if steps is None else steps,
            lr=self.inner.lr,
            first_order=True,  # no meta-gradient is needed at evaluation time
            learn_lr=self.inner.learn_lr,
        )
        sx, sy, qx, qy = self._to_torch(episode)
        params = {n: p.detach() for n, p in self.model.named_parameters()}
        for p in params.values():
            p.requires_grad_(True)
        lrs = self._current_lrs()
        if lrs is not None:
            lrs = {k: v.detach() for k, v in lrs.items()}

        adapted = adapt(self.model, params, sx, sy, cfg, lrs, create_graph=False)
        with torch.no_grad():
            logits = forward_with(self.model, adapted, qx)
            probs = logits.softmax(-1)
        return {
            "y_true": episode.query_y,
            "y_pred": logits.argmax(-1).cpu().numpy(),
            "probs": probs.cpu().numpy(),
            "loss": float(F.cross_entropy(logits, qy).item()),
        }

    def state_dict(self) -> dict:
        state = {"model": self.model.state_dict()}
        if self.inner.learn_lr:
            state["log_lrs"] = self._log_lrs.state_dict()
        return state

    def load_state_dict(self, state: dict) -> None:
        self.model.load_state_dict(state["model"])
        if self.inner.learn_lr and "log_lrs" in state:
            self._log_lrs.load_state_dict(state["log_lrs"])


class Reptile(MAML):
    """First-order meta-learner that moves the initialisation toward adapted weights.

    Reptile needs no query set during meta-training: it adapts on the support
    set and takes the difference as the meta-direction. The query set is still
    scored so that training curves remain comparable with MAML's.
    """

    name = "reptile"

    def __init__(self, *args, outer_step_size: float = 1.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.outer_step_size = outer_step_size
        # Reptile's outer direction is a parameter difference rather than a
        # meta-gradient, so it uses its own sequential step.
        self.vectorized = False

    def meta_step(self, episodes: list[Episode]) -> dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        params = clone_params(self.model)
        names = list(params)
        accumulated = {n: torch.zeros_like(params[n]) for n in names}
        total_loss = 0.0
        total_acc = 0.0

        for episode in episodes:
            sx, sy, qx, qy = self._to_torch(episode)
            detached = {n: params[n].detach().clone().requires_grad_(True) for n in names}
            adapted = adapt(
                self.model, detached, sx, sy, self.inner, create_graph=False
            )
            for n in names:
                # Meta-gradient surrogate: (initialisation - adapted), so an
                # SGD-style step moves the initialisation toward the adapted
                # point. Using the optimizer keeps weight decay and clipping
                # consistent with the MAML path.
                accumulated[n] += (params[n].detach() - adapted[n].detach()) / len(episodes)
            with torch.no_grad():
                logits = forward_with(self.model, adapted, qx)
                total_loss += F.cross_entropy(logits, qy).item() / len(episodes)
                total_acc += (logits.argmax(1) == qy).float().mean().item() / len(episodes)

        for n in names:
            params[n].grad = accumulated[n] * self.outer_step_size
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        return {"loss": total_loss, "accuracy": total_acc}


class ProtoNet:
    """Prototypical networks: the metric-based few-shot reference.

    No inner loop; a class prototype is the mean support embedding and
    classification is by negative squared Euclidean distance. The encoder is
    reused from ``models`` with its head removed, so the comparison against
    MAML holds the representation capacity fixed.
    """

    name = "protonet"

    def __init__(
        self,
        model: nn.Module,
        meta_lr: float = 1e-3,
        weight_decay: float = 1e-5,
        grad_clip: float | None = 1.0,
        device: torch.device | str = "cpu",
        **_ignored,
    ) -> None:
        self.model = model.to(device)
        self.device = torch.device(device)
        self.grad_clip = grad_clip
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=meta_lr, weight_decay=weight_decay
        )
        self.inner = InnerConfig(steps=0, lr=0.0)

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        """Penultimate representation: the classifier head is bypassed."""
        m = self.model
        if isinstance(getattr(m, "head", None), nn.Linear):
            head = m.head
            m.head = nn.Identity()
            try:
                return m(x)
            finally:
                m.head = head
        return m(x)

    def _logits(
        self, sx: torch.Tensor, sy: torch.Tensor, qx: torch.Tensor, n_way: int
    ) -> torch.Tensor:
        z_support = self._embed(sx)
        z_query = self._embed(qx)
        prototypes = torch.stack(
            [z_support[sy == c].mean(0) for c in range(n_way)]
        )
        return -torch.cdist(z_query, prototypes).pow(2)

    def _to_torch(self, episode: Episode):
        return MAML._to_torch(self, episode)  # type: ignore[arg-type]

    def meta_step(self, episodes: list[Episode]) -> dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_acc = 0.0
        for episode in episodes:
            sx, sy, qx, qy = self._to_torch(episode)
            logits = self._logits(sx, sy, qx, episode.n_way)
            loss = F.cross_entropy(logits, qy) / len(episodes)
            loss.backward()
            total_loss += loss.item()
            total_acc += (logits.argmax(1) == qy).float().mean().item() / len(episodes)
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        return {"loss": total_loss, "accuracy": total_acc}

    @torch.no_grad()
    def evaluate_episode(self, episode: Episode, steps: int | None = None) -> dict:
        self.model.eval()
        sx, sy, qx, qy = self._to_torch(episode)
        logits = self._logits(sx, sy, qx, episode.n_way)
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


def build_meta_learner(
    algorithm: str, model: nn.Module, inner: InnerConfig, **kwargs
):
    """Construct a meta-learner by name."""
    algorithm = algorithm.lower()
    if algorithm in ("maml", "fomaml"):
        inner.first_order = algorithm == "fomaml"
        return MAML(model, inner, **kwargs)
    if algorithm == "reptile":
        inner.first_order = True
        return Reptile(model, inner, **kwargs)
    if algorithm == "protonet":
        return ProtoNet(model, **kwargs)
    raise ValueError(
        f"unknown algorithm '{algorithm}'; "
        "choose from maml, fomaml, reptile, protonet"
    )
