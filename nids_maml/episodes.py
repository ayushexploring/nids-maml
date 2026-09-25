"""Episodic task sampling for N-way, K-shot meta-learning.

Two properties distinguish this sampler from the one it replaces:

1. Episodes are drawn from a single meta-split's flow pool, so an episode can
   never mix training and test flows.
2. Episodes are drawn on demand rather than materialised once, so meta-training
   sees a fresh task on every step instead of cycling a fixed list. Evaluation
   uses a seeded sampler, which makes the episode stream reproducible and
   identical across methods -- the precondition for a paired significance test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Episode:
    """One N-way, K-shot task.

    Labels are *episode-local*: ``support_y`` and ``query_y`` index into
    ``classes``, not into the dataset's global class ids. ``classes`` records
    the global ids so that per-family results can be recovered afterwards.
    """

    support_x: np.ndarray  # (n_way * k_shot, n_features)
    support_y: np.ndarray  # (n_way * k_shot,)
    query_x: np.ndarray    # (n_way * n_query, n_features)
    query_y: np.ndarray    # (n_way * n_query,)
    classes: np.ndarray    # (n_way,) global class ids, in episode-label order

    @property
    def n_way(self) -> int:
        return len(self.classes)


class EpisodeSampler:
    """Samples N-way, K-shot episodes from one meta-split.

    Args:
        X: feature matrix for this split.
        y: global class ids for this split.
        n_way: classes per episode.
        k_shot: support examples per class.
        n_query: query examples per class.
        classes: restrict sampling to these global class ids. Defaults to every
            class present with enough examples.
        seed: when given, the sampler is deterministic and replayable.

    Support and query indices within an episode are always disjoint: they are
    taken from a single draw of ``k_shot + n_query`` distinct rows per class.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_way: int = 5,
        k_shot: int = 5,
        n_query: int = 15,
        classes: list[int] | None = None,
        seed: int | None = None,
    ) -> None:
        self.X = X
        self.y = y
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.rng = np.random.default_rng(seed)

        needed = k_shot + n_query
        self._pool: dict[int, np.ndarray] = {}
        available = classes if classes is not None else sorted(set(y.tolist()))
        for cid in available:
            idx = np.flatnonzero(y == cid)
            if len(idx) >= needed:
                self._pool[int(cid)] = idx
        self.classes = sorted(self._pool)

        if len(self.classes) < n_way:
            present = {int(c): int((y == c).sum()) for c in sorted(set(y.tolist()))}
            raise ValueError(
                f"{n_way}-way episodes require {n_way} classes with at least "
                f"{needed} examples each ({k_shot} support + {n_query} query), "
                f"but only {len(self.classes)} qualify: {self.classes}.\n"
                f"Class id -> count in this split: {present}\n"
                "A class missing here that you expected usually means its label "
                "spelling is absent from LABEL_MAP in data.py, so its flows were "
                "dropped at load time -- check the 'unmapped labels' warning above."
            )

    @property
    def is_degenerate(self) -> bool:
        """True when the class pool exactly equals the episode width.

        In that case every episode spans the same class inventory and tasks
        differ only in which instances and which label permutation are drawn.
        The original 5-class / 5-way configuration sat in this regime; it is
        reported so the condition is visible in results rather than inferred.
        """
        return len(self.classes) == self.n_way

    def sample(self) -> Episode:
        chosen = self.rng.choice(self.classes, size=self.n_way, replace=False)
        sx, sy, qx, qy = [], [], [], []
        for local_label, cid in enumerate(chosen):
            idx = self._pool[int(cid)]
            picked = self.rng.choice(idx, size=self.k_shot + self.n_query, replace=False)
            sx.append(self.X[picked[:self.k_shot]])
            qx.append(self.X[picked[self.k_shot:]])
            sy.append(np.full(self.k_shot, local_label, dtype=np.int64))
            qy.append(np.full(self.n_query, local_label, dtype=np.int64))
        return Episode(
            support_x=np.concatenate(sx),
            support_y=np.concatenate(sy),
            query_x=np.concatenate(qx),
            query_y=np.concatenate(qy),
            classes=np.asarray(chosen, dtype=np.int64),
        )

    def batch(self, size: int) -> list[Episode]:
        return [self.sample() for _ in range(size)]

    def fixed_set(self, n_episodes: int, seed: int) -> list[Episode]:
        """Materialise a reproducible episode list without disturbing ``self.rng``.

        Every method under comparison is evaluated on the episode list produced
        by this call with the same seed, so differences between methods are not
        confounded with differences in the tasks they were shown.
        """
        saved = self.rng
        self.rng = np.random.default_rng(seed)
        try:
            return [self.sample() for _ in range(n_episodes)]
        finally:
            self.rng = saved


class BinaryEpisodeSampler(EpisodeSampler):
    """Benign-vs-one-family binary episodes (the manuscript's Protocol B).

    Subclassing keeps the support/query disjointness logic in one place. The
    important difference from the original implementation is that these
    episodes are 2-way, so a 2-logit head is used and predictions cannot fall
    outside {0, 1} -- the defect that produced 3x3 and 4x4 confusion matrices.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        benign_class: int,
        attack_class: int,
        k_shot: int = 5,
        n_query: int = 15,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            X, y, n_way=2, k_shot=k_shot, n_query=n_query,
            classes=[benign_class, attack_class], seed=seed,
        )
        self.benign_class = benign_class
        self.attack_class = attack_class

    def sample(self) -> Episode:
        """Draw a 2-way episode with a fixed label convention.

        Local label 0 is always benign and 1 is always the attack family, so
        precision and recall are computed against a stable positive class.
        """
        sx, sy, qx, qy = [], [], [], []
        for local_label, cid in enumerate((self.benign_class, self.attack_class)):
            idx = self._pool[int(cid)]
            picked = self.rng.choice(idx, size=self.k_shot + self.n_query, replace=False)
            sx.append(self.X[picked[:self.k_shot]])
            qx.append(self.X[picked[self.k_shot:]])
            sy.append(np.full(self.k_shot, local_label, dtype=np.int64))
            qy.append(np.full(self.n_query, local_label, dtype=np.int64))
        return Episode(
            support_x=np.concatenate(sx),
            support_y=np.concatenate(sy),
            query_x=np.concatenate(qx),
            query_y=np.concatenate(qy),
            classes=np.asarray([self.benign_class, self.attack_class], dtype=np.int64),
        )
