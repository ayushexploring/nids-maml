"""Regression tests for the defects found in the original implementation.

Each test below corresponds to a specific defect in the TensorFlow notebook
that this package replaces. They are written as executable assertions so the
defects cannot reappear silently.

Run with:  python -m tests.test_correctness
"""

from __future__ import annotations

import sys

import numpy as np
import torch
import torch.nn.functional as F

from nids_maml.data import make_synthetic, _assert_disjoint, normalise_label
from nids_maml.episodes import BinaryEpisodeSampler, EpisodeSampler
from nids_maml.meta import InnerConfig, MAML, adapt, clone_params, forward_with
from nids_maml.metrics import mean_ci, paired_test, wilson_interval
from nids_maml.models import FTTransformer

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str):
    def decorator(fn):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - this is a test harness
            FAILED.append(f"{name}: {type(exc).__name__}: {exc}")
            print(f"FAIL  {name}\n      {type(exc).__name__}: {exc}")
        else:
            PASSED.append(name)
            print(f"ok    {name}")
        return fn
    return decorator


# ---------------------------------------------------------------------------
# Defect 1: attention ran over a length-1 sequence, collapsing every encoder
# block to a linear map.
# ---------------------------------------------------------------------------

@check("attention operates over one token per feature, not a single token")
def test_token_axis():
    n_features, d_model = 12, 32
    model = FTTransformer(n_features, n_classes=5, d_model=d_model, n_heads=4, n_blocks=2)
    x = torch.randn(7, n_features)

    tokens = model.tokenizer(x)
    assert tokens.shape == (7, n_features + 1, d_model), (
        f"expected {n_features + 1} tokens (features + CLS), got {tokens.shape[1]}"
    )

    maps = model.attention_maps(x)
    assert len(maps) == 2, f"expected one attention map per block, got {len(maps)}"
    for m in maps:
        assert m.shape == (7, n_features + 1, n_features + 1), (
            f"attention matrix should be square over tokens, got {tuple(m.shape)}"
        )
        # A length-1 sequence would make every attention row trivially 1.0.
        # Genuine feature attention distributes mass across several tokens.
        assert m.shape[-1] > 1, "attention collapsed to a single token"
        assert torch.allclose(m.sum(-1), torch.ones_like(m.sum(-1)), atol=1e-4)


@check("permuting feature order changes the model output")
def test_permutation_sensitivity():
    # If every column shared one projection and no slot identity, the encoder
    # would be permutation-invariant and column order would carry no meaning.
    torch.manual_seed(0)
    model = FTTransformer(10, n_classes=5, d_model=32, n_heads=4, n_blocks=2).eval()
    x = torch.randn(4, 10)
    perm = torch.randperm(10)
    with torch.no_grad():
        a = model(x)
        b = model(x[:, perm])
    assert not torch.allclose(a, b, atol=1e-5), (
        "output is invariant to feature permutation; per-column identity is lost"
    )


# ---------------------------------------------------------------------------
# Defect 2: the weight assignment sat outside the inner-step loop, so N steps
# applied exactly one update.
# ---------------------------------------------------------------------------

@check("N inner steps apply N distinct updates")
def test_inner_steps_are_distinct():
    torch.manual_seed(0)
    # Dropout off and eval mode: this test is about the structure of the inner
    # loop, so the support loss must be a deterministic function of the
    # parameters. With dropout active each measurement draws a fresh mask and
    # the comparison would be between noise realisations.
    model = FTTransformer(8, n_classes=3, d_model=16, n_heads=2, n_blocks=1,
                          dropout=0.0).eval()
    params = clone_params(model)
    sx = torch.randn(9, 8)
    sy = torch.tensor([0, 1, 2] * 3)

    # lr is kept at the operating point of 0.01. At lr >= 0.05 this inner loop
    # collapses to the uniform-prediction fixed point (support loss plateaus at
    # ln(n_way)), which is a property of the optimiser rather than of the loop
    # structure this test is checking; it is measured as ablation F3 instead.
    one = adapt(model, params, sx, sy, InnerConfig(steps=1, lr=0.01))
    five = adapt(model, params, sx, sy, InnerConfig(steps=5, lr=0.01))

    ref = list(params)[0]
    assert not torch.allclose(one[ref], five[ref]), (
        "one-step and five-step adaptation produced identical parameters; "
        "the update is being applied outside the step loop"
    )

    # Support loss must fall monotonically enough that five steps beat one.
    def support_loss(p):
        with torch.no_grad():
            return F.cross_entropy(forward_with(model, p, sx), sy).item()

    l0, l1, l5 = support_loss(params), support_loss(one), support_loss(five)
    assert l5 < l1 < l0, f"support loss did not decrease with steps: {l0=} {l1=} {l5=}"


@check("adaptation trajectory has one entry per step, including step 0")
def test_trajectory_length():
    model = FTTransformer(8, n_classes=3, d_model=16, n_heads=2, n_blocks=1)
    params = clone_params(model)
    sx, sy = torch.randn(9, 8), torch.tensor([0, 1, 2] * 3)
    traj = adapt(model, params, sx, sy, InnerConfig(steps=4, lr=0.01),
                 return_trajectory=True)
    assert len(traj) == 5, f"expected 5 entries for 4 steps + initial, got {len(traj)}"


# ---------------------------------------------------------------------------
# Defect 3: set_weights severed the autograd graph, so second-order MAML was
# unavailable and the meta-gradient was first-order by accident.
# ---------------------------------------------------------------------------

@check("second-order meta-gradient differs from the first-order one")
def test_second_order_is_real():
    torch.manual_seed(0)
    n_features = 8

    def meta_grad(first_order: bool) -> torch.Tensor:
        torch.manual_seed(0)
        model = FTTransformer(n_features, 3, d_model=16, n_heads=2, n_blocks=1)
        params = clone_params(model)
        sx, sy = torch.randn(9, n_features), torch.tensor([0, 1, 2] * 3)
        qx, qy = torch.randn(9, n_features), torch.tensor([0, 1, 2] * 3)
        cfg = InnerConfig(steps=3, lr=0.05, first_order=first_order)
        adapted = adapt(model, params, sx, sy, cfg)
        loss = F.cross_entropy(forward_with(model, adapted, qx), qy)
        grads = torch.autograd.grad(loss, list(model.parameters()), allow_unused=True)
        return torch.cat([g.flatten() for g in grads if g is not None])

    g_first = meta_grad(True)
    g_second = meta_grad(False)
    assert g_first.shape == g_second.shape
    assert not torch.allclose(g_first, g_second, atol=1e-7), (
        "first- and second-order meta-gradients are identical; the inner loop "
        "is not differentiable and create_graph has no effect"
    )


# ---------------------------------------------------------------------------
# Defect 4: dropout was hard-wired active, so every evaluation was stochastic.
# ---------------------------------------------------------------------------

@check("eval() makes predictions deterministic")
def test_eval_disables_dropout():
    torch.manual_seed(0)
    model = FTTransformer(10, 5, d_model=32, n_heads=4, n_blocks=2, dropout=0.5)
    x = torch.randn(16, 10)

    model.eval()
    with torch.no_grad():
        a, b = model(x), model(x)
    assert torch.allclose(a, b), "eval mode still stochastic; dropout is not disabled"

    model.train()
    with torch.no_grad():
        c, d = model(x), model(x)
    assert not torch.allclose(c, d), "train mode is deterministic; dropout is inactive"


@check("episodic evaluation is reproducible across repeated calls")
def test_evaluation_reproducible():
    torch.manual_seed(0)
    bundle = make_synthetic(n_classes=5, n_features=12, n_per_class=200, seed=1)
    sampler = EpisodeSampler(bundle.X_test, bundle.y_test, 5, 5, 10, seed=3)
    episode = sampler.sample()
    model = FTTransformer(12, 5, d_model=32, n_heads=4, n_blocks=1, dropout=0.3)
    learner = MAML(model, InnerConfig(steps=3, lr=0.01))
    first = learner.evaluate_episode(episode)
    second = learner.evaluate_episode(episode)
    assert np.array_equal(first["y_pred"], second["y_pred"]), (
        "repeated evaluation of the same episode gave different predictions"
    )


# ---------------------------------------------------------------------------
# Defect 5: episodes were drawn from the full dataset and only then split, so
# meta-train and meta-test shared flow records.
# ---------------------------------------------------------------------------

@check("meta-splits are instance-disjoint")
def test_splits_disjoint():
    bundle = make_synthetic(n_classes=5, n_features=10, n_per_class=300, seed=0)
    n = len(bundle.y_train) + len(bundle.y_val) + len(bundle.y_test)
    rows = set()
    for X in (bundle.X_train, bundle.X_val, bundle.X_test):
        for row in X:
            rows.add(row.tobytes())
    assert len(rows) == n, (
        f"{n - len(rows)} identical rows appear in more than one meta-split"
    )
    _assert_disjoint(np.array([0, 1, 2]), np.array([3, 4]), np.array([5]))
    try:
        _assert_disjoint(np.array([0, 1]), np.array([1, 2]))
    except AssertionError:
        pass
    else:
        raise AssertionError("_assert_disjoint failed to detect an overlap")


@check("support and query sets within an episode never share an instance")
def test_support_query_disjoint():
    bundle = make_synthetic(n_classes=5, n_features=10, n_per_class=300, seed=0)
    sampler = EpisodeSampler(bundle.X_train, bundle.y_train, 5, 5, 15, seed=0)
    for _ in range(20):
        ep = sampler.sample()
        support = {row.tobytes() for row in ep.support_x}
        query = {row.tobytes() for row in ep.query_x}
        assert not (support & query), "an instance appears in both support and query"


@check("fixed_set is reproducible and does not disturb the sampler stream")
def test_fixed_set_reproducible():
    bundle = make_synthetic(n_classes=5, n_features=10, n_per_class=300, seed=0)
    s1 = EpisodeSampler(bundle.X_test, bundle.y_test, 5, 5, 10, seed=0)
    s2 = EpisodeSampler(bundle.X_test, bundle.y_test, 5, 5, 10, seed=99)
    a = s1.fixed_set(5, seed=7)
    b = s2.fixed_set(5, seed=7)
    for ea, eb in zip(a, b):
        assert np.array_equal(ea.query_x, eb.query_x), (
            "same seed produced different evaluation episodes; paired tests "
            "across methods would be invalid"
        )
    # The main stream must be unaffected by the excursion.
    s3 = EpisodeSampler(bundle.X_test, bundle.y_test, 5, 5, 10, seed=0)
    s3.fixed_set(5, seed=7)
    assert np.array_equal(s1.sample().query_x, s3.sample().query_x)


# ---------------------------------------------------------------------------
# Defect 6: binary (Protocol B) tasks were evaluated with a 5-way head, so
# predictions fell outside {0, 1} and confusion matrices came out 3x3 or 4x4.
# ---------------------------------------------------------------------------

@check("binary episodes yield predictions confined to {0, 1}")
def test_binary_head_is_binary():
    bundle = make_synthetic(n_classes=5, n_features=10, n_per_class=300, seed=0)
    sampler = BinaryEpisodeSampler(
        bundle.X_test, bundle.y_test, benign_class=0, attack_class=2,
        k_shot=5, n_query=15, seed=0,
    )
    model = FTTransformer(10, n_classes=2, d_model=32, n_heads=4, n_blocks=1)
    learner = MAML(model, InnerConfig(steps=3, lr=0.01))
    for episode in sampler.fixed_set(10, seed=0):
        assert set(np.unique(episode.support_y)) <= {0, 1}
        out = learner.evaluate_episode(episode)
        assert set(np.unique(out["y_pred"])) <= {0, 1}, (
            f"binary task produced labels {np.unique(out['y_pred'])}"
        )
        assert episode.classes[0] == 0, "label 0 must always denote benign"


# ---------------------------------------------------------------------------
# Supporting machinery
# ---------------------------------------------------------------------------

@check("degenerate class pools are detected and reported")
def test_degeneracy_flag():
    bundle = make_synthetic(n_classes=5, n_features=10, n_per_class=300, seed=0)
    five_way = EpisodeSampler(bundle.X_train, bundle.y_train, 5, 5, 10, seed=0)
    three_way = EpisodeSampler(bundle.X_train, bundle.y_train, 3, 5, 10, seed=0)
    assert five_way.is_degenerate, "5-way over 5 classes should be flagged degenerate"
    assert not three_way.is_degenerate


@check("Wilson interval on a perfect score does not claim certainty")
def test_wilson_perfect_score():
    interval = wilson_interval(30, 30)
    assert interval.mean == 1.0
    assert interval.lo < 0.9, (
        f"a perfect score on 30 trials should have a lower bound below 0.9, "
        f"got {interval.lo:.4f}"
    )


@check("confidence intervals widen as the episode count falls")
def test_ci_width_scales():
    rng = np.random.default_rng(0)
    values = rng.normal(0.75, 0.1, size=600)
    wide = mean_ci(values[:30])
    narrow = mean_ci(values)
    assert (narrow.hi - narrow.lo) < (wide.hi - wide.lo)


@check("paired test rejects mismatched episode counts")
def test_paired_test_guards():
    try:
        paired_test(np.zeros(10), np.zeros(9))
    except ValueError:
        pass
    else:
        raise AssertionError("paired test accepted unequal-length inputs")


@check("CIC-IDS2017 label variants normalise to a single canonical form")
def test_label_normalisation():
    variants = [
        "Web Attack – Brute Force",
        "Web Attack - Brute Force",
        "  WEB ATTACK - BRUTE FORCE  ",
    ]
    canonical = {normalise_label(v) for v in variants}
    assert len(canonical) == 1, f"label variants did not unify: {canonical}"
    assert normalise_label("BENIGN") == "BENIGN"
    assert normalise_label("DoS Hulk") == "DOS HULK"


# ---------------------------------------------------------------------------
# The vectorised meta-step is an optimisation, not a different algorithm.
# ---------------------------------------------------------------------------

@check("vectorised and sequential meta-steps produce the same update")
def test_vectorized_matches_sequential():
    bundle = make_synthetic(n_classes=5, n_features=16, n_per_class=200, seed=0)
    sampler = EpisodeSampler(bundle.X_train, bundle.y_train, 5, 5, 10, seed=0)
    episodes = sampler.fixed_set(4, seed=0)

    def meta_gradient(vectorized: bool) -> torch.Tensor:
        torch.manual_seed(0)
        model = FTTransformer(16, 5, d_model=32, n_heads=4, n_blocks=2, dropout=0.0)
        learner = MAML(model, InnerConfig(steps=3, lr=0.01, first_order=True),
                       meta_lr=0.01, vectorized=vectorized, grad_clip=None)
        learner.model.eval()   # hold dropout out of the comparison

        # Compare the meta-gradient rather than the post-update parameters.
        # Adam's first step is close to lr * sign(g), so it amplifies
        # float32 reduction-order differences in near-zero gradient components
        # into parameter differences of order lr -- which would say nothing
        # about whether the two paths compute the same thing.
        captured: dict[str, torch.Tensor] = {}

        def capture(*_args, **_kwargs):
            for name, param in learner.model.named_parameters():
                captured[name] = param.grad.detach().clone()

        learner.optimizer.step = capture
        learner.meta_step(episodes)
        return torch.cat([captured[k].flatten() for k in sorted(captured)])

    a, b = meta_gradient(True), meta_gradient(False)
    relative = ((a - b).norm() / b.norm()).item()
    assert relative < 1e-5, (
        f"vectorised and sequential meta-gradients diverge; "
        f"relative difference {relative:.2e}"
    )


@check("the inner loop adapts the attention projections, not just the head")
def test_attention_is_adapted():
    # The manuscript claims the whole encoder, including Q/K/V, is task-adapted.
    # This checks the claim against the actual parameter updates.
    torch.manual_seed(0)
    model = FTTransformer(16, 5, d_model=32, n_heads=4, n_blocks=2, dropout=0.0)
    params = clone_params(model)
    sx, sy = torch.randn(25, 16), torch.arange(5).repeat(5)
    adapted = adapt(model, params, sx, sy, InnerConfig(steps=5, lr=0.01))

    attention_moved = {
        name: (adapted[name] - params[name]).norm().item()
        for name in params
        if ".attn." in name and name.endswith("weight")
    }
    assert attention_moved, "no attention projection parameters found"
    for name, delta in attention_moved.items():
        assert delta > 0, f"{name} was not adapted by the inner loop"


@check("batched and per-episode evaluation agree")
def test_batched_evaluation_matches():
    torch.manual_seed(0)
    bundle = make_synthetic(n_classes=5, n_features=12, n_per_class=300, seed=0)
    sampler = EpisodeSampler(bundle.X_test, bundle.y_test, 5, 5, 10, seed=2)
    episodes = sampler.fixed_set(7, seed=5)
    model = FTTransformer(12, 5, d_model=32, n_heads=4, n_blocks=2, dropout=0.1)
    learner = MAML(model, InnerConfig(steps=4, lr=0.1))

    batched = learner.evaluate_batch(episodes, chunk=3)   # chunk < len, so the
    looped = [learner.evaluate_episode(ep) for ep in episodes]  # chunking path runs
    assert len(batched) == len(looped) == len(episodes)
    for b, l in zip(batched, looped):
        assert np.array_equal(b["y_pred"], l["y_pred"]), (
            "batched evaluation disagrees with the per-episode path"
        )
        assert abs(b["loss"] - l["loss"]) < 1e-4


@check("pre-consolidated label spellings map to the same five classes")
def test_preconsolidated_labels():
    # Redistributed copies of CIC-IDS2017 often ship with the fourteen raw
    # labels already merged, under different spellings. Both forms must land on
    # the same class inventory, or flows are silently dropped at load time.
    from nids_maml.data import LABEL_MAP

    raw_form = ["BENIGN", "DDoS", "DoS Hulk", "PortScan",
                "FTP-Patator", "SSH-Patator", "Web Attack - XSS"]
    merged_form = ["BENIGN", "DDOS", "PORT SCANNING", "BRUTE FORCE", "WEB ATTACKS"]

    for label in raw_form + merged_form:
        key = normalise_label(label)
        assert key in LABEL_MAP, f"{label!r} normalises to {key!r}, absent from LABEL_MAP"

    assert {LABEL_MAP[normalise_label(l)] for l in merged_form} == {
        "Benign", "DoS/DDoS", "Port Scan", "Brute Force", "Web Attack"
    }
    # The two spellings of a family must agree.
    assert LABEL_MAP[normalise_label("PortScan")] ==            LABEL_MAP[normalise_label("PORT SCANNING")]
    assert LABEL_MAP[normalise_label("FTP-Patator")] ==            LABEL_MAP[normalise_label("BRUTE FORCE")]
    assert LABEL_MAP[normalise_label("Web Attack - XSS")] ==            LABEL_MAP[normalise_label("WEB ATTACKS")]


@check("deduplication collapses near-identical flows and keeps distinct ones")
def test_deduplication():
    from nids_maml.data import deduplicate_indices

    rng = np.random.default_rng(0)
    distinct = rng.normal(0, 5, size=(50, 8))
    # Each distinct row is repeated with perturbations far below the rounding
    # precision, exactly the regime CIC-IDS2017 attack bursts produce.
    noisy = np.repeat(distinct, 4, axis=0)
    noisy += rng.normal(0, 1e-5, size=noisy.shape)

    kept = deduplicate_indices(noisy, decimals=2)
    # 200 rows collapse to roughly 50. The grouping is a rounding grid, so a
    # few groups straddle a grid line and survive as two; the tolerance below
    # admits that while still failing if deduplication stopped working.
    assert 50 <= len(kept) <= 55, (
        f"expected about one representative per distinct flow, kept {len(kept)}"
    )
    # Genuinely separate points must all survive.
    assert len(deduplicate_indices(distinct, decimals=2)) == 50

    # The retained rows must be an index into the original array.
    assert kept.min() >= 0 and kept.max() < len(noisy)
    assert len(set(kept.tolist())) == len(kept)


@check("deduplication leaves the splits content-disjoint, not just record-disjoint")
def test_dedup_removes_near_twins():
    from nids_maml.data import deduplicate_indices

    rng = np.random.default_rng(1)
    base = rng.normal(0, 3, size=(40, 6))
    duplicated = np.repeat(base, 5, axis=0) + rng.normal(0, 1e-6, size=(200, 6))

    kept = duplicated[deduplicate_indices(duplicated, decimals=2)]
    assert len(kept) < 60, f"200 near-identical rows collapsed only to {len(kept)}"
    # The property that matters: the retained rows are no longer numerically
    # indistinguishable from one another, which is the condition that fails on
    # CIC-IDS2017 without this step.
    d = np.linalg.norm(kept[:, None, :] - kept[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    assert d.min() > 1e-4, (
        f"two retained flows are still {d.min():.2e} apart; "
        "deduplication did not separate them"
    )


if __name__ == "__main__":
    print()
    print(f"{len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        print("\nFailures:")
        for item in FAILED:
            print(f"  - {item}")
    sys.exit(1 if FAILED else 0)
