# Few-shot network intrusion detection with meta-learned Transformer encoders

A reimplementation of the Transformer + MAML intrusion-detection pipeline, built
to support a journal submission. It replaces an earlier TensorFlow notebook
whose reported results were affected by several defects (documented below).

```
nids_maml/
  data.py        CIC-IDS2017 loading, label consolidation, instance-disjoint splits
  episodes.py    N-way K-shot samplers, seeded and reproducible
  models.py      feature-tokenizing Transformer, MLP and CNN base learners
  meta.py        MAML (first and second order), Reptile, Prototypical Networks
  baselines.py   supervised pre-train + fine-tune, and classical classifiers
  train.py       meta-training loop, early stopping, episodic evaluation
  metrics.py     confidence intervals, Wilson bounds, paired tests, ECE
  analyse.py     result files -> tables
  figures.py     result files -> 300 dpi figures
  run.py         CLI entry point
configs/         primary.yaml, smoke.yaml
scripts/         experiments.py -- the full run matrix
tests/           regression tests, one per defect found in the original code
colab_driver.ipynb
```

## Quick start

```bash
pip install -r requirements.txt
python -m tests.test_correctness                                   # must pass
python -m nids_maml.run --config configs/smoke.yaml                # ~1 min, synthetic
python -m nids_maml.run --config configs/primary.yaml --data-path /path/to/cicids2017.csv
python scripts/experiments.py --list                               # inspect the matrix
python scripts/experiments.py --run --only baselines --data-path /path/to/data.csv
python -m nids_maml.analyse --results results --out paper_assets --figures
```

On Colab, open `colab_driver.ipynb` and run it top to bottom; it clones this
repo, runs the tests, and writes results to Drive so a dropped session resumes
by re-running one cell.

Each run writes one JSON file containing the resolved config, a data
fingerprint, the training history, and meta-test results with confidence
intervals. `analyse.py` reads only those files, so every number in the
manuscript is traceable to a recorded run.

## What was wrong with the previous implementation

Each item was verified against the original notebook and is now covered by a
test in `tests/test_correctness.py`.

**1. Attention ran over a sequence of length 1.** The model projected the whole
feature vector to one embedding and added a length-1 sequence axis before the
encoder blocks. Self-attention over a single token is the identity after
softmax, so every block collapsed to a linear map and the positional encoding
became a constant bias. The central claim — that the inner loop adapts the
attention pattern — had nothing to adapt. *Fixed:* one token per feature, so
attention runs over `n_features + 1` tokens.

**2. Meta-train, meta-validation and meta-test shared flow records.** Episodes
were sampled from the entire dataset and the resulting 200 *episodes* were then
split 70/15/15. The same flow could appear in a training support set and a test
query set, and the scaler was fitted on all data before any split. *Fixed:*
flows are split first, episodes are drawn within a split, and preprocessing
statistics are fitted on the training pool alone.

**3. "Five inner steps" applied one update.** In the evaluation routine the
weight assignment sat outside the inner-step loop, so every iteration recomputed
the gradient at the unchanged initialisation and a single update was applied at
the end. Reported accuracies were one-step numbers, while the adaptation-curve
figure used a different code path that looped correctly. *Fixed:* one `adapt()`
function serves meta-training and evaluation.

**4. Dropout was never disabled.** Encoder blocks were constructed with
`training=True` baked in, which `model(x, training=False)` cannot override.
Every reported number was an average over stochastic forward passes, and some
of the meta-validation instability analysed at length in the manuscript was this
noise. *Fixed:* dropout follows `model.train()` / `model.eval()`.

**5. Binary tasks were scored with a five-way head.** The benign-versus-family
evaluation reused the 5-way classifier, so predictions fell outside `{0, 1}`;
this is the source of the `Unexpected confusion matrix shape: (3, 3)` warnings
and of one family scoring F1 = 0.0000 while the manuscript reported 0.9667–1.0
for the same families from a different code path. *Fixed:* 2-way episodes with a
2-logit head and a fixed positive-class convention.

**6. The meta-gradient was first-order by accident.** Inner-loop updates were
applied with `set_weights` on NumPy arrays, which severs the autograd graph, so
the second-order term was silently discarded and true MAML could not be run at
all. *Fixed:* a functional inner loop via `torch.func`, with `first_order` as an
explicit switch.

**7. Single-run point estimates.** Meta-test accuracy was reported from 30
episodes of one run with no interval. *Fixed:* every result is a mean over
episodes with a 95% confidence interval, across multiple seeds, with paired
significance tests on a shared episode stream.

## A finding that changes the configuration

The inner learning rate used previously, α = 0.01, is below the threshold at
which a Transformer base learner meta-learns at all. At α = 0.01 meta-validation
accuracy stays at chance indefinitely: the inner loop moves the parameters too
little for the meta-gradient to carry usable signal. At α = 0.1 the same model
reaches high validation accuracy within a few hundred meta-steps.

The earlier pipeline tolerated α = 0.01 only because its encoder had collapsed
to a linear stack (defect 1), which plain SGD adapts easily. A correctly
implemented Transformer does not. The default is therefore α = 0.1, and α is
carried as an explicit ablation factor including the 0.01 level.

Two related observations, both worth reporting rather than hiding: above roughly
α = 0.5 the inner loop diverges and then settles at the uniform-prediction fixed
point, where the support loss equals `ln(n_way)`; and with exactly five classes
under a 5-way protocol, class selection is degenerate — every episode spans the
same inventory, so no class is ever novel at meta-test time. The samplers flag
this condition (`EpisodeSampler.is_degenerate`) and it is recorded in every
result file, because it bounds what a claim about unseen attacks can say.

## Experiment matrix

`scripts/experiments.py` declares 141 runs: eight baselines plus an MLP-encoder
variant over five seeds, and ten ablation factors over three seeds. Time one run
before launching the matrix; the script prints an estimate and skips runs whose
result file already exists.

Baselines are run on identical splits and identical episode streams, which is
what makes the paired tests valid. `analyse.py` checks this and refuses to
compare runs with different data fingerprints.

## Reproducibility

Seeds control the data split, the model initialisation and the episode streams.
Evaluation episodes are derived from the seed through `EpisodeSampler.fixed_set`,
so two methods at the same seed see the identical episode list. The data
fingerprint recorded in each result file is a hash of the split contents; runs
that report the same fingerprint used byte-identical data.
