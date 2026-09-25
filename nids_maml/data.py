"""CIC-IDS2017 loading, label consolidation and instance-disjoint splitting.

The central correctness requirement of this module: the meta-train, meta-validation
and meta-test pools must not share a single flow record, and every fitted
preprocessing statistic (scaler means/variances, quantile clip bounds) must be
estimated on the meta-train pool alone.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label consolidation
# ---------------------------------------------------------------------------

# The fourteen raw CIC-IDS2017 labels mapped onto the episodic class inventory.
# Keys are upper-cased and whitespace-normalised before lookup, because the
# published CSVs contain inconsistent spacing and a non-ASCII hyphen in the
# web-attack labels.
LABEL_MAP: dict[str, str] = {
    "BENIGN": "Benign",
    # DoS / DDoS family
    "DOS HULK": "DoS/DDoS",
    "DOS GOLDENEYE": "DoS/DDoS",
    "DOS SLOWLORIS": "DoS/DDoS",
    "DOS SLOWHTTPTEST": "DoS/DDoS",
    "DDOS": "DoS/DDoS",
    # Reconnaissance
    "PORTSCAN": "Port Scan",
    # Credential brute forcing
    "FTP-PATATOR": "Brute Force",
    "SSH-PATATOR": "Brute Force",
    # Application-layer
    "WEB ATTACK - BRUTE FORCE": "Web Attack",
    "WEB ATTACK - XSS": "Web Attack",
    "WEB ATTACK - SQL INJECTION": "Web Attack",
    # Held out / excluded (see EXCLUDED_CLASSES)
    "BOT": "Bot",
    "INFILTRATION": "Infiltration",
    "HEARTBLEED": "Heartbleed",
}

# Classes carrying too few flows to supply disjoint support and query sets
# across episodes. Retained in the dataframe but never used to build episodes.
EXCLUDED_CLASSES = ("Heartbleed", "Infiltration")

# Families withheld from meta-training entirely, reserved for the
# leave-one-family-out protocol.
DEFAULT_HELD_OUT_FAMILIES = ("Bot",)

# The five classes used by the primary 5-way protocol.
PRIMARY_CLASSES = ("Benign", "DoS/DDoS", "Port Scan", "Brute Force", "Web Attack")

# Columns that leak host identity or capture-session artefacts rather than
# describing traffic behaviour. Dropping them prevents the model from keying on
# the testbed's addressing scheme.
IDENTIFIER_COLUMNS = (
    "Flow ID",
    "Source IP",
    "Src IP",
    "Source Port",
    "Src Port",
    "Destination IP",
    "Dst IP",
    "Timestamp",
    "SimillarHTTP",
    "Unnamed: 0",
)


def normalise_label(raw: object) -> str:
    """Canonicalise a raw label string for lookup in LABEL_MAP."""
    text = str(raw).strip().upper()
    # The published CSVs use several dash characters in the web-attack labels
    # (U+2013 EN DASH appears in the Thursday capture).
    for dash in ("–", "—", "−"):
        text = text.replace(dash, "-")
    text = " ".join(text.split())
    # Normalise "WEB ATTACK BRUTE FORCE" / "WEB ATTACK - BRUTE FORCE" variants.
    text = text.replace("WEB ATTACK ", "WEB ATTACK - ").replace("- - ", "- ")
    return text


@dataclass
class SplitSpec:
    """Fractions of *flows* (not episodes) assigned to each meta-split."""

    train: float = 0.70
    val: float = 0.15
    test: float = 0.15

    def __post_init__(self) -> None:
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"split fractions must sum to 1.0, got {total}")


@dataclass
class DatasetBundle:
    """Feature matrices and labels for the three meta-splits.

    ``X_*`` are float32 arrays scaled with statistics fitted on the training
    pool only. ``y_*`` hold integer class ids indexing into ``class_names``.
    """

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    class_names: list[str]
    feature_names: list[str]
    meta: dict

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]

    def counts(self, split: str) -> dict[str, int]:
        y = {"train": self.y_train, "val": self.y_val, "test": self.y_test}[split]
        return {
            name: int((y == i).sum()) for i, name in enumerate(self.class_names)
        }


def _read_csvs(path: Path) -> pd.DataFrame:
    """Read either a single consolidated CSV or a directory of daily captures."""
    if path.is_dir():
        files = sorted(path.glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"no CSV files found under {path}")
        log.info("reading %d CSV files from %s", len(files), path)
        frames = [pd.read_csv(f, low_memory=False) for f in files]
        for f, frame in zip(files, frames):
            frame.columns = [c.strip() for c in frame.columns]
            log.info("  %s: %d rows", f.name, len(frame))
        df = pd.concat(frames, ignore_index=True)
    else:
        log.info("reading %s", path)
        df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    return df


def _find_label_column(df: pd.DataFrame) -> str:
    for candidate in ("Label", "label", "Attack", "attack"):
        if candidate in df.columns:
            return candidate
    raise ValueError(
        f"no label column found; available columns: {list(df.columns)[:20]}"
    )


def load_cicids2017(
    path: str | Path,
    classes: tuple[str, ...] = PRIMARY_CLASSES,
    split: SplitSpec | None = None,
    seed: int = 0,
    max_per_class: int | None = 60_000,
    clip_quantile: float = 0.999,
    drop_constant: bool = True,
) -> DatasetBundle:
    """Load CIC-IDS2017 and return instance-disjoint meta-splits.

    Args:
        path: a consolidated CSV, or a directory containing the daily captures.
        classes: episodic class inventory to retain, in label-id order.
        split: flow-level split fractions.
        seed: controls the stratified shuffle.
        max_per_class: cap per class *before* splitting, to bound memory. The
            cap is applied by uniform subsampling without replacement, so the
            retained flows remain representative of the class.
        clip_quantile: upper quantile used to winsorise features. CIC-IDS2017
            contains infinities and values near the float32 ceiling in the
            Flow Bytes/s and Flow Packets/s columns; clipping at a train-fitted
            quantile keeps standardisation numerically stable.
        drop_constant: drop features with zero variance on the training pool.

    Returns:
        A :class:`DatasetBundle` whose splits share no flow record.
    """
    split = split or SplitSpec()
    path = Path(path)
    rng = np.random.default_rng(seed)

    df = _read_csvs(path)
    label_col = _find_label_column(df)
    raw_n = len(df)

    # --- label consolidation -------------------------------------------------
    canonical = df[label_col].map(normalise_label)
    unknown = sorted(set(canonical) - set(LABEL_MAP))
    if unknown:
        log.warning("dropping %d rows with unmapped labels: %s",
                    int(canonical.isin(unknown).sum()), unknown)
    df = df.assign(_class=canonical.map(LABEL_MAP))
    df = df[df["_class"].notna()]

    keep = [c for c in classes if c not in EXCLUDED_CLASSES]
    if len(keep) != len(classes):
        log.info("excluded low-count classes: %s",
                 [c for c in classes if c in EXCLUDED_CLASSES])
    df = df[df["_class"].isin(keep)]
    if df.empty:
        raise ValueError(f"no rows remain after filtering to classes {keep}")

    class_names = list(keep)
    class_to_id = {name: i for i, name in enumerate(class_names)}

    # --- feature selection ---------------------------------------------------
    drop_cols = [c for c in IDENTIFIER_COLUMNS if c in df.columns]
    drop_cols += [label_col, "_class"]
    # Any remaining non-numeric column is an identifier or a free-text field;
    # CIC-IDS2017 has no genuinely categorical behavioural feature once the
    # address columns are removed, so these are dropped rather than encoded.
    features = df.drop(columns=drop_cols, errors="ignore")
    non_numeric = features.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        log.info("dropping %d non-numeric columns: %s", len(non_numeric), non_numeric)
        features = features.drop(columns=non_numeric)

    X_all = features.to_numpy(dtype=np.float64, copy=True)
    y_all = df["_class"].map(class_to_id).to_numpy(dtype=np.int64)
    feature_names = features.columns.tolist()

    # Infinities arise in the rate columns when duration is zero.
    X_all[~np.isfinite(X_all)] = np.nan

    # --- per-class cap, then stratified instance-level split -----------------
    idx_train, idx_val, idx_test = [], [], []
    for cid in range(len(class_names)):
        idx = np.flatnonzero(y_all == cid)
        rng.shuffle(idx)
        if max_per_class is not None and len(idx) > max_per_class:
            idx = idx[:max_per_class]
        n = len(idx)
        n_tr = int(round(n * split.train))
        n_va = int(round(n * split.val))
        idx_train.append(idx[:n_tr])
        idx_val.append(idx[n_tr:n_tr + n_va])
        idx_test.append(idx[n_tr + n_va:])

    idx_train = np.concatenate(idx_train)
    idx_val = np.concatenate(idx_val)
    idx_test = np.concatenate(idx_test)

    # Disjointness is guaranteed by construction above; assert it anyway, since
    # a silent violation here invalidates every downstream number.
    _assert_disjoint(idx_train, idx_val, idx_test)

    Xtr_raw, Xva_raw, Xte_raw = X_all[idx_train], X_all[idx_val], X_all[idx_test]
    ytr, yva, yte = y_all[idx_train], y_all[idx_val], y_all[idx_test]

    # --- preprocessing fitted on the TRAIN POOL ONLY -------------------------
    medians = np.nanmedian(Xtr_raw, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)

    def impute(X: np.ndarray) -> np.ndarray:
        out = X.copy()
        bad = ~np.isfinite(out)
        if bad.any():
            out[bad] = np.take(medians, np.nonzero(bad)[1])
        return out

    Xtr_i = impute(Xtr_raw)
    lo = np.quantile(Xtr_i, 1.0 - clip_quantile, axis=0)
    hi = np.quantile(Xtr_i, clip_quantile, axis=0)

    def prepare(X: np.ndarray) -> np.ndarray:
        return np.clip(impute(X), lo, hi)

    Xtr_c, Xva_c, Xte_c = prepare(Xtr_raw), prepare(Xva_raw), prepare(Xte_raw)

    mean = Xtr_c.mean(axis=0)
    std = Xtr_c.std(axis=0)

    if drop_constant:
        keep_feat = std > 1e-8
        if not keep_feat.all():
            dropped = [f for f, k in zip(feature_names, keep_feat) if not k]
            log.info("dropping %d zero-variance features: %s",
                     len(dropped), dropped[:10])
            Xtr_c, Xva_c, Xte_c = Xtr_c[:, keep_feat], Xva_c[:, keep_feat], Xte_c[:, keep_feat]
            mean, std = mean[keep_feat], std[keep_feat]
            feature_names = [f for f, k in zip(feature_names, keep_feat) if k]

    std = np.where(std > 1e-8, std, 1.0)
    scale = lambda X: ((X - mean) / std).astype(np.float32)

    meta = {
        "source": str(path),
        "rows_read": int(raw_n),
        "rows_retained": int(len(idx_train) + len(idx_val) + len(idx_test)),
        "seed": seed,
        "split": asdict(split),
        "max_per_class": max_per_class,
        "clip_quantile": clip_quantile,
        "n_features": len(feature_names),
        "class_names": class_names,
        "counts": {
            "train": {class_names[i]: int((ytr == i).sum()) for i in range(len(class_names))},
            "val": {class_names[i]: int((yva == i).sum()) for i in range(len(class_names))},
            "test": {class_names[i]: int((yte == i).sum()) for i in range(len(class_names))},
        },
    }

    bundle = DatasetBundle(
        X_train=scale(Xtr_c), y_train=ytr,
        X_val=scale(Xva_c), y_val=yva,
        X_test=scale(Xte_c), y_test=yte,
        class_names=class_names,
        feature_names=feature_names,
        meta=meta,
    )
    log.info("loaded %s: %d train / %d val / %d test flows, %d features",
             path.name, len(ytr), len(yva), len(yte), bundle.n_features)
    return bundle


def _assert_disjoint(*index_arrays: np.ndarray) -> None:
    """Raise if any two index arrays share an element."""
    seen: set[int] = set()
    for arr in index_arrays:
        as_set = set(arr.tolist())
        if len(as_set) != len(arr):
            raise AssertionError("duplicate indices within a single split")
        overlap = seen & as_set
        if overlap:
            raise AssertionError(
                f"splits share {len(overlap)} flow indices; "
                "meta-splits must be instance-disjoint"
            )
        seen |= as_set


def fingerprint(bundle: DatasetBundle) -> str:
    """Stable hash of the split contents, recorded alongside results.

    Two runs that report the same fingerprint used byte-identical data, which
    is what makes a paired significance test between them valid.
    """
    h = hashlib.sha256()
    for arr in (bundle.X_train, bundle.y_train, bundle.X_val,
                bundle.y_val, bundle.X_test, bundle.y_test):
        h.update(np.ascontiguousarray(arr).tobytes())
    h.update(json.dumps(bundle.class_names).encode())
    return h.hexdigest()[:16]


def make_synthetic(
    n_classes: int = 5,
    n_features: int = 40,
    n_per_class: int = 2000,
    seed: int = 0,
    separation: float = 1.2,
) -> DatasetBundle:
    """Gaussian-mixture stand-in used by the test suite and smoke runs.

    Each class is an axis-aligned Gaussian with a distinct mean, so a correct
    few-shot learner should comfortably exceed chance. This exists so the
    training machinery can be exercised without the 1 GB download.
    """
    rng = np.random.default_rng(seed)
    centres = rng.normal(0.0, separation, size=(n_classes, n_features))
    X, y = [], []
    for cid in range(n_classes):
        X.append(rng.normal(centres[cid], 1.0, size=(n_per_class, n_features)))
        y.append(np.full(n_per_class, cid, dtype=np.int64))
    X = np.concatenate(X).astype(np.float32)
    y = np.concatenate(y)

    perm = rng.permutation(len(y))
    X, y = X[perm], y[perm]
    n_tr, n_va = int(0.7 * len(y)), int(0.15 * len(y))
    # Named after the real inventory so that Protocol B, which looks for a
    # "Benign" class, is exercised by smoke runs too.
    class_names = list(PRIMARY_CLASSES[:n_classes]) + [
        f"class_{i}" for i in range(len(PRIMARY_CLASSES), n_classes)
    ]
    return DatasetBundle(
        X_train=X[:n_tr], y_train=y[:n_tr],
        X_val=X[n_tr:n_tr + n_va], y_val=y[n_tr:n_tr + n_va],
        X_test=X[n_tr + n_va:], y_test=y[n_tr + n_va:],
        class_names=class_names,
        feature_names=[f"f{i}" for i in range(n_features)],
        meta={"source": "synthetic", "seed": seed, "class_names": class_names},
    )
