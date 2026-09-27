"""Fetch and verify the UNSW-NB15 partitioned CSVs.

    python scripts/get_unsw.py --out /content/drive/MyDrive/PhD/unsw

The dataset's own project page distributes through a file-sharing service that
does not serve stable direct links, so this pulls from public mirrors and then
verifies what arrived rather than trusting the source. A file that lacks the
``attack_cat`` column cannot support N-way episodes, and the binary-label-only
variant of this dataset is common enough that the check is worth making
explicit.

What is downloaded is the standard partitioned release: 175,341 training and
82,332 testing records over nine attack categories plus normal traffic. This
pipeline re-splits them itself, so the published partition is used only as a
convenient packaging of the same records.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import urllib.error
import urllib.request
from pathlib import Path

log = logging.getLogger("get_unsw")

# Tried in order; the first that yields a file with the expected columns wins.
MIRRORS: dict[str, list[str]] = {
    "UNSW_NB15_training-set.csv": [
        "https://huggingface.co/datasets/Mouwiya/UNSW-NB15/resolve/main/UNSW_NB15_training-set.csv",
        "https://raw.githubusercontent.com/Nir-J/ML-Projects/master/UNSW-Network_Packet_Classification/UNSW_NB15_training-set.csv",
    ],
    "UNSW_NB15_testing-set.csv": [
        "https://huggingface.co/datasets/Mouwiya/UNSW-NB15/resolve/main/UNSW_NB15_testing-set.csv",
        "https://raw.githubusercontent.com/Nir-J/ML-Projects/master/UNSW-Network_Packet_Classification/UNSW_NB15_testing-set.csv",
    ],
}

REQUIRED_COLUMNS = ("attack_cat", "proto", "service", "state", "dur")


def download(url: str, target: Path) -> bool:
    """Fetch one URL to a temporary file, then move it into place."""
    tmp = target.with_suffix(".part")
    request = urllib.request.Request(
        url, headers={"User-Agent": "nids-maml/0.1 (research)"}
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            if response.status != 200:
                log.warning("  %s -> HTTP %s", url, response.status)
                return False
            payload = response.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        log.warning("  %s -> %s", url, exc)
        return False

    if len(payload) < 1_000_000:
        log.warning("  %s -> only %d bytes, treating as a failure",
                    url, len(payload))
        return False
    tmp.write_bytes(payload)
    tmp.replace(target)
    return True


def verify(path: Path) -> tuple[bool, str]:
    """Check the file parses and carries the columns episodes depend on."""
    try:
        import pandas as pd
        head = pd.read_csv(path, nrows=200)
    except Exception as exc:  # noqa: BLE001 - any parse failure is a failure
        return False, f"unreadable: {type(exc).__name__}: {exc}"

    columns = {c.strip().lower() for c in head.columns}
    missing = [c for c in REQUIRED_COLUMNS if c.lower() not in columns]
    if missing:
        return False, (f"missing column(s) {missing}; this looks like the "
                       "binary-label-only variant, which cannot support "
                       "N-way episodes")
    return True, f"{len(head.columns)} columns"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="directory to write into")
    parser.add_argument("--force", action="store_true",
                        help="re-download even if a valid file is present")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    failures = []

    for name, urls in MIRRORS.items():
        target = out / name
        if target.exists() and not args.force:
            ok, detail = verify(target)
            if ok:
                log.info("%s already present (%s, %.1f MB)",
                         name, detail, target.stat().st_size / 1e6)
                continue
            log.warning("%s present but unusable (%s); re-downloading", name, detail)

        log.info("fetching %s", name)
        for url in urls:
            if not download(url, target):
                continue
            ok, detail = verify(target)
            if ok:
                digest = hashlib.sha256(target.read_bytes()).hexdigest()[:16]
                log.info("  ok: %.1f MB, %s, sha256:%s",
                         target.stat().st_size / 1e6, detail, digest)
                break
            log.warning("  downloaded but %s", detail)
            target.unlink(missing_ok=True)
        else:
            failures.append(name)
            log.error("all mirrors failed for %s", name)

    if failures:
        log.error("could not obtain: %s", ", ".join(failures))
        log.error("Download manually from "
                  "https://research.unsw.edu.au/projects/unsw-nb15-dataset "
                  "and place the CSVs in %s", out)
        return 1

    # Report the class inventory, which is what decides whether episodes can
    # hold classes out at meta-test time.
    import pandas as pd
    frames = [pd.read_csv(out / name) for name in MIRRORS]
    combined = pd.concat(frames, ignore_index=True)
    counts = (combined["attack_cat"].fillna("Normal").replace("", "Normal")
              .value_counts())
    print(f"\n{len(combined):,} records over {len(counts)} classes\n")
    for name, count in counts.items():
        print(f"  {name:<18} {count:>8,}")
    print(f"\nfiles in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
