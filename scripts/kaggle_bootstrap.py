"""One-cell bootstrap for running this project on Kaggle Notebooks.

Paste the contents of this file into a Kaggle notebook cell, or run

    !curl -sL https://raw.githubusercontent.com/ayushexploring/nids-maml/main/scripts/kaggle_bootstrap.py | python -

Kaggle is used because its GPU quota is separate from Colab's (30 hours a week
against Colab's opaque free-tier allowance), and because UNSW-NB15 downloads
itself here -- no dataset upload is needed for the experiments that matter most.

Two settings must be enabled in the notebook's sidebar before this works:
  Settings -> Accelerator -> GPU T4 x2 (or P100)
  Settings -> Internet -> On        (required for the clone and the download)

Results are written to /kaggle/working, which persists when the notebook is
saved, and can be downloaded from the Output tab.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

REPO_URL = "https://github.com/ayushexploring/nids-maml.git"
REPO = "/kaggle/working/nids-maml"
RESULTS = "/kaggle/working/results"
CACHE = "/kaggle/working/cache"
UNSW = "/kaggle/working/unsw"


def run(*args: str, cwd: str | None = None, check: bool = True) -> str:
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=True)
    if check and result.returncode:
        raise RuntimeError(f"{' '.join(args)} failed:\n{result.stdout}{result.stderr}")
    return result.stdout.strip()


def main() -> int:
    for path in (RESULTS, CACHE, UNSW):
        os.makedirs(path, exist_ok=True)

    # --- code ---------------------------------------------------------------
    os.chdir("/kaggle/working")
    if os.path.isdir(os.path.join(REPO, ".git")):
        run("git", "-C", REPO, "fetch", "--all", "--quiet")
        run("git", "-C", REPO, "reset", "--hard", "origin/main")
        print("updated existing clone")
    else:
        shutil.rmtree(REPO, ignore_errors=True)
        run("git", "clone", "-q", "-b", "main", REPO_URL, REPO)
        print("cloned fresh")
    os.chdir(REPO)
    sys.path.insert(0, REPO)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyyaml"], check=False)
    print("commit  :", run("git", "-C", REPO, "log", "-1", "--format=%h %s"))

    # --- accelerator --------------------------------------------------------
    try:
        import torch
        if torch.cuda.is_available():
            print("GPU     :", torch.cuda.get_device_name(0))
        else:
            print("GPU     : NONE -- Settings > Accelerator > GPU, then re-run")
            return 1
    except ImportError:
        print("torch not importable; is this a Kaggle Python image?")
        return 1

    # --- data ---------------------------------------------------------------
    # UNSW-NB15 fetches itself, so the experiments that matter need no upload.
    # CIC-IDS2017 is not public in the cleaned form used here; attach it as a
    # Kaggle dataset and point --data-path at /kaggle/input/... when needed.
    have_unsw = any(f.endswith(".csv") for f in os.listdir(UNSW))
    if not have_unsw:
        print("\nfetching UNSW-NB15 ...")
        proc = subprocess.run(
            [sys.executable, "scripts/get_unsw.py", "--out", UNSW],
            cwd=REPO, capture_output=True, text=True,
        )
        print(proc.stdout[-2000:])
        if proc.returncode:
            print(proc.stderr[-1500:])
            print("download failed -- attach UNSW-NB15 as a Kaggle dataset instead")
            return 1
    else:
        print("UNSW    : already present")

    print("\nresults :", len([f for f in os.listdir(RESULTS) if f.endswith(".json")]),
          "files in", RESULTS)
    print("\nReady. Launch runs with, for example:\n")
    print(f"  !cd {REPO} && python -u -m nids_maml.run \\\n"
          f"      --config configs/unsw_proto_medoid.yaml \\\n"
          f"      --data-path {UNSW} --output-dir {RESULTS} \\\n"
          f"      --cache-dir {CACHE} --tag unsw_medoid_seed0 --seed 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
