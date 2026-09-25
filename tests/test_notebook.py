"""The Colab driver is the only entry point most runs go through, and a broken
cell is not discovered until someone is sitting in front of Colab waiting. This
checks the notebook is valid JSON and that every cell which is plain Python
compiles.

Run with:  python -m tests.test_notebook
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[1] / "colab_driver.ipynb"


def is_shell_cell(source: list[str]) -> bool:
    """IPython magics and shell escapes are not valid Python on their own."""
    return any(line.lstrip().startswith(("!", "%")) for line in source)


def main() -> int:
    failures: list[str] = []

    if not NOTEBOOK.exists():
        print(f"FAIL  notebook missing: {NOTEBOOK}")
        return 1

    try:
        nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"FAIL  notebook is not valid JSON: {exc}")
        return 1
    print("ok    notebook is valid JSON")

    checked = 0
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", [])
        if is_shell_cell(source):
            continue
        try:
            compile("".join(source), f"<cell {i}>", "exec")
            checked += 1
        except SyntaxError as exc:
            failures.append(f"cell {i}: {exc.msg} (line {exc.lineno})")

    if failures:
        for f in failures:
            print(f"FAIL  {f}")
    else:
        print(f"ok    all {checked} pure-Python cells compile")

    # The driver must point at the real repository, not the placeholder.
    text = NOTEBOOK.read_text(encoding="utf-8")
    if "<YOUR-USERNAME>" in text:
        failures.append("notebook still contains the <YOUR-USERNAME> placeholder")
        print("FAIL  notebook still contains the <YOUR-USERNAME> placeholder")
    else:
        print("ok    repository URL is filled in")

    print()
    print(f"{'FAILED' if failures else 'PASSED'}: {len(failures)} problem(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
