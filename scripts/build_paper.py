"""Build docs/paper.pdf from docs/paper.tex.

Runs pdflatex twice so that cross-references, the table of figures, and
the hyperref outline all resolve on the second pass. Produces only the
.pdf as a tracked artefact; all intermediate files (.aux, .log, .out)
are deleted on success (and are gitignored anyway).

Requires a working TeX distribution on PATH (MiKTeX / TeX Live).

Run from the repo root:

    python scripts/build_paper.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR = os.path.join(REPO_ROOT, "docs")
TEX_FILE = "paper.tex"
PDF_FILE = "paper.pdf"
AUX_EXTS = (".aux", ".log", ".out", ".toc", ".synctex.gz", ".fls",
            ".fdb_latexmk", ".bbl", ".blg")


def _check_pdflatex() -> str:
    exe = shutil.which("pdflatex")
    if exe is None:
        sys.stderr.write(
            "error: pdflatex not found on PATH. Install MiKTeX or TeX Live "
            "and make sure `pdflatex` is on PATH.\n"
        )
        sys.exit(1)
    return exe


def _run_pdflatex(exe: str, pass_num: int) -> None:
    print(f"[build_paper] pdflatex pass {pass_num}/2 ...")
    result = subprocess.run(
        [exe, "-interaction=nonstopmode", "-halt-on-error", TEX_FILE],
        cwd=DOCS_DIR,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        # On failure echo the tail of pdflatex output so the user can see
        # the actual LaTeX error.
        tail = "\n".join(result.stdout.splitlines()[-40:])
        sys.stderr.write(
            f"error: pdflatex failed on pass {pass_num} with exit code "
            f"{result.returncode}.\n--- pdflatex output (last 40 lines) "
            f"---\n{tail}\n"
        )
        sys.exit(result.returncode)


def _cleanup_aux() -> None:
    removed = []
    for name in os.listdir(DOCS_DIR):
        base, ext = os.path.splitext(name)
        if base != "paper":
            continue
        # .synctex.gz has a compound extension.
        compound = name[len("paper"):]
        if compound in AUX_EXTS or ext in AUX_EXTS:
            path = os.path.join(DOCS_DIR, name)
            try:
                os.remove(path)
                removed.append(name)
            except OSError:
                pass
    if removed:
        print(f"[build_paper] cleaned: {', '.join(removed)}")


def main() -> int:
    tex_path = os.path.join(DOCS_DIR, TEX_FILE)
    if not os.path.exists(tex_path):
        sys.stderr.write(f"error: {tex_path} not found.\n")
        return 1
    exe = _check_pdflatex()
    _run_pdflatex(exe, 1)
    _run_pdflatex(exe, 2)
    _cleanup_aux()
    pdf_path = os.path.join(DOCS_DIR, PDF_FILE)
    if not os.path.exists(pdf_path):
        sys.stderr.write("error: pdflatex completed but paper.pdf was not produced.\n")
        return 1
    size_kb = os.path.getsize(pdf_path) / 1024
    print(f"\n[build_paper] OK -> {pdf_path} ({size_kb:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
