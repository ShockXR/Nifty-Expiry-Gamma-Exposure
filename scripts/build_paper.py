"""Build the working paper: Markdown to standalone HTML (pandoc) to PDF (Chromium).

    python scripts/build_paper.py              # writes docs/WORKING_PAPER.html and docs/WORKING_PAPER.pdf
    python scripts/build_paper.py --html-only  # skip the PDF step

The draft itself (docs/WORKING_PAPER.*) is git-ignored: it is kept outside this public repository
until the author decides how to share it.

Steps:

1. pandoc converts docs/WORKING_PAPER.md to one self-contained HTML file
   (math as MathML, figures and the style sheet embedded), so the HTML can be
   opened or mailed on its own.
2. Playwright's Chromium prints that HTML to an A4 PDF with page numbers.

Requirements: pandoc (``--pandoc`` or the PANDOC environment variable if it
is not at /usr/bin/pandoc or on PATH) and the Python package ``playwright``
with a Chromium build already present (for example through
PLAYWRIGHT_BROWSERS_PATH). This script never downloads a browser.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "WORKING_PAPER.md"
HTML = ROOT / "docs" / "WORKING_PAPER.html"
PDF = ROOT / "docs" / "WORKING_PAPER.pdf"
TITLE = "Expiry Days, Dealer Gamma and the Close (working paper draft)"

CSS = """
:root { --ink: #1a1a1a; --muted: #555; --rule: #bbb; --head: #f2f2f2; }
body {
  font-family: "Charter", "Georgia", "DejaVu Serif", "Liberation Serif", serif;
  font-size: 10pt; color: var(--ink); background: #fff; line-height: 1.4;
  max-width: 46em; margin: 2em auto; padding: 0 1em;
  text-align: left; hyphens: auto;
}
h1 { font-size: 1.55em; line-height: 1.25; margin: 0 0 0.6em; }
h2 { font-size: 1.25em; margin: 1.6em 0 0.5em; break-after: avoid; }
h3 { font-size: 1.08em; margin: 1.3em 0 0.4em; break-after: avoid; }
h4 { font-size: 1em; margin: 1.1em 0 0.3em; break-after: avoid; }
p { margin: 0.55em 0; orphans: 3; widows: 3; }
a { color: inherit; }
code { font-family: "DejaVu Sans Mono", "Liberation Mono", monospace; font-size: 0.82em;
       overflow-wrap: anywhere; }
pre { background: #f6f6f6; padding: 0.6em 0.8em; white-space: pre-wrap; font-size: 0.95em;
      break-inside: avoid; }
pre code { font-size: 0.8em; }
table { border-collapse: collapse; width: 100%; margin: 0.6em 0 0.3em; font-size: 0.8em;
        line-height: 1.3; break-inside: auto; }
thead { display: table-header-group; }
tr { break-inside: avoid; }
th, td { border-top: 1px solid var(--rule); border-bottom: 1px solid var(--rule);
         padding: 0.22em 0.4em; vertical-align: top; text-align: left; }
th { background: var(--head); font-weight: 600; }
td code, th code { font-size: 0.9em; }
figure { margin: 1em 0; text-align: center; break-inside: avoid; }
img { max-width: 100%; max-height: 120mm; height: auto; }
/* keep a table or figure title with what follows it */
p:has(+ table), p:has(+ figure), p:has(+ math) { break-after: avoid; }
figcaption { font-size: 0.85em; color: var(--muted); margin-top: 0.3em; text-align: left; }
math[display="block"] { margin: 0.5em 0; overflow-x: auto; }
ol, ul { padding-left: 1.6em; }
li { margin: 0.25em 0; }
.footnotes { font-size: 0.85em; color: var(--muted); }
@media print {
  body { max-width: none; margin: 0; padding: 0; font-size: 10pt; }
  h2 { break-before: auto; }
}
"""


def find_pandoc(cli: str | None) -> str:
    for cand in (cli, os.environ.get("PANDOC"), "/usr/bin/pandoc", shutil.which("pandoc")):
        if cand and Path(cand).exists():
            return cand
    sys.exit("pandoc not found: pass --pandoc PATH or set PANDOC.")


def build_html(pandoc: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        header = Path(tmp) / "style.html"
        header.write_text(f"<style>{CSS}</style>\n", encoding="utf-8")
        cmd = [
            pandoc, str(SRC),
            "--from", "markdown",
            "--to", "html5",
            "--standalone",
            "--mathml",
            "--embed-resources",
            # figure paths in the Markdown are relative to docs/
            "--resource-path", f"{ROOT / 'docs'}{os.pathsep}{ROOT}",
            "--include-in-header", str(header),
            "--metadata", f"pagetitle={TITLE}",
            "--metadata", "lang=en-GB",
            "--output", str(HTML),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            sys.exit(f"pandoc failed:\n{res.stderr}")
        if res.stderr.strip():
            print(res.stderr.strip(), file=sys.stderr)
    print(f"wrote {HTML.relative_to(ROOT)} ({HTML.stat().st_size:,} bytes)")


def build_pdf() -> None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        sys.exit("The Python package 'playwright' is not installed (pip install playwright). "
                 "A Chromium build must already be available; this script does not download one.")
    footer = ('<div style="width:100%;font-size:8px;color:#666;text-align:center;'
              'font-family:serif;"><span class="pageNumber"></span> / <span class="totalPages"></span></div>')
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(HTML.as_uri(), wait_until="load")
        page.emulate_media(media="print")
        page.pdf(
            path=str(PDF),
            format="A4",
            margin={"top": "18mm", "bottom": "18mm", "left": "20mm", "right": "20mm"},
            print_background=True,
            display_header_footer=True,
            header_template="<span></span>",
            footer_template=footer,
        )
        browser.close()
    print(f"wrote {PDF.relative_to(ROOT)} ({PDF.stat().st_size:,} bytes)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pandoc", help="path to the pandoc executable")
    ap.add_argument("--html-only", action="store_true", help="build the HTML and stop")
    args = ap.parse_args()
    if not SRC.exists():
        sys.exit(f"missing {SRC}")
    build_html(find_pandoc(args.pandoc))
    if not args.html_only:
        build_pdf()


if __name__ == "__main__":
    main()
