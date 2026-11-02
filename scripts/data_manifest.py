"""Write or check data/MANIFEST.csv: the SHA-256 hash and size of every raw data file.

Raw NSE files and the third-party minute bars are not committed (licence and
size), so the manifest is what lets anyone confirm they have exactly the
data our results came from.

    python scripts/data_manifest.py write    # after downloading, records the hashes
    python scripts/data_manifest.py check    # verifies data/raw against the committed manifest

`check` fails if any listed file is missing or has a different hash (for .gz files the
hash is of the decompressed content, because gzip headers carry a time stamp). Files on
disk that are not in the manifest are reported but do not fail the check.
The minute-bar datasets are pinned by git commit in data/raw/minute_bars_source.json,
which is itself in the manifest.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import sys
from pathlib import Path

import pandas as pd

RAW = Path("data/raw")
MANIFEST = Path("data/MANIFEST.csv")


SKIP = {"download_misses.csv"}  # download logs, not data


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    """Hash of the file's content; for .gz files, of the decompressed content.

    A gzip header records a time stamp, so the same data downloaded twice gives
    different .gz bytes. Hashing what is inside makes the check about the data.
    """
    h = hashlib.sha256()
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def scan(raw: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(p for p in raw.rglob("*") if p.is_file() and p.name not in SKIP):
        rows.append({"path": f.relative_to(raw).as_posix(), "bytes": f.stat().st_size, "sha256": sha256(f)})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["write", "check"])
    ap.add_argument("--raw", default=str(RAW))
    ap.add_argument("--manifest", default=str(MANIFEST))
    a = ap.parse_args()
    raw, man = Path(a.raw), Path(a.manifest)
    if a.mode == "write":
        df = scan(raw)
        df.to_csv(man, index=False)
        print(f"{man}: {len(df)} files, {df['bytes'].sum() / 1e6:.1f} MB")
        return 0
    want = pd.read_csv(man)
    have = scan(raw).set_index("path")
    missing, changed = [], []
    for r in want.itertuples():
        if r.path not in have.index:
            missing.append(r.path)
        elif have.at[r.path, "sha256"] != r.sha256:
            changed.append(r.path)
    extra = sorted(set(have.index) - set(want["path"]))
    print(f"manifest: {len(want)} files; missing {len(missing)}, different {len(changed)}, extra on disk {len(extra)}")
    for name, lst in (("missing", missing), ("different", changed), ("extra (ignored)", extra)):
        for p in lst[:20]:
            print(f"  {name}: {p}")
        if len(lst) > 20:
            print(f"  ... and {len(lst) - 20} more {name}")
    return 1 if (missing or changed) else 0


if __name__ == "__main__":
    sys.exit(main())
