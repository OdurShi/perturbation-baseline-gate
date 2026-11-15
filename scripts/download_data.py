#!/usr/bin/env python
"""Download the Norman et al. 2019 Perturb-seq dataset (scPerturb release).

The file is cached outside version control: by default in ``data/raw/`` (which is
git-ignored), or in the folder given by ``--cache-dir`` / the ``PBGATE_CACHE``
environment variable. The download is skipped when a file with the expected size
and SHA-256 is already present.

Source: scPerturb Zenodo record 13350497 (version 1.4), licence CC BY 4.0
    https://zenodo.org/records/13350497
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
import urllib.request
from pathlib import Path

RECORD_URL = "https://zenodo.org/records/13350497"
FILE_NAME = "NormanWeissman2019_filtered.h5ad"
FILE_URL = f"https://zenodo.org/api/records/13350497/files/{FILE_NAME}/content"
FILE_SIZE = 698_680_199  # bytes, as listed on the Zenodo record
FILE_MD5 = "c870e6967d91c017d9da827bab183cd6"  # as listed on the Zenodo record
FILE_SHA256 = "efde6f5301fe256725dce1d980f37bd96a13481a9a16135515897368e631affc"  # computed on download

REPO_ROOT = Path(__file__).resolve().parents[1]


def default_cache_dir() -> Path:
    return Path(os.environ.get("PBGATE_CACHE", REPO_ROOT / "data" / "raw"))


def file_digests(path: Path, chunk: int = 1 << 22) -> tuple[str, str]:
    """Return (md5, sha256) hex digests of a file."""
    md5, sha = hashlib.md5(), hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            md5.update(block)
            sha.update(block)
    return md5.hexdigest(), sha.hexdigest()


def verify(path: Path) -> bool:
    if not path.exists() or path.stat().st_size != FILE_SIZE:
        return False
    md5, sha = file_digests(path)
    return md5 == FILE_MD5 and sha == FILE_SHA256


def download(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    request = urllib.request.Request(FILE_URL, headers={"User-Agent": "python-urllib"})
    start = time.time()
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as out:
        done = 0
        while block := response.read(1 << 22):
            out.write(block)
            done += len(block)
            print(f"\r{done / 1e6:8.1f} / {FILE_SIZE / 1e6:.1f} MB", end="", file=sys.stderr)
    print(f"\ndownloaded in {time.time() - start:.0f} s", file=sys.stderr)
    partial.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache-dir", type=Path, default=default_cache_dir())
    args = parser.parse_args()
    path = args.cache_dir / FILE_NAME
    if verify(path):
        print(f"cached and verified: {path}")
        return
    print(f"downloading {FILE_URL}\n        -> {path}")
    download(path)
    md5, sha = file_digests(path)
    print(f"size   {path.stat().st_size}\nmd5    {md5}\nsha256 {sha}")
    if path.stat().st_size != FILE_SIZE or md5 != FILE_MD5 or sha != FILE_SHA256:
        raise SystemExit("checksum mismatch: the downloaded file is not the expected one")
    print("verified")


if __name__ == "__main__":
    main()
