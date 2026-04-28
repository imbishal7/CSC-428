"""Download the ASCAD fixed-key dataset (one-time, idempotent).

Pulls the official ANSSI bundle (~4.2 GB zip → ~7.3 GB extracted) and
extracts ASCAD.h5 into data/. Skips the download if the file is already
present. Run from the repo root:

    python scripts/download_data.py
"""

from __future__ import annotations

import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://www.data.gouv.fr/api/1/datasets/r/e7ab6f9e-79bf-431f-a5ed-faf0ebe9b08e"
DATA = Path("data")
TARGET = DATA / "ASCAD.h5"


def _progress(blocks, block_size, total_size):
    done = blocks * block_size
    if total_size > 0:
        sys.stdout.write(f"\r  {100 * done / total_size:5.1f}%  {done/1e9:.2f} / {total_size/1e9:.2f} GB")
    else:
        sys.stdout.write(f"\r  {done/1e9:.2f} GB")
    sys.stdout.flush()


def main() -> None:
    DATA.mkdir(exist_ok=True)
    if TARGET.exists():
        print(f"already present: {TARGET} ({TARGET.stat().st_size / 1e9:.2f} GB)")
        return

    zip_path = DATA / "ASCAD_data.zip"
    if not zip_path.exists():
        print(f"downloading -> {zip_path.name}")
        urllib.request.urlretrieve(URL, zip_path, reporthook=_progress)
        print()

    print("extracting ASCAD.h5 ...")
    staging = DATA / "_extract"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(staging)
    for h5 in staging.rglob("ASCAD.h5"):
        shutil.move(str(h5), str(TARGET))
        break
    shutil.rmtree(staging, ignore_errors=True)
    zip_path.unlink()
    print(f"done: {TARGET} ({TARGET.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
