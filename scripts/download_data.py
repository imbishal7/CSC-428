"""One-time ASCAD download. Idempotent: skips files that already exist with the right size.

Run from the repo root:

    python scripts/download_data.py              # essentials (~4.6 GB compressed)
    python scripts/download_data.py --all        # also pull variable-key desync variants

Final layout under data/:
    ASCAD.h5                            fixed key, synchronized      (source / training)
    ASCAD_desync50.h5                   fixed key, +/- 50 sample shift
    ASCAD_desync100.h5                  fixed key, +/- 100 sample shift
    ASCAD_variable.h5                   variable key, synchronized
    [--all only]
    ASCAD_variable_desync50.h5
    ASCAD_variable_desync100.h5
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

# Confirmed in ANSSI README (ATM_AES_v1_fixed_key/Readme.md)
FIXED_KEY_ZIP_URL = "https://www.data.gouv.fr/api/1/datasets/r/e7ab6f9e-79bf-431f-a5ed-faf0ebe9b08e"

# From ATM_AES_v1_variable_key/Readme.md. If any of these 404, check the README on
# https://github.com/ANSSI-FR/ASCAD for refreshed UUIDs and update here.
VARIABLE_KEY_FILES = {
    "ASCAD_variable.h5":          "https://www.data.gouv.fr/api/1/datasets/r/b4ace767-c2a4-4db4-8e01-4527b5b91f00",
    "ASCAD_variable_desync50.h5": "https://www.data.gouv.fr/api/1/datasets/r/4ad6d44a-f6de-483f-807f-d0ccab76d2a9",
    "ASCAD_variable_desync100.h5":"https://www.data.gouv.fr/api/1/datasets/r/f1936388-71be-408f-b8ec-472bb3398e39",
}

FIXED_KEY_EXTRACTS = ["ASCAD.h5", "ASCAD_desync50.h5", "ASCAD_desync100.h5"]


def _download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  skip (exists): {dest.name} ({dest.stat().st_size / 1e9:.2f} GB)")
        return
    print(f"  downloading -> {dest.name}")
    tmp = dest.with_suffix(dest.suffix + ".part")

    def hook(blocks, block_size, total_size):
        done = blocks * block_size
        if total_size > 0:
            pct = min(100.0, 100.0 * done / total_size)
            sys.stdout.write(f"\r    {pct:5.1f}%  {done/1e9:.2f} / {total_size/1e9:.2f} GB")
        else:
            sys.stdout.write(f"\r    {done/1e9:.2f} GB")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, tmp, reporthook=hook)
    sys.stdout.write("\n")
    tmp.rename(dest)


def _extract_fixed_key(zip_path: Path, data_dir: Path) -> None:
    if all((data_dir / name).exists() for name in FIXED_KEY_EXTRACTS):
        print("  fixed-key .h5 files already extracted, skipping unzip")
        return
    print(f"  unzipping {zip_path.name} ...")
    staging = data_dir / "_fixed_key_unzip"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(staging)
    moved = 0
    for h5 in staging.rglob("*.h5"):
        if h5.name in FIXED_KEY_EXTRACTS:
            target = data_dir / h5.name
            if not target.exists():
                shutil.move(str(h5), str(target))
                moved += 1
    shutil.rmtree(staging, ignore_errors=True)
    print(f"  extracted {moved} files")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--all", action="store_true",
                    help="also download variable-key desync variants (extra ~840 MB)")
    ap.add_argument("--keep-zip", action="store_true",
                    help="keep the fixed-key zip after extracting (default: delete to save 4.2 GB)")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    print("[1/2] fixed-key bundle (ASCAD.h5, ASCAD_desync50.h5, ASCAD_desync100.h5)")
    zip_path = data_dir / "ASCAD_fixed_key.zip"
    if not all((data_dir / n).exists() for n in FIXED_KEY_EXTRACTS):
        _download(FIXED_KEY_ZIP_URL, zip_path)
        _extract_fixed_key(zip_path, data_dir)
        if not args.keep_zip and zip_path.exists():
            zip_path.unlink()
            print("  removed zip")
    else:
        print("  all fixed-key .h5 files already present")

    print("[2/2] variable-key h5 files")
    targets = list(VARIABLE_KEY_FILES.items())
    if not args.all:
        targets = targets[:1]   # only the synchronized variable-key file by default
    for fname, url in targets:
        _download(url, data_dir / fname)

    print("\nDone. Files under", data_dir.resolve())
    for f in sorted(data_dir.glob("*.h5")):
        print(f"  {f.name:36s}  {f.stat().st_size / 1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
