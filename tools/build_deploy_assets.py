"""Build the small bundle the hosted app needs, from a full local data/ directory.

    python -m tools.build_deploy_assets ../photo-scorer-app/data deploy_bundle

Copies models/ and processed/, and packs 200px thumbnails of the training photos
into thumbs.zip (one <post id>.jpg per row of clustered.csv).
"""

import shutil
import sys
import zipfile
from io import BytesIO
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm

THUMB_PX = 200


def main(src: Path, out: Path) -> None:
    for sub in ("models", "processed"):
        shutil.copytree(src / sub, out / sub, dirs_exist_ok=True)

    df = pd.read_csv(src / "processed" / "clustered.csv")
    missing = 0
    with zipfile.ZipFile(out / "thumbs.zip", "w", zipfile.ZIP_STORED) as z:  # JPEGs don't compress
        for pid, lp in tqdm(zip(df["id"], df["local_path"]), total=len(df)):
            path = src / "raw" / "images" / Path(lp.replace("\\", "/")).name
            try:
                img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
            except Exception:
                missing += 1
                continue
            img.thumbnail((THUMB_PX, THUMB_PX))
            buf = BytesIO()
            img.save(buf, "JPEG", quality=70)
            z.writestr(f"{pid}.jpg", buf.getvalue())
    print(f"done, {missing} of {len(df)} images unreadable/missing")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
