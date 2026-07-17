"""
Fashionpedia dataset setup -- fully automatic, no manual download.

Pulls the "detection-datasets/fashionpedia" mirror via the Hugging Face
`datasets` library (script-friendly, no license click-through required,
unlike the original fashionpedia.github.io zip download), then converts it
into the same COCO-format annotation file `indexer/build_index.py` already
expects -- so the rest of the pipeline needs zero changes.

Note on segmentation: this HF mirror provides bounding boxes but not the
original pixel-level polygon/RLE masks. We write each box's rectangle as its
"segmentation" polygon, so `decode_mask()` in build_index.py still works
unchanged -- it just yields a rectangular mask instead of the garment's exact
silhouette. Color extraction (K-means over the masked region) is therefore
slightly less precise near garment edges than with the original release, but
this only matters at the margins -- category + dominant color are still
reliable. If you need pixel-exact masks, use the original release instead
(see OFFICIAL_SOURCE_INSTRUCTIONS below).

Usage:
    pip install datasets
    python data/download_fashionpedia.py --split val --limit 1000

This downloads the "val" split (1,158 images -- fits the assignment's
500-1,000 image requirement out of the box), saves images to data/images/,
and writes the COCO-format annotations to the path configured in config.py.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    CHROMA_PERSIST_DIR,
    FASHIONPEDIA_IMAGE_DIR,
    FASHIONPEDIA_ANNOTATION,
    GLOBAL_COLLECTION,
    REGION_COLLECTION,
)  # noqa: E402

# Order matches the HF ClassLabel encoding exactly (category id = list index).
CATEGORY_NAMES = [
    "shirt, blouse", "top, t-shirt, sweatshirt", "sweater", "cardigan", "jacket",
    "vest", "pants", "shorts", "skirt", "coat", "dress", "jumpsuit", "cape",
    "glasses", "hat", "headband, head covering, hair accessory", "tie", "glove",
    "watch", "belt", "leg warmer", "tights, stockings", "sock", "shoe",
    "bag, wallet", "scarf", "umbrella", "hood", "collar", "lapel", "epaulette",
    "sleeve", "pocket", "neckline", "buckle", "zipper", "applique", "bead",
    "bow", "flower", "fringe", "ribbon", "rivet", "ruffle", "sequin", "tassel",
]

OFFICIAL_SOURCE_INSTRUCTIONS = """
For pixel-exact segmentation masks instead of bbox-derived rectangles, use
the original release instead:
  1. https://fashionpedia.github.io/home/Fashionpedia_download.html
  2. Download "Validation/Test images" (val_test2020.zip) and
     "Instance & attributes annotations" (instances_attributes_val2020.json)
  3. Unzip images into data/images/, place the json at the path in config.py
"""


def download_and_convert(split: str = "val", limit: int = 1000):
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "Run `pip install datasets` first (Hugging Face datasets library)."
        )

    os.makedirs(FASHIONPEDIA_IMAGE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(FASHIONPEDIA_ANNOTATION), exist_ok=True)

    print(f"Downloading detection-datasets/fashionpedia [{split}] via Hugging Face...")
    ds = load_dataset("detection-datasets/fashionpedia", split=split)
    n = min(limit, len(ds)) if limit else len(ds)
    print(f"Converting {n} / {len(ds)} images to COCO format...")

    images, annotations = [], []
    ann_id = 0

    for i in range(n):
        row = ds[i]
        image_id = row["image_id"]
        pil_img = row["image"].convert("RGB")
        file_name = f"{image_id}.jpg"
        pil_img.save(os.path.join(FASHIONPEDIA_IMAGE_DIR, file_name), quality=90)

        images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": row["width"],
                "height": row["height"],
            }
        )

        objects = row["objects"]
        for cat, bbox in zip(objects["category"], objects["bbox"]):
            x, y, x2, y2 = bbox
            w, h = x2 - x, y2 - y
            if w <= 0 or h <= 0:
                continue
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": cat,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                    # Rectangle polygon standing in for the true mask -- see
                    # module docstring. decode_mask() in build_index.py
                    # consumes this format unchanged.
                    "segmentation": [[x, y, x + w, y, x + w, y + h, x, y + h]],
                }
            )
            ann_id += 1

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{n} images converted")

    coco_json = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": idx, "name": name} for idx, name in enumerate(CATEGORY_NAMES)],
    }
    with open(FASHIONPEDIA_ANNOTATION, "w") as f:
        json.dump(coco_json, f)

    print(f"Done. {len(images)} images -> {FASHIONPEDIA_IMAGE_DIR}")
    print(f"Annotations -> {FASHIONPEDIA_ANNOTATION}")
    print("Next: python -m indexer.build_index --limit", n)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--limit", type=int, default=1000)
    args = parser.parse_args()
    download_and_convert(split=args.split, limit=args.limit)
