"""
Part A: The Indexer.

For every image in the Fashionpedia subset:
  1. Compute ONE global embedding for the whole scene (captures environment +
     overall style/vibe -- handles queries like "casual weekend outfit").
  2. For EACH annotated garment instance in that image, use its ground-truth
     segmentation mask (Fashionpedia gives us this for free, so we don't need
     to run our own detector -- a deliberate CPU-budget decision, see README)
     to:
       a. crop the garment region
       b. extract its dominant color deterministically (color_extraction.py)
       c. embed the crop with FashionCLIP
     This region-level embedding + exact color/category metadata is what
     later resolves compositional queries ("red tie AND white shirt") that a
     single global embedding cannot.
  3. Zero-shot classify the image's environment (studio / runway / street
     style / editorial / unknown -- see environment_classifier.py).
  4. Persist everything to Chroma: one collection for global image vectors,
     one for garment-region vectors, both with rich metadata for filtering.
     Every record also stores which embedding model produced it
     (embedder.model_name), so the retriever can call
     FashionEmbedder.verify_model_match() before trusting a search -- this
     catches the case where indexing used FashionCLIP but a later query-time
     process silently fell back to vanilla CLIP.

Usage:
    python -m indexer.build_index --limit 800
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from tqdm import tqdm

from config import (
    CHROMA_PERSIST_DIR,
    FASHIONPEDIA_ANNOTATION,
    FASHIONPEDIA_IMAGE_DIR,
    GLOBAL_COLLECTION,
    REGION_COLLECTION,
)
from indexer.attribute_taxonomy import FASHIONPEDIA_CATEGORY_MAP
from indexer.color_extraction import extract_color_name
from indexer.embed_model import FashionEmbedder
from indexer.environment_classifier import EnvironmentClassifier


def decode_mask(coco: COCO, ann: dict, img_h: int, img_w: int) -> np.ndarray:
    """Fashionpedia stores segmentation as polygons or RLE; normalize to a bool mask."""
    seg = ann["segmentation"]
    if isinstance(seg, list):
        rles = mask_utils.frPyObjects(seg, img_h, img_w)
        rle = mask_utils.merge(rles)
    elif isinstance(seg["counts"], list):
        rle = mask_utils.frPyObjects(seg, img_h, img_w)
    else:
        rle = seg
    return mask_utils.decode(rle).astype(bool)


def build_index(limit: int = 800, batch_size: int = 16):
    if not os.path.exists(FASHIONPEDIA_ANNOTATION):
        raise FileNotFoundError(
            f"Annotations not found at {FASHIONPEDIA_ANNOTATION}.\n"
            "Run `python data/download_fashionpedia.py` first (see its docstring "
            "for the manual download steps if the network needs auth)."
        )

    print("Loading Fashionpedia annotations...")
    coco = COCO(FASHIONPEDIA_ANNOTATION)
    cat_id_to_name = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}

    embedder = FashionEmbedder()
    env_classifier = EnvironmentClassifier(embedder)
    print(f"Using embedding model: {embedder.model_name} (dim={embedder.embedding_dim})")

    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    global_collection = client.get_or_create_collection(
        GLOBAL_COLLECTION, metadata={"hnsw:space": "cosine"}
    )
    region_collection = client.get_or_create_collection(
        REGION_COLLECTION, metadata={"hnsw:space": "cosine"}
    )

    image_ids = coco.getImgIds()[:limit]
    print(f"Indexing {len(image_ids)} images...")

    for img_id in tqdm(image_ids):
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(FASHIONPEDIA_IMAGE_DIR, img_info["file_name"])
        if not os.path.exists(img_path):
            continue

        pil_img = Image.open(img_path).convert("RGB")
        np_img = np.array(pil_img)
        h, w = img_info["height"], img_info["width"]

        # ---- Global (whole-image) embedding ----
        global_emb = embedder.embed_image(pil_img)
        env_label, env_score = env_classifier.classify(global_emb)

        # ---- Per-garment region embeddings ----
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        crops, crop_meta = [], []
        for ann in anns:
            raw_cat = cat_id_to_name.get(ann["category_id"], "")
            category = FASHIONPEDIA_CATEGORY_MAP.get(raw_cat)
            if category is None:
                continue  # skip categories outside our retrieval vocabulary

            x, y, bw, bh = [int(v) for v in ann["bbox"]]
            if bw < 5 or bh < 5:
                continue
            crop_np = np_img[y : y + bh, x : x + bw]
            if crop_np.size == 0:
                continue

            try:
                mask = decode_mask(coco, ann, h, w)[y : y + bh, x : x + bw]
                color_name = extract_color_name(crop_np, mask if mask.any() else None)
            except Exception:
                color_name = extract_color_name(crop_np)

            crops.append(Image.fromarray(crop_np))
            crop_meta.append(
                {
                    "image_id": str(img_id),
                    "image_path": img_path,
                    "category": category,
                    "color": color_name,
                    "bbox": f"{x},{y},{bw},{bh}",
                    "embed_model": embedder.model_name,
                }
            )

        region_embs = embedder.embed_images(crops)

        # ---- Persist global record ----
        global_collection.upsert(
            ids=[str(img_id)],
            embeddings=[global_emb.tolist()],
            metadatas=[
                {
                    "image_path": img_path,
                    "environment": env_label,
                    "environment_score": env_score,
                    "garment_categories": ",".join(sorted({m["category"] for m in crop_meta})),
                    "garment_colors": ",".join(sorted({m["color"] for m in crop_meta})),
                    "embed_model": embedder.model_name,
                }
            ],
        )

        # ---- Persist region records ----
        if len(crops) > 0:
            region_collection.upsert(
                ids=[f"{img_id}_{i}" for i in range(len(crops))],
                embeddings=[e.tolist() for e in region_embs],
                metadatas=crop_meta,
            )

    print(f"Done. Indexed to {CHROMA_PERSIST_DIR}")
    print(f"  {GLOBAL_COLLECTION}: {global_collection.count()} images")
    print(f"  {REGION_COLLECTION}: {region_collection.count()} garment regions")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=800, help="Number of images to index")
    args = parser.parse_args()
    build_index(limit=args.limit)
