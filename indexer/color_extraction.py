"""
Deterministic, cheap color extraction for a garment region.

Why not just ask CLIP "what color is this"? Zero-shot CLIP color
classification is noticeably worse than a direct pixel-statistics approach
for garments -- lighting, shadows and CLIP's coarse color vocabulary all hurt
it. Since Fashionpedia gives us pixel-accurate segmentation masks for free,
we use them: K-means on the masked pixels finds the dominant color cluster,
then we snap it to the nearest name in our fashion color taxonomy. This runs
in milliseconds on CPU and is far more reliable than a learned classifier for
this sub-task.
"""
import numpy as np
from sklearn.cluster import KMeans

from indexer.attribute_taxonomy import COLOR_NAME_TO_RGB

# A cluster smaller than this fraction of total pixels is treated as noise
# (stray background pixels, mask edge artifacts, a stitching highlight) and
# is never selected as the "dominant" color even if it happens to be the
# most saturated one.
MIN_CLUSTER_FRACTION = 0.10


def dominant_rgb(image_np: np.ndarray, mask: np.ndarray = None, k: int = 3) -> tuple:
    """
    image_np: HxWx3 uint8 RGB array (already cropped to the garment's bbox is fine)
    mask: optional HxW boolean array (True = pixel belongs to the garment).
          If provided, only those pixels are clustered -- this is what lets us
          ignore background bleeding into a bounding-box crop.
    Returns the dominant (R, G, B) tuple, ignoring near-black/near-white
    shadow & highlight clusters when a more saturated, sufficiently-large
    cluster is available.
    """
    pixels = image_np.reshape(-1, 3).astype(np.float32)
    if mask is not None:
        flat_mask = mask.reshape(-1)
        pixels = pixels[flat_mask]

    if len(pixels) < k:
        # Degenerate tiny region -- just average it.
        return tuple(int(v) for v in pixels.mean(axis=0))

    km = KMeans(n_clusters=min(k, len(pixels)), n_init=3, random_state=0)
    labels = km.fit_predict(pixels)
    counts = np.bincount(labels)
    centers = km.cluster_centers_
    total = counts.sum()

    # Rank clusters by size, largest first.
    order = np.argsort(-counts)
    largest_idx = order[0]
    min_count = MIN_CLUSTER_FRACTION * total

    # Walk clusters largest-to-smallest and return the first one that is
    # BOTH (a) not pure shadow/highlight and (b) not a tiny noise cluster.
    # This is the actual fix: previously the loop's fallback condition
    # (`counts[idx] == counts[order[0]]`) was trivially true on the very
    # first iteration, so the function always returned the largest cluster
    # regardless of brightness/saturation -- the shadow-skip never fired.
    for idx in order:
        if counts[idx] < min_count:
            continue  # too small to trust, e.g. a sliver of mask-edge bleed
        r, g, b = centers[idx]
        brightness = (r + g + b) / 3
        saturation = max(r, g, b) - min(r, g, b)
        is_shadow_or_highlight = brightness <= 25 or brightness >= 235
        if not is_shadow_or_highlight or saturation > 40:
            return tuple(int(v) for v in centers[idx])

    # Nothing cleared the bar (e.g. every large cluster is genuinely black,
    # white, or gray -- a real black jacket, not a shadow artifact). Fall
    # back to the largest cluster overall rather than returning nothing.
    return tuple(int(v) for v in centers[largest_idx])


def nearest_color_name(rgb: tuple) -> str:
    """Snap an (R, G, B) tuple to the closest name in our fashion color vocabulary."""
    best_name, best_dist = None, float("inf")
    for name, ref_rgb in COLOR_NAME_TO_RGB.items():
        dist = sum((a - b) ** 2 for a, b in zip(rgb, ref_rgb))
        if dist < best_dist:
            best_dist, best_name = dist, name
    return best_name


def extract_color_name(image_np: np.ndarray, mask: np.ndarray = None) -> str:
    rgb = dominant_rgb(image_np, mask)
    return nearest_color_name(rgb)
