"""
Zero-shot environment/scene classification for the whole image.

Unlike garment color (where pixel statistics beat CLIP), scene context
("studio" vs "runway" vs "street style" vs "editorial") is exactly the kind
of holistic, global visual concept CLIP-style embeddings are good at --
there's no cheaper deterministic signal available. So here we DO use
zero-shot similarity: embed each candidate environment's description with
the same FashionCLIP text tower used everywhere else, and pick the closest
to the image embedding.

This is intentionally a separate, small vocabulary rather than open free
text, so we can store it as an exact-match metadata field in Chroma (cheap
to filter on) while still being derived zero-shot, with no labeled training
data or fine-tuning required. See attribute_taxonomy.py for why the
vocabulary is {studio, runway, street style, editorial} rather than generic
lifestyle scenes.
"""
import numpy as np

from indexer.attribute_taxonomy import ENVIRONMENTS

# Below this cosine similarity, none of the candidate labels are a confident
# enough match -- report "unknown" instead of forcing the image into the
# least-wrong bucket. Zero-shot CLIP similarities for a genuine match on
# short scene descriptions typically land noticeably higher than this; a
# score below it usually means the image doesn't clearly resemble ANY label
# (product flat-lay, extreme close-up, unusual crop, etc.).
CONFIDENCE_THRESHOLD = 0.20


class EnvironmentClassifier:
    def __init__(self, embedder, confidence_threshold: float = CONFIDENCE_THRESHOLD):
        self.embedder = embedder
        self.labels = list(ENVIRONMENTS.keys())
        self.label_embeddings = embedder.embed_texts(list(ENVIRONMENTS.values()))
        self.confidence_threshold = confidence_threshold

    def classify(self, global_image_embedding: np.ndarray) -> tuple:
        """Returns (best_label, similarity_score).

        best_label is "unknown" if the top similarity doesn't clear
        self.confidence_threshold -- callers should treat "unknown" as a
        valid, expected value (e.g. exclude it from environment filters at
        query time) rather than an error case.
        """
        sims = self.label_embeddings @ global_image_embedding
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])
        if best_score < self.confidence_threshold:
            return "unknown", best_score
        return self.labels[best_idx], best_score
