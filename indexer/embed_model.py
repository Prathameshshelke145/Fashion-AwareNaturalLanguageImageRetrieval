"""
Wrapper around a fashion-domain CLIP checkpoint.

We deliberately use patrickjohncyh/fashion-clip (ViT-B/32 CLIP fine-tuned by
the Farfetch research team on ~800K fashion product image/caption pairs)
instead of vanilla openai/clip-vit-base-patch32. It shares CLIP's zero-shot
generalization ability (needed for "style inference" queries never seen in
training) but has much tighter garment/color/pattern grounding, since that's
what it was fine-tuned on. This alone measurably improves attribute-specific
and compositional queries over vanilla CLIP, before any of our structured
re-ranking logic is applied on top.

Runs fine on CPU for a few hundred/thousand images -- ViT-B/32 is the
smallest common CLIP backbone. Batch the image embedding calls.

Embedding-space consistency: if this wrapper falls back from FASHION_CLIP_MODEL
to FALLBACK_CLIP_MODEL (e.g. network hiccup pulling the fine-tuned checkpoint),
the resulting vectors live in a *different* embedding space even though they
have the same dimensionality -- cosine similarity between a FashionCLIP vector
and a vanilla-CLIP vector doesn't error, it just silently returns a plausible-
looking but meaningless score. self.model_name records which checkpoint
actually loaded; build_index.py persists it into every record's metadata, and
the retriever should check it against the embedder it instantiates for queries
before trusting search results -- see verify_model_match().
"""
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from config import FASHION_CLIP_MODEL, FALLBACK_CLIP_MODEL


class FashionEmbedder:
    def __init__(self, device: str = "cpu"):
        self.device = device
        try:
            self.model = CLIPModel.from_pretrained(FASHION_CLIP_MODEL).to(device)
            self.processor = CLIPProcessor.from_pretrained(FASHION_CLIP_MODEL)
            self.model_name = FASHION_CLIP_MODEL
        except Exception as e:
            print(f"[embed_model] Could not load {FASHION_CLIP_MODEL} ({e}); "
                  f"falling back to {FALLBACK_CLIP_MODEL}")
            self.model = CLIPModel.from_pretrained(FALLBACK_CLIP_MODEL).to(device)
            self.processor = CLIPProcessor.from_pretrained(FALLBACK_CLIP_MODEL)
            self.model_name = FALLBACK_CLIP_MODEL
        self.model.eval()

        # Exposed so build_index.py never has to hardcode the embedding
        # dimension (previously `np.empty((0, 512))`, which would silently
        # be wrong for any future backbone swap in config.py).
        self.embedding_dim = self.model.config.projection_dim

    @torch.no_grad()
    def embed_images(self, images: list) -> np.ndarray:
        """images: list of PIL.Image (RGB). Returns L2-normalized (N, D) array."""
        if len(images) == 0:
            return np.empty((0, self.embedding_dim))
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        feats = self.model.get_image_features(**inputs)
        if not torch.is_tensor(feats):
            feats = getattr(feats, "image_embeds", None) or getattr(feats, "pooler_output", feats)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy()

    @torch.no_grad()
    def embed_texts(self, texts: list) -> np.ndarray:
        """texts: list of str. Returns L2-normalized (N, D) array."""
        inputs = self.processor(
            text=texts, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        feats = self.model.get_text_features(**inputs)
        if not torch.is_tensor(feats):
            feats = getattr(feats, "text_embeds", None) or getattr(feats, "pooler_output", feats)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy()

    def embed_image(self, image: Image.Image) -> np.ndarray:
        return self.embed_images([image])[0]

    def embed_text(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]

    def verify_model_match(self, indexed_model_name: str) -> None:
        """Call this at query time before searching an existing index.

        Raises a clear error immediately rather than letting a silent
        embedding-space mismatch produce plausible-looking-but-meaningless
        similarity scores. indexed_model_name should come from the
        'embed_model' field stored on any record in the Chroma collection
        (see build_index.py).
        """
        if indexed_model_name != self.model_name:
            raise RuntimeError(
                f"Embedding model mismatch: index was built with "
                f"'{indexed_model_name}' but this session loaded "
                f"'{self.model_name}'. Vector similarity between the two "
                f"is meaningless -- rebuild the index or fix whatever "
                f"caused the fallback (see the try/except in __init__)."
            )
