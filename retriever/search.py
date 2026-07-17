"""
Part B: The Retriever.

Retrieval is a two-stage process:

  Stage 1 (recall) -- embed the full raw query with FashionCLIP's text tower
  and pull a broad candidate pool from the GLOBAL image collection by cosine
  similarity. This stage is what gives the system its zero-shot ability: for
  a "style inference" query like "casual weekend outfit for a city walk",
  there is no fixed attribute to match against, so we lean entirely on the
  dense embedding's learned notion of "casual".

  Stage 2 (compositional rerank) -- this is what actually gets us past
  vanilla CLIP. We parse the query into structured (color, garment) pairs and
  an environment slot. For each candidate image, we look up its indexed
  GARMENT_REGIONS and score how well the *specific* parsed pairs are
  satisfied by *specific* regions in that image (via the Hungarian algorithm
  when there are multiple garments, so "red tie" can't double-count as
  matching both a tie region AND a shirt region). This directly solves the
  "red tie AND white shirt" vs "white tie AND red shirt" binding problem that
  a single global embedding cannot represent.

Final score = weighted sum of (global similarity, compositional match,
environment match). Weights are in config.py.
"""
import chromadb
import numpy as np
from scipy.optimize import linear_sum_assignment

from config import (
    CHROMA_PERSIST_DIR,
    GLOBAL_CANDIDATE_POOL,
    GLOBAL_COLLECTION,
    REGION_COLLECTION,
    TOP_K_DEFAULT,
    WEIGHT_COMPOSITIONAL,
    WEIGHT_ENVIRONMENT,
    WEIGHT_GLOBAL_SIM,
)
from indexer.embed_model import FashionEmbedder
from retriever.query_parser import parse_query


class FashionRetriever:
    def __init__(self):
        self.embedder = FashionEmbedder()
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        self.global_collection = client.get_collection(GLOBAL_COLLECTION)
        self.region_collection = client.get_collection(REGION_COLLECTION)

    # -- Stage 2 helper -----------------------------------------------------
    def _compositional_score(self, image_id: str, parsed_garments: list) -> float:
        """
        Fetches all indexed garment regions for this image and finds the best
        one-to-one assignment between parsed (color, garment) query slots and
        actual regions, using exact category match (required) and color match
        (bonus) -- then Hungarian-assigns them so each region is used at most
        once. Returns the mean per-slot match score in [0, 1].
        """
        if not parsed_garments:
            return 0.0

        results = self.region_collection.get(
            where={"image_id": image_id}, include=["metadatas"]
        )
        regions = results["metadatas"]
        if not regions:
            return 0.0

        n_slots, n_regions = len(parsed_garments), len(regions)
        cost = np.ones((n_slots, n_regions))  # 1 = no match, lower = better

        for i, slot in enumerate(parsed_garments):
            for j, region in enumerate(regions):
                if slot["category"] != region["category"]:
                    continue  # category must match -- this is the hard constraint
                if slot["color"] is None:
                    cost[i, j] = 0.3  # category-only match, decent but not exact
                elif slot["color"] == region["color"]:
                    cost[i, j] = 0.0  # exact color + category match
                else:
                    cost[i, j] = 0.6  # right garment, wrong color -- partial credit

        row_idx, col_idx = linear_sum_assignment(cost)
        matched_costs = cost[row_idx, col_idx]
        # pad unmatched slots (more query garments than detected regions) as full miss
        total_cost = matched_costs.sum() + (n_slots - len(row_idx))
        mean_cost = total_cost / n_slots
        return 1.0 - mean_cost

    # -- Public API -----------------------------------------------------
    def search(self, query: str, top_k: int = TOP_K_DEFAULT, explain: bool = False):
        parsed = parse_query(query)

        # Stage 1: broad recall via global embedding
        query_emb = self.embedder.embed_text(query)
        raw = self.global_collection.query(
            query_embeddings=[query_emb.tolist()],
            n_results=GLOBAL_CANDIDATE_POOL,
            include=["metadatas", "distances"],
        )

        # Redistribute weight away from slots the query didn't actually specify,
        # so a pure "vibe" query (no garments/environment parsed) isn't unfairly
        # capped by a compositional/environment score that's 0 for every candidate.
        w_global, w_comp, w_env = WEIGHT_GLOBAL_SIM, WEIGHT_COMPOSITIONAL, WEIGHT_ENVIRONMENT
        if not parsed["garments"]:
            w_global += w_comp
            w_comp = 0.0
        if parsed["environment"] is None:
            w_global += w_env
            w_env = 0.0

        candidates = []
        for img_id, meta, dist in zip(
            raw["ids"][0], raw["metadatas"][0], raw["distances"][0]
        ):
            global_sim = 1 - dist  # chroma cosine distance -> similarity
            comp_score = self._compositional_score(img_id, parsed["garments"])

            env_score = 0.0
            if parsed["environment"] is not None:
                env_score = 1.0 if meta.get("environment") == parsed["environment"] else 0.0
            else:
                env_score = 0.5  # neutral when query didn't specify environment

            final_score = w_global * global_sim + w_comp * comp_score + w_env * env_score

            candidates.append(
                {
                    "image_id": img_id,
                    "image_path": meta["image_path"],
                    "score": final_score,
                    "global_sim": global_sim,
                    "compositional_score": comp_score,
                    "environment": meta.get("environment"),
                    "environment_match": env_score,
                }
            )

        candidates.sort(key=lambda c: -c["score"])
        top = candidates[:top_k]

        if explain:
            return top, parsed
        return top
