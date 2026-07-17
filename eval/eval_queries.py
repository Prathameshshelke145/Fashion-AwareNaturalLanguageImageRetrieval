"""
Runs the 5 required evaluation queries end-to-end and prints/saves the top-k
results with score breakdowns, plus optionally renders a contact-sheet image
per query for visual inspection.

Usage:
    python -m eval.eval_queries --top_k 5 --save_sheets
"""
import argparse
import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retriever.search import FashionRetriever

EVAL_QUERIES = [
    "A person in a bright yellow raincoat.",
    "Professional business attire inside a modern office.",
    "Someone wearing a blue shirt sitting on a park bench.",
    "Casual weekend outfit for a city walk.",
    "A red tie and a white shirt in a formal setting.",
]

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")


def make_contact_sheet(image_paths: list, title: str, out_path: str, thumb=(180, 220)):
    n = len(image_paths)
    sheet = Image.new("RGB", (thumb[0] * n, thumb[1] + 40), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((5, 5), title[:120], fill="black")
    for i, p in enumerate(image_paths):
        try:
            im = Image.open(p).convert("RGB")
            im.thumbnail(thumb)
            sheet.paste(im, (i * thumb[0], 40))
        except Exception as e:
            draw.text((i * thumb[0] + 5, 45), f"error: {e}", fill="red")
    sheet.save(out_path)


def run(top_k: int = 5, save_sheets: bool = False):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    retriever = FashionRetriever()

    for i, query in enumerate(EVAL_QUERIES, 1):
        print(f"\n=== Query {i}: {query} ===")
        results, parsed = retriever.search(query, top_k=top_k, explain=True)
        print(f"Parsed: {parsed}")
        for rank, r in enumerate(results, 1):
            print(
                f"  {rank}. score={r['score']:.3f} "
                f"(global={r['global_sim']:.3f} comp={r['compositional_score']:.3f} "
                f"env={r['environment_match']:.2f})  {os.path.basename(r['image_path'])}"
            )
        if save_sheets:
            out_path = os.path.join(OUTPUT_DIR, f"query_{i}.png")
            make_contact_sheet([r["image_path"] for r in results], query, out_path)
            print(f"  saved contact sheet -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--save_sheets", action="store_true")
    args = parser.parse_args()
    run(top_k=args.top_k, save_sheets=args.save_sheets)
