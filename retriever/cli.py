"""
Command-line entry point.

Usage:
    python -m retriever.cli "A person in a bright yellow raincoat." --top_k 5
"""
import argparse

from retriever.search import FashionRetriever


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--explain", action="store_true")
    args = parser.parse_args()

    retriever = FashionRetriever()
    if args.explain:
        results, parsed = retriever.search(args.query, top_k=args.top_k, explain=True)
        print(f"Parsed query -> {parsed}\n")
    else:
        results = retriever.search(args.query, top_k=args.top_k)

    for rank, r in enumerate(results, 1):
        print(
            f"{rank:>2}. score={r['score']:.3f} "
            f"(global={r['global_sim']:.3f} comp={r['compositional_score']:.3f} "
            f"env={r['environment_match']:.2f}/{r['environment']})  {r['image_path']}"
        )


if __name__ == "__main__":
    main()
