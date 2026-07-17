"""
Parses a free-text query into the same structured slots we indexed:
    {
      "garments": [{"color": "yellow", "category": "raincoat"}, ...],
      "environment": "park" | None,
      "style_hints": ["casual", ...],
      "raw_text": "..."
    }

This is intentionally a lightweight rule-based parser (phrase matching +
positional adjective-noun binding) rather than an LLM call or a dependency
parser -- it needs zero extra model downloads, runs in microseconds, and for
short fashion-query sentences ("a red tie and a white shirt") standard
English adjective-before-noun order is a very strong, reliable signal. The
README's "future work" section covers swapping this for a small LLM-based
slot-filler if query complexity grows (e.g. relative clauses, negation).

The key trick for compositionality: we match the LONGEST vocabulary phrases
first (e.g. "bright yellow" and "bow tie" before "yellow" and "tie") so
multi-word colors/garments aren't fragmented, then bind each garment to the
nearest preceding unclaimed color within a small window.
"""
import re

from indexer.attribute_taxonomy import COLOR_NAME_TO_RGB, ENVIRONMENTS, STYLE_HINTS

GARMENT_ALIASES = {
    "shirt": "shirt", "button-down": "shirt", "button down": "shirt",
    "blouse": "shirt", "t-shirt": "t-shirt", "tee": "t-shirt",
    "sweater": "sweater", "cardigan": "cardigan", "jacket": "jacket",
    "raincoat": "raincoat", "coat": "coat", "blazer": "blazer",
    "vest": "vest", "hoodie": "hoodie", "dress": "dress",
    "jumpsuit": "jumpsuit", "pants": "pants", "trousers": "pants",
    "jeans": "jeans", "shorts": "shorts", "skirt": "skirt",
    "leggings": "leggings", "suit": "suit", "bow tie": "bow tie",
    "tie": "tie", "scarf": "scarf", "glasses": "glasses", "hat": "hat",
    "cap": "cap", "gloves": "gloves", "shoe": "shoe", "shoes": "shoe",
    "boot": "boot", "boots": "boot", "sneaker": "sneaker",
    "sneakers": "sneaker", "bag": "bag", "belt": "belt", "watch": "watch",
}

ENVIRONMENT_KEYWORDS = {
    "office": ["office", "business", "corporate", "workplace", "boardroom"],
    "urban street": ["street", "city", "urban", "sidewalk", "downtown", "city walk"],
    "park": ["park", "bench", "outdoors", "garden", "trees"],
    "home": ["home", "indoors", "living room", "bedroom", "house", "apartment"],
}

# Sort vocab by phrase length (word count) descending so multi-word terms win.
_COLOR_PHRASES = sorted(COLOR_NAME_TO_RGB.keys(), key=lambda s: -len(s.split()))
_GARMENT_PHRASES = sorted(GARMENT_ALIASES.keys(), key=lambda s: -len(s.split()))


def _tag_phrases(text: str, phrases: list, tag: str) -> str:
    """Replace occurrences of vocab phrases with a placeholder token so later
    tokenization treats e.g. 'bright yellow' as one unit."""
    for phrase in phrases:
        pattern = r"\b" + re.escape(phrase) + r"\b"
        token = f"__{tag}_{phrase.replace(' ', '~')}__"
        text = re.sub(pattern, token, text)
    return text


def parse_query(query: str) -> dict:
    text = query.lower()
    text = re.sub(r"[^\w\s'-]", " ", text)  # strip punctuation so it doesn't stick to tokens
    text = _tag_phrases(text, _COLOR_PHRASES, "COLOR")
    text = _tag_phrases(text, _GARMENT_PHRASES, "GARMENT")

    tokens = text.split()

    garments = []
    pending_color = None
    for i, tok in enumerate(tokens):
        if tok.startswith("__COLOR_"):
            pending_color = tok[len("__COLOR_"):-2].replace("~", " ")
        elif tok.startswith("__GARMENT_"):
            garment_raw = tok[len("__GARMENT_"):-2].replace("~", " ")
            category = GARMENT_ALIASES.get(garment_raw, garment_raw)
            garments.append({"color": pending_color, "category": category})
            pending_color = None  # consumed
        # plain words (and/with/a/in/...) don't reset pending_color across
        # short conjunctions like "a red tie and a white shirt"

    # Environment: keyword scan over the ORIGINAL lowercase text
    orig_lower = query.lower()
    environment = None
    for env, keywords in ENVIRONMENT_KEYWORDS.items():
        if any(kw in orig_lower for kw in keywords):
            environment = env
            break

    style_hints = [s for s in STYLE_HINTS if s in orig_lower]

    return {
        "garments": garments,
        "environment": environment,
        "style_hints": style_hints,
        "raw_text": query,
    }


if __name__ == "__main__":
    tests = [
        "A person in a bright yellow raincoat.",
        "Professional business attire inside a modern office.",
        "Someone wearing a blue shirt sitting on a park bench.",
        "Casual weekend outfit for a city walk.",
        "A red tie and a white shirt in a formal setting.",
    ]
    for t in tests:
        print(t)
        print(" ->", parse_query(t))
        print()
